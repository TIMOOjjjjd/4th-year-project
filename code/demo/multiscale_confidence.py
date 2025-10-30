import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleModel(nn.Module):
    """四分支（1h/1d/1w/1m）+ 分支辅助头 + MC Dropout 支持"""

    def __init__(self, hidden_size, p_drop=0.2):
        super(MultiScaleModel, self).__init__()

        self.hidden_size = hidden_size

        # LSTMs (daily & weekly)
        self.lstm_1d = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_1w = nn.LSTM(1, hidden_size, batch_first=True)

        # Transformer (monthly)
        self.input_projection = nn.Linear(1, hidden_size)
        self.transformer_1m = nn.Transformer(
            hidden_size, nhead=4, num_encoder_layers=2, batch_first=True
        )

        # ✨ 分支级 Dropout（用于 MC Dropout）
        self.do_1d = nn.Dropout(p_drop)
        self.do_1w = nn.Dropout(p_drop)
        self.do_1m = nn.Dropout(p_drop)
        self.do_1h = nn.Dropout(p_drop)

        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_size * 3, hidden_size)

        # GRU on fused features + raw 1h
        self.gru = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)

        # Final regressor（主头）
        self.fc = nn.Linear(hidden_size, 1)

        # ✨ 分支辅助头：从各自 embedding 直接回归 y
        self.head_1d = nn.Linear(hidden_size, 1)
        self.head_1w = nn.Linear(hidden_size, 1)
        self.head_1m = nn.Linear(hidden_size, 1)
        self.head_1h = nn.Linear(hidden_size, 1)

    # ---------------- 内部子流程 ----------------
    def _encode_1d_1w_1m(self, x):
        """编码 1d/1w/1m 分支，返回各自 embedding（已过 Dropout）"""
        x_1d, x_1w, x_1m = x["1d"], x["1w"], x["1m"]

        # LSTM 分支
        _, (h_1d, _) = self.lstm_1d(x_1d)
        h_1d = h_1d[-1]  # [B,H]
        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]

        # Transformer（月）
        x_1m_proj = self.input_projection(x_1m)  # [B,T,H]
        x_1m_proj = x_1m_proj.permute(1, 0, 2)   # [T,B,H]
        h_1m = self.transformer_1m(x_1m_proj, x_1m_proj)[-1]  # 取最后时刻 [B,H]

        # 分支级 Dropout
        h_1d = self.do_1d(h_1d)
        h_1w = self.do_1w(h_1w)
        h_1m = self.do_1m(h_1m)
        return h_1d, h_1w, h_1m

    def _build_gru_input(self, x, fused_trend):
        """把融合后的趋势与 1h 原序列拼接，喂给 GRU"""
        x_1h = x["1h"]  # [B,L,1]
        batch_size, seq_len, _ = x_1h.shape
        fused_trend_expanded = fused_trend.unsqueeze(1).repeat(1, seq_len, 1)  # [B,L,H]
        x_gru_input = torch.cat([x_1h, fused_trend_expanded], dim=2)  # [B,L,1+H]
        return x_gru_input

    # ---------------- 常规前向（训练/推理） ----------------
    def forward(self, x):
        """
        返回：
          y_main: [B,1]
          (y_1h, y_1d, y_1w, y_1m): 各分支辅助头输出
          以及各分支 embedding（供上层可选使用）
        """
        # 1d/1w/1m 编码
        h_1d, h_1w, h_1m = self._encode_1d_1w_1m(x)

        # 融合
        fused_trend = self.feature_fusion(torch.cat([h_1d, h_1w, h_1m], dim=1))  # [B,H]

        # GRU 主干 + 1h 分支 embedding（取最后隐藏态）
        x_gru_input = self._build_gru_input(x, fused_trend)
        gru_out, h_n = self.gru(x_gru_input)           # gru_out: [B,L,H], h_n: [1,B,H]
        h_1h = gru_out[:, -1, :]                        # [B,H] 作为 1h 分支 embedding
        h_1h = self.do_1h(h_1h)                         # 分支级 Dropout

        # 主头输出
        y_main = self.fc(h_n[-1])                       # [B,1]

        # 分支辅助输出
        y_1h = self.head_1h(h_1h)
        y_1d = self.head_1d(h_1d)
        y_1w = self.head_1w(h_1w)
        y_1m = self.head_1m(h_1m)

        return y_main, (y_1h, y_1d, y_1w, y_1m), (h_1h, h_1d, h_1w, h_1m)

    # ---------------- MC 前向：获取分支 embedding 的样本 ----------------
    def mc_branch_embeddings(self, x, K: int):
        """
        用于训练/诊断：做 K 次带 Dropout 的前向，收集四个分支的 embedding 样本。
        返回：
          emb_1h/1d/1w/1m: [K,B,H]
        """
        self.train()  # 关键：激活 Dropout
        emb_1h, emb_1d, emb_1w, emb_1m = [], [], [], []
        with torch.no_grad():
            for _ in range(K):
                # 只要拿到 embedding 就行
                _, _, (h_1h, h_1d, h_1w, h_1m) = self.forward(x)
                emb_1h.append(h_1h.unsqueeze(0))
                emb_1d.append(h_1d.unsqueeze(0))
                emb_1w.append(h_1w.unsqueeze(0))
                emb_1m.append(h_1m.unsqueeze(0))
        return (torch.cat(emb_1h, 0),
                torch.cat(emb_1d, 0),
                torch.cat(emb_1w, 0),
                torch.cat(emb_1m, 0))

    # ---------------- MC 推理：整体输出的均值/std ----------------
    def mc_predict(self, x, M: int):
        """
        返回：
          mean: [B,1]  预测均值（缩放空间）
          std:  [B,1]  预测标准差（缩放空间）
        """
        self.train()  # 激活 Dropout
        preds = []
        with torch.no_grad():
            for _ in range(M):
                y_main, _, _ = self.forward(x)
                preds.append(y_main.unsqueeze(0))
        preds = torch.cat(preds, dim=0)  # [M,B,1]
        return preds.mean(dim=0), preds.std(dim=0)


class MultiScaleModelManager:
    """Train once per zone, save model+scaler, and reuse for inference."""

    def __init__(self,
                 checkpoint_dir: str = "checkpoints_multiscale",
                 hidden_size: int = 64,
                 p_drop: float = 0.2,
                 lambda_aux: float = 0.5,
                 K_mc_train: int = 5,
                 M_mc_test: int = 20):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size

        # Training hyperparams
        self.epochs = 50
        self.patience = 5
        self.learning_rate = 0.001

        # Uncertainty / MC 超参
        self.p_drop = p_drop
        self.lambda_aux = lambda_aux
        self.K_mc_train = K_mc_train
        self.M_mc_test = M_mc_test

        # Time scales
        self.durations = {
            '1h': 1,
            '1d': 24,
            '1w': 24 * 7,
            '1m': 24 * 30
        }
        self.forecast_length = 1
        self.sequence_length = self.durations['1m']

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"multiscale_zone_{zone_id}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"scaler_zone_{zone_id}.pkl"

    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._scaler_path(zone_id).exists()

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        with open(self._scaler_path(zone_id), 'wb') as f:
            pickle.dump(scaler, f)

    def _load(self, zone_id: int) -> tuple[nn.Module, MinMaxScaler]:
        model = MultiScaleModel(self.hidden_size, p_drop=self.p_drop).to(device)
        state = torch.load(self._model_path(zone_id), map_location=device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> pd.DataFrame:
        """Return full hourly series for the zone from (target_date - 1m) to (target_date - 1h)."""
        assert 'datetime' in df.columns and 'PULocationID' in df.columns

        # Time window
        start_date = target_date - pd.Timedelta(hours=self.sequence_length)

        zone_df = df[df['PULocationID'] == zone_id].copy()
        hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')

        # Restrict to needed range and fill missing hours with zeros
        rng = pd.date_range(start=start_date, end=target_date, freq='H')
        hourly = (
            hourly.set_index('datetime')
            .reindex(rng)
            .fillna(0)
            .rename_axis('datetime')
            .reset_index()
        )
        hourly['passenger_count'] = hourly['passenger_count'].astype(float)
        return hourly

    def _build_training_windows(self, hourly: pd.DataFrame, scaler: MinMaxScaler | None = None):
        """Build sliding windows for training; returns tensors X_dict, y_tensor, scaler."""
        seq_len = self.sequence_length
        flen = self.forecast_length
        durations = self.durations

        # Fit or reuse scaler (fit on entire history here like in demo path before target)
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(hourly[['passenger_count']])

        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])

        X_1h, X_1d, X_1w, X_1m, y = [], [], [], [], []
        # We need at least seq_len + flen samples to build one example
        for i in range(len(hourly) + 1 - seq_len - flen):
            x_1h = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1h']: i + seq_len].values
            x_1d = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1d']: i + seq_len].values
            x_1w = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1w']: i + seq_len].values
            x_1m = hourly['passenger_count_scaled'].iloc[i: i + seq_len].values

            X_1h.append(x_1h)
            X_1d.append(x_1d)
            X_1w.append(x_1w)
            X_1m.append(x_1m)

            y_val = hourly['passenger_count_scaled'].iloc[i + seq_len: i + seq_len + flen].values
            if len(y_val) == flen:
                y.append(y_val)

        X_1h = np.array(X_1h, dtype=np.float32)
        X_1d = np.array(X_1d, dtype=np.float32)
        X_1w = np.array(X_1w, dtype=np.float32)
        X_1m = np.array(X_1m, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X_tensor = {
            '1h': torch.tensor(X_1h, dtype=torch.float32).unsqueeze(-1).to(device),
            '1d': torch.tensor(X_1d, dtype=torch.float32).unsqueeze(-1).to(device),
            '1w': torch.tensor(X_1w, dtype=torch.float32).unsqueeze(-1).to(device),
            '1m': torch.tensor(X_1m, dtype=torch.float32).unsqueeze(-1).to(device),
        }
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        return X_tensor, y_tensor, scaler

    def _build_inference_window(self, hourly: pd.DataFrame, scaler: MinMaxScaler):
        durations = self.durations
        seq_len = self.sequence_length

        hourly = hourly.copy()
        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])

        # 丢掉 target_date 当小时，使用 [target-720h, target-1h]
        end = len(hourly) - 1
        start = end - seq_len

        x_1h = hourly['passenger_count_scaled'].iloc[end - durations['1h']: end].values
        x_1d = hourly['passenger_count_scaled'].iloc[end - durations['1d']: end].values
        x_1w = hourly['passenger_count_scaled'].iloc[end - durations['1w']: end].values
        x_1m = hourly['passenger_count_scaled'].iloc[start: end].values

        X_tensor = {
            '1h': torch.tensor(x_1h, dtype=torch.float32).view(1, -1, 1).to(device),
            '1d': torch.tensor(x_1d, dtype=torch.float32).view(1, -1, 1).to(device),
            '1w': torch.tensor(x_1w, dtype=torch.float32).view(1, -1, 1).to(device),
            '1m': torch.tensor(x_1m, dtype=torch.float32).view(1, -1, 1).to(device),
        }
        return X_tensor

    def train_once(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> None:
        """Train and save model+scaler for a zone if no checkpoint exists."""
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, target_date)
        X_tensor, y_tensor, scaler = self._build_training_windows(hourly)

        model = MultiScaleModel(self.hidden_size, p_drop=self.p_drop).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        best_loss = float('inf')
        patience_ctr = 0

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            # 主前向（单次）
            y_main, (y_1h, y_1d, y_1w, y_1m), _ = model(X_tensor)
            L_main = criterion(y_main, y_tensor)

            # === 分支不确定性（MC） ===
            with torch.no_grad():
                emb_1h, emb_1d, emb_1w, emb_1m = model.mc_branch_embeddings(
                    X_tensor, self.K_mc_train
                )  # [K,B,H]

                # 以隐向量维度求方差并对 H 做均值 → 每样本一个标量方差
                var_1h = emb_1h.var(dim=0).mean(dim=1, keepdim=True)  # [B,1]
                var_1d = emb_1d.var(dim=0).mean(dim=1, keepdim=True)
                var_1w = emb_1w.var(dim=0).mean(dim=1, keepdim=True)
                var_1m = emb_1m.var(dim=0).mean(dim=1, keepdim=True)

                eps = 1e-6
                conf_1h = 1.0 / (var_1h + eps)
                conf_1d = 1.0 / (var_1d + eps)
                conf_1w = 1.0 / (var_1w + eps)
                conf_1m = 1.0 / (var_1m + eps)

                conf_sum = conf_1h + conf_1d + conf_1w + conf_1m
                w_1h = conf_1h / conf_sum
                w_1d = conf_1d / conf_sum
                w_1w = conf_1w / conf_sum
                w_1m = conf_1m / conf_sum

            # 分支辅助损失（逐样本加权）
            L_1h = ((y_1h - y_tensor) ** 2) * w_1h
            L_1d = ((y_1d - y_tensor) ** 2) * w_1d
            L_1w = ((y_1w - y_tensor) ** 2) * w_1w
            L_1m = ((y_1m - y_tensor) ** 2) * w_1m
            L_aux = (L_1h + L_1d + L_1w + L_1m).mean()

            loss = L_main + self.lambda_aux * L_aux
            loss.backward()
            optimizer.step()

            current = loss.item()
            if current < best_loss:
                best_loss = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        # Save after training
        self._save(zone_id, model, scaler)

    def _inverse_std_minmax(self, std_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        """把缩放空间的 std 换算回原空间（MinMax 线性比例）。"""
        data_range = scaler.data_max_[0] - scaler.data_min_[0]
        return std_scaled * data_range

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        """Predict next hour demand for the zone at target_date using saved model (point estimate)."""
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(
                f"No checkpoint for zone {zone_id}. Call train_once() first or enable auto-train.")

        model, scaler = self._load(zone_id)
        hourly = self._prepare_zone_series(df, zone_id, target_date)
        X_last = self._build_inference_window(hourly, scaler)

        # 用 MC 均值作为点预测
        model.eval()
        with torch.no_grad():
            mean_scaled, _ = model.mc_predict(X_last, self.M_mc_test)  # [1,1]
        mean_np = mean_scaled.cpu().numpy()
        pred = scaler.inverse_transform(mean_np)[0, 0]
        return float(pred)

    def predict_with_uncertainty(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp):
        """
        返回 (point, std, branch_var_dict)
          - point: 单点预测（均值，原空间）
          - std:   标准差（原空间，基于 MC Dropout）
          - branch_var_dict: 分支 embedding 方差（用于诊断，越小越确定）
        """
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(
                f"No checkpoint for zone {zone_id}. Call train_once() first or enable auto-train.")

        model, scaler = self._load(zone_id)
        hourly = self._prepare_zone_series(df, zone_id, target_date)
        X_last = self._build_inference_window(hourly, scaler)

        model.eval()
        with torch.no_grad():
            mean_scaled, std_scaled = model.mc_predict(X_last, self.M_mc_test)  # [1,1] each
            # 分支不确定性（embedding 方差）
            emb_1h, emb_1d, emb_1w, emb_1m = model.mc_branch_embeddings(X_last, self.M_mc_test)
            var_1h = emb_1h.var(dim=0).mean().item()
            var_1d = emb_1d.var(dim=0).mean().item()
            var_1w = emb_1w.var(dim=0).mean().item()
            var_1m = emb_1m.var(dim=0).mean().item()

        mean_np = mean_scaled.cpu().numpy()
        std_np = std_scaled.cpu().numpy()
        point = scaler.inverse_transform(mean_np)[0, 0]
        std_orig = float(self._inverse_std_minmax(std_np)[0, 0])

        branch_var = {
            "1h": var_1h,
            "1d": var_1d,
            "1w": var_1w,
            "1m": var_1m,
        }
        return float(point), float(std_orig), branch_var

    def train_and_predict_if_needed(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp,
                                    auto_train: bool = True) -> float:
        """Convenience: train once if missing, then predict."""
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No checkpoint for zone {zone_id} and auto_train is False.")
            self.train_once(df, zone_id, target_date)
        return self.predict(df, zone_id, target_date)


def _prepare_df_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Helper to load the same columns and preprocessing as demo.py relies on."""
    columns_to_load = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(parquet_path, columns=columns_to_load)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['datetime'] = df['pickup_datetime'].dt.floor('H')
    return df


if __name__ == "__main__":
    # Example CLI usage (minimal):
    #   python code/persistent_multiscale.py
    # It will train once per zone present in the data and then produce a
    # prediction for target_date for each zone without retraining.

    import argparse

    parser = argparse.ArgumentParser(description="Train-once and predict with MultiScaleModel per zone.")
    parser.add_argument("--data", type=str, default="data.parquet", help="Input parquet file path.")
    parser.add_argument("--target", type=str, default="2021-03-05 12:00", help="Target timestamp (YYYY-mm-dd HH:MM)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size.")
    parser.add_argument("--checkpoints", type=str, default="checkpoints_multiscale", help="Checkpoint directory.")
    parser.add_argument("--zones", type=int, nargs="*", default=None, help="Optional list of PULocationID to process.")
    parser.add_argument("--no-auto-train", action="store_true", help="Disable auto-train when checkpoint missing.")
    args = parser.parse_args()

    target_date = pd.Timestamp(args.target)
    df = _prepare_df_from_parquet(args.data)

    manager = MultiScaleModelManager(checkpoint_dir=args.checkpoints, hidden_size=args.hidden)

    # Select zones
    zones = args.zones if args.zones else sorted(df['PULocationID'].unique().tolist())

    results = []
    for zid in zones:
        try:
            pred = manager.train_and_predict_if_needed(df, zid, target_date, auto_train=not args.no_auto_train)
            results.append({"PULocationID": zid, "Prediction": pred})
            print(f"Zone {zid}: Prediction @ {target_date} = {pred:.4f}")
        except Exception as e:
            print(f"Zone {zid}: skipped due to error -> {e}")

    if results:
        out_csv = Path(args.checkpoints) / f"predictions_{target_date.strftime('%Y%m%d_%H%M')}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")
