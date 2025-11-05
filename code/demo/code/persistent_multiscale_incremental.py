# persistent_multiscale_incremental.py
# 完整版：带“第一次完整训练 + 后续小时增量训练”的 MultiScaleModelManager
# 以及一个可在 IDE 里直接运行的 main() 演示（滚动 k 小时）


from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# ============== 基础模型（与您原版一致，仅修正 Transformer 的 batch_first 逻辑） ==============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleModel(nn.Module):
    """
    LSTM(1d/1w) + Transformer(1m) → 特征融合 → 与 1h 拼接进 GRU → FC
    注意：Transformer使用 batch_first=True，因此不需要 permute。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTMs (daily & weekly)
        self.lstm_1d = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_1w = nn.LSTM(1, hidden_size, batch_first=True)

        # Transformer (monthly)
        self.input_projection = nn.Linear(1, hidden_size)
        self.transformer_1m = nn.Transformer(
            d_model=hidden_size, nhead=4, num_encoder_layers=2, batch_first=True
        )

        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_size * 3, hidden_size)

        # GRU on fused features + raw 1h
        self.gru = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)

        # Final regressor
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_1h, x_1d, x_1w, x_1m = x["1h"], x["1d"], x["1w"], x["1m"]  # [B, T, 1]

        # LSTMs
        _, (h_1d, _) = self.lstm_1d(x_1d)
        h_1d = h_1d[-1]  # [B, H]

        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]  # [B, H]

        # Transformer (batch_first=True → 输入 [B, T, H])
        x_1m = self.input_projection(x_1m)            # [B, Tm, H]
        h_seq = self.transformer_1m(x_1m, x_1m)       # [B, Tm, H]
        h_1m = h_seq[:, -1, :]                        # 取最后时间步 [B, H]

        # Fuse
        fused_trend = torch.cat([h_1d, h_1w, h_1m], dim=1)  # [B, 3H]
        fused_trend = self.feature_fusion(fused_trend)      # [B, H]

        # GRU input
        batch_size, seq_len, _ = x_1h.shape
        fused_trend_expanded = fused_trend.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, H]
        x_gru_input = torch.cat([x_1h, fused_trend_expanded], dim=2)           # [B, T, H+1]

        # GRU
        _, h_gru = self.gru(x_gru_input)
        h_gru = h_gru[-1]   # [B, H]

        # Output
        output = self.fc(h_gru)  # [B, 1]
        return output


# ============== 管理器：新增“增量训练”能力 ==============

@dataclass
class ManagerConfig:
    hidden_size: int = 64
    epochs_full: int = 50
    epochs_incremental: int = 1     # 每小时微调的 epoch 数（建议 1~2）
    patience_full: int = 5
    lr_full: float = 1e-3
    lr_incremental: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    sequence_length: int = 24 * 30  # 720h
    forecast_length: int = 1
    learning_rate = 0.001
    patience = 5

class MultiScaleModelManager:
    """
    - 第一次：完整训练到 target-1h，并保存 meta(last_trained_until)
    - 后续：仅用 (last_trained_until, target] 的新标签小时做微调（增量训练）
    """
    def __init__(self, checkpoint_dir: str = "checkpoints_multiscale", cfg: Optional[ManagerConfig] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ManagerConfig()
        self.durations = {'1h': 1, '1d': 24, '1w': 24 * 7, '1m': 24 * 30}

    # ---------- 路径 ----------
    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"multiscale_zone_{zone_id}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"scaler_zone_{zone_id}.pkl"

    def _meta_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"meta_zone_{zone_id}.json"

    # ---------- 状态 ----------
    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._scaler_path(zone_id).exists()

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        import pickle
        with open(self._scaler_path(zone_id), 'wb') as f:
            pickle.dump(scaler, f)

    def _load(self, zone_id: int) -> Tuple[nn.Module, MinMaxScaler]:
        model = MultiScaleModel(self.cfg.hidden_size).to(device)
        state = torch.load(self._model_path(zone_id), map_location=device)
        model.load_state_dict(state)
        model.eval()
        import pickle
        with open(self._scaler_path(zone_id), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    def _save_meta(self, zone_id: int, last_trained_until: pd.Timestamp):
        meta = {"last_trained_until": str(last_trained_until)}
        with open(self._meta_path(zone_id), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_meta(self, zone_id: int) -> Optional[pd.Timestamp]:
        p = self._meta_path(zone_id)
        if not p.exists():
            return None
        meta = json.load(open(p, "r", encoding="utf-8"))
        return pd.Timestamp(meta["last_trained_until"])

    # ---------- 数据准备 ----------
        hourly['passenger_count'] = hourly['passenger_count'].astype(float)
        return hourly  # 列: datetime, passenger_count
    # def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> pd.DataFrame:
    #     """Return full hourly series for the zone from (target_date - 1m) to (target_date - 1h)."""
    #     assert 'datetime' in df.columns and 'PULocationID' in df.columns
    #
    #     # Time window
    #     start_date = target_date - pd.Timedelta(hours=self.sequence_length)
    #
    #     zone_df = df[df['PULocationID'] == zone_id].copy()
    #     hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')
    #
    #     # Restrict to needed range and fill missing hours with zeros
    #     rng = pd.date_range(start=start_date, end=target_date, freq='H')
    #     hourly = (
    #         hourly.set_index('datetime')
    #         .reindex(rng)
    #         .fillna(0)
    #         .rename_axis('datetime')
    #         .reset_index()
    #     )
    #     hourly['passenger_count'] = hourly['passenger_count'].astype(float)
    #     return hourly

    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, end_inclusive: pd.Timestamp) -> pd.DataFrame:
        """Return full hourly series for the zone from (end_inclusive - sequence_length) to end_inclusive."""
        assert 'datetime' in df.columns and 'PULocationID' in df.columns

        # Time window
        start_date = end_inclusive - pd.Timedelta(hours=self.cfg.sequence_length)

        zone_df = df[df['PULocationID'] == zone_id].copy()
        hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')

        # Restrict to needed range and fill missing hours with zeros
        rng = pd.date_range(start=start_date, end=end_inclusive, freq='h')
        hourly = (
            hourly.set_index('datetime')
            .reindex(rng)
            .fillna(0)
            .rename_axis('datetime')
            .reset_index()
        )
        hourly['passenger_count'] = hourly['passenger_count'].astype(float).fillna(0.0)
        return hourly



    def _fit_scaler_hist(self, hourly: pd.DataFrame, fit_until_exclusive: pd.Timestamp) -> MinMaxScaler:
        """只用 < fit_until_exclusive 的历史拟合 scaler，避免未来泄露；含健壮性处理。"""
        scaler = MinMaxScaler()
        hist = hourly[hourly['datetime'] < fit_until_exclusive]
        values = hist[['passenger_count']].astype(float)

        if values.empty:
            scaler.fit(pd.DataFrame({'passenger_count': [0.0]}))
            return scaler

        vmin = float(values['passenger_count'].min())
        vmax = float(values['passenger_count'].max())

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            base = 0.0 if not np.isfinite(vmin) else vmin
            scaler.fit(pd.DataFrame({'passenger_count': [base, base + 1.0]}))
            return scaler

        scaler.fit(values)
        return scaler

    def _build_training_windows(self, hourly: pd.DataFrame, scaler: MinMaxScaler | None = None):
        """Build sliding windows for training; returns tensors X_dict, y_tensor, scaler."""
        seq_len = self.cfg.sequence_length
        flen = self.cfg.forecast_length
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

    def _build_inference_window(self, hourly: pd.DataFrame, scaler: MinMaxScaler, target_date: pd.Timestamp) -> dict:
        """
        用 [target-720, target-1] 的 720 个点做输入；
        注意：仅 transform，不重新 fit，避免泄漏
        """
        dur = self.durations
        L = self.cfg.sequence_length

        df = hourly.copy()
        df['passenger_count_scaled'] = scaler.transform(df[['passenger_count']])

        # 找到 target 的行号
        idx = {t: i for i, t in enumerate(df['datetime'])}
        if target_date not in idx:
            raise ValueError("target_date 不在该区间序列内（请检查 DF 是否覆盖 target）")
        end_i = idx[target_date]           # 标签在 target_date
        start_i = end_i - L
        if start_i < 0:
            raise ValueError("历史不足 720 小时，无法构造推理窗口")

        series = df['passenger_count_scaled']
        x_1h = series.iloc[end_i - dur['1h']: end_i].values
        x_1d = series.iloc[end_i - dur['1d']: end_i].values
        x_1w = series.iloc[end_i - dur['1w']: end_i].values
        x_1m = series.iloc[start_i: end_i].values

        X = {
            '1h': torch.tensor(x_1h, dtype=torch.float32, device=device).view(1, -1, 1),
            '1d': torch.tensor(x_1d, dtype=torch.float32, device=device).view(1, -1, 1),
            '1w': torch.tensor(x_1w, dtype=torch.float32, device=device).view(1, -1, 1),
            '1m': torch.tensor(x_1m, dtype=torch.float32, device=device).view(1, -1, 1),
        }
        return X


    # ---------- 训练/预测 ----------
    def train_once(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> None:
        """Train and save model+scaler for a zone if no checkpoint exists."""
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, target_date)

        # Need enough history
        # if len(hourly) < self.sequence_length + self.forecast_length:
        #     raise ValueError(f"Zone {zone_id}: insufficient data for training. Got {len(hourly)} hours.")

        X_tensor, y_tensor, scaler = self._build_training_windows(hourly)

        model = MultiScaleModel(self.cfg.hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate)

        best_loss = float('inf')
        patience_ctr = 0

        model.train()
        for epoch in range(self.cfg.epochs_full):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            current = loss.item()
            if current < best_loss:
                best_loss = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience:
                    break

        # Save after training
        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, target_date)

    def incremental_update(
            self,
            df: pd.DataFrame,
            zone_id: int,
            prev_until: pd.Timestamp,
            new_until: pd.Timestamp,
            epochs: Optional[int] = None,
            lr: Optional[float] = None,
    ) -> None:
        """
        用 (prev_until, new_until] 的每个标签小时 s 做微调。
        对于每个 s：输入窗口 = [s-720, s-1]，标签 = s。
        在每次微调前，按 new_until 之前的历史重新拟合 scaler，并在训练后保存更新后的 scaler。
        """
        # 零步保护
        if new_until <= prev_until:
            return

        epochs = epochs if epochs is not None else self.cfg.epochs_incremental
        lr = lr if lr is not None else self.cfg.lr_incremental

        # 载入模型；scaler 不沿用旧的，按 new_until 重拟合
        model, _old_scaler = self._load(zone_id)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)
        criterion = nn.SmoothL1Loss()

        # 准备覆盖到 new_until 的完整序列；这里仅 transform，不再 fit
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=new_until)
        # 关键：每次微调前用 < new_until 的历史重拟合 scaler（可改为 14/30 天窗口，你已有思路）
        fit_start = new_until - pd.Timedelta(days=30)
        hourly_fit = hourly[(hourly['datetime'] < new_until) & (hourly['datetime'] >= fit_start)].copy()
        scaler = self._fit_scaler_hist(hourly_fit, fit_until_exclusive=new_until)
        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])

        idx = {t: i for i, t in enumerate(hourly['datetime'])}

        s_list = pd.date_range(start=prev_until + pd.Timedelta(hours=1), end=new_until, freq="H")
        dur = self.durations
        L = self.cfg.sequence_length
        series = hourly['passenger_count_scaled']

        X_1h, X_1d, X_1w, X_1m, Y = [], [], [], [], []
        for s in s_list:
            end_i = idx.get(s, None)
            if end_i is None:
                continue
            start_i = end_i - L
            if start_i < 0:
                continue
            if end_i + self.cfg.forecast_length > len(series):
                continue

            X_1h.append(series.iloc[end_i - dur['1h']: end_i].values)
            X_1d.append(series.iloc[end_i - dur['1d']: end_i].values)
            X_1w.append(series.iloc[end_i - dur['1w']: end_i].values)
            X_1m.append(series.iloc[start_i: end_i].values)
            Y.append(series.iloc[end_i: end_i + self.cfg.forecast_length].values)

        if not X_1m:  # 没有新增样本（如数据缺失）
            self._save(zone_id, model, scaler)
            self._save_meta(zone_id, prev_until)
            return

        X = {
            '1h': torch.tensor(np.array(X_1h), dtype=torch.float32, device=device).unsqueeze(-1),
            '1d': torch.tensor(np.array(X_1d), dtype=torch.float32, device=device).unsqueeze(-1),
            '1w': torch.tensor(np.array(X_1w), dtype=torch.float32, device=device).unsqueeze(-1),
            '1m': torch.tensor(np.array(X_1m), dtype=torch.float32, device=device).unsqueeze(-1),
        }
        Y = torch.tensor(np.array(Y), dtype=torch.float32, device=device)

        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

        # 保存微调后权重 + 更新后的 scaler + meta
        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, new_until)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        """
        用保存的模型对 target_date 小时做预测（输入为 [target-720, target-1]）。
        预测前按 target_date 之前的历史重拟合 scaler，不使用旧的 scaler。
        """
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} 没有 checkpoint，请先训练")

        model, _old_scaler = self._load(zone_id)  # 只用模型，scaler 重新拟合
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=target_date)
        scaler = self._fit_scaler_hist(hourly, fit_until_exclusive=target_date)

        X_last = self._build_inference_window(hourly, scaler, target_date)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_last).cpu().numpy()  # shape (1,1)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        return float(pred)

    def train_and_predict_if_needed(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp,
                                    auto_train: bool = True) -> float:
        """
        统一入口：
        - 若无 ckpt → 完整训练到 target-1h
        - 若有 ckpt → 对 (last_trained_until, target] 做增量微调
        - 然后预测 target
        """
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No checkpoint for zone {zone_id} and auto_train is False.")
            self.train_once(df, zone_id, target_date)
        else:
            prev = self._load_meta(zone_id)
            if prev is None:
                # 旧模型没有 meta，保守起见完整训一次
                self.train_once(df, zone_id, target_date)
            elif prev < target_date:
                print(f"[inc] zone={zone_id} {prev} -> {target_date}")
                self.incremental_update(df, zone_id, prev_until=prev, new_until=target_date)
            else:
                pass #已覆盖目标小时
        return self.predict(df, zone_id, target_date)


# def _prepare_df_from_parquet(parquet_path: str) -> pd.DataFrame:
#     """Helper to load the same columns and preprocessing as demo.py relies on."""
#     columns_to_load = ['pickup_datetime', 'PULocationID']
#     df = pd.read_parquet(parquet_path, columns=columns_to_load)
#     df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
#     df['datetime'] = df['pickup_datetime'].dt.floor('h')
#     return df
#
#
# if __name__ == "__main__":
#     # Example CLI usage (minimal):
#     #   python code/persistent_multiscale.py
#     # It will train once per zone present in the data and then produce a
#     # prediction for target_date for each zone without retraining.
#
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Train-once and predict with MultiScaleModel per zone.")
#     parser.add_argument("--data", type=str, default="data.parquet", help="Input parquet file path.")
#     parser.add_argument("--target", type=str, default="2021-03-05 12:00", help="Target timestamp (YYYY-mm-dd HH:MM)")
#     parser.add_argument("--hidden", type=int, default=64, help="Hidden size.")
#     parser.add_argument("--checkpoints", type=str, default="checkpoints_multiscale", help="Checkpoint directory.")
#     parser.add_argument("--zones", type=int, nargs="*", default=None, help="Optional list of PULocationID to process.")
#     parser.add_argument("--no-auto-train", action="store_true", help="Disable auto-train when checkpoint missing.")
#     args = parser.parse_args()
#
#     target_date = pd.Timestamp(args.target)
#     df = _prepare_df_from_parquet(args.data)
#
#     manager = MultiScaleModelManager(checkpoint_dir=args.checkpoints, hidden_size=args.hidden)
#
#     # Select zones
#     zones = args.zones if args.zones else sorted(df['PULocationID'].unique().tolist())
#
#     results = []
#     for zid in zones:
#         try:
#             pred = manager.train_and_predict_if_needed(df, zid, target_date, auto_train=not args.no_auto_train)
#             results.append({"PULocationID": zid, "Prediction": pred})
#             print(f"Zone {zid}: Prediction @ {target_date} = {pred:.4f}")
#         except Exception as e:
#             print(f"Zone {zid}: skipped due to error -> {e}")
#
#     if results:
#         out_csv = Path(args.checkpoints) / f"predictions_{target_date.strftime('%Y%m%d_%H%M')}.csv"
#         out_csv.parent.mkdir(parents=True, exist_ok=True)
#         pd.DataFrame(results).to_csv(out_csv, index=False)
#         print(f"Saved predictions to {out_csv}")
