# persistent_multiscale_incremental.py
# 完整版：带“第一次完整训练 + 后续小时增量训练”的 MultiScaleModelManager
# 以及一个可在 IDE 里直接运行的 main() 演示（滚动 k 小时）

from __future__ import annotations
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
    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, end_inclusive: pd.Timestamp) -> pd.DataFrame:
        """返回该区从最早有数据到 end_inclusive 的完整逐小时序列（缺失补 0）"""
        assert {'datetime', 'PULocationID'}.issubset(df.columns)
        zone_df = df[df['PULocationID'] == zone_id].copy()
        hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')

        rng = pd.date_range(start=hourly['datetime'].min(), end=end_inclusive, freq='H') \
              if not hourly.empty else pd.date_range(end=end_inclusive, periods=self.cfg.sequence_length + 1, freq='H')
        hourly = (
            hourly.set_index('datetime')
                  .reindex(rng)
                  .fillna(0.0)
                  .rename_axis('datetime')
                  .reset_index()
        )
        hourly['passenger_count'] = hourly['passenger_count'].astype(float)
        return hourly  # 列: datetime, passenger_count

    def _fit_scaler_hist(self, hourly: pd.DataFrame, fit_until_exclusive: pd.Timestamp) -> MinMaxScaler:
        """只用 < fit_until_exclusive 的历史拟合 scaler，避免未来泄漏"""
        scaler = MinMaxScaler()
        hist = hourly[hourly['datetime'] < fit_until_exclusive]
        if hist.empty:
            # 若历史为空，退化为全量（避免报错）；实际场景可自定义
            hist = hourly
        scaler.fit(hist[['passenger_count']])
        return scaler

    def _build_windows(self, series_scaled: pd.Series) -> Tuple[dict, torch.Tensor]:
        """从一段连续的缩放后序列构造多样本滑窗（最后一步为标签）"""
        L = self.cfg.sequence_length
        flen = self.cfg.forecast_length
        dur = self.durations

        X_1h, X_1d, X_1w, X_1m, Y = [], [], [], [], []
        n = len(series_scaled)
        for i in range(n + 1 - L - flen):
            x_1h = series_scaled.iloc[i + L - dur['1h']: i + L].values
            x_1d = series_scaled.iloc[i + L - dur['1d']: i + L].values
            x_1w = series_scaled.iloc[i + L - dur['1w']: i + L].values
            x_1m = series_scaled.iloc[i: i + L].values
            y = series_scaled.iloc[i + L: i + L + flen].values  # flen=1

            X_1h.append(x_1h)
            X_1d.append(x_1d)
            X_1w.append(x_1w)
            X_1m.append(x_1m)
            Y.append(y)

        X = {
            '1h': torch.tensor(np.array(X_1h), dtype=torch.float32).unsqueeze(-1).to(device),
            '1d': torch.tensor(np.array(X_1d), dtype=torch.float32).unsqueeze(-1).to(device),
            '1w': torch.tensor(np.array(X_1w), dtype=torch.float32).unsqueeze(-1).to(device),
            '1m': torch.tensor(np.array(X_1m), dtype=torch.float32).unsqueeze(-1).to(device),
        }
        Y = torch.tensor(np.array(Y), dtype=torch.float32).to(device)
        return X, Y

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
        """第一次：用所有 < target 的历史构造多样本滑窗，完整训练，然后把 meta 记到 target-1h"""
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=target_date)
        scaler = self._fit_scaler_hist(hourly, fit_until_exclusive=target_date)

        # 仅用 < target 的数据构窗（最后标签 ≤ target-1h）
        hourly_hist = hourly[hourly['datetime'] < target_date].reset_index(drop=True)
        series_scaled = pd.Series(
            scaler.transform(hourly_hist[['passenger_count']]).squeeze(),
            index=hourly_hist.index
        )
        X, Y = self._build_windows(series_scaled)

        model = MultiScaleModel(self.cfg.hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr_full, weight_decay=self.cfg.weight_decay)

        best = float('inf'); patience = 0
        model.train()
        for ep in range(self.cfg.epochs_full):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

            cur = loss.item()
            if cur < best:
                best, patience = cur, 0
            else:
                patience += 1
                if patience >= self.cfg.patience_full:
                    break

        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, last_trained_until=target_date - pd.Timedelta(hours=1))

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
        """
        if new_until <= prev_until:
            return

        epochs = epochs if epochs is not None else self.cfg.epochs_incremental
        lr = lr if lr is not None else self.cfg.lr_incremental

        model, scaler = self._load(zone_id)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)
        criterion = nn.SmoothL1Loss()  # 稍稳健一些

        # 准备完整序列（覆盖到 new_until）并仅 transform
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=new_until)
        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])
        idx = {t: i for i, t in enumerate(hourly['datetime'])}

        s_list = pd.date_range(start=prev_until + pd.Timedelta(hours=1), end=new_until, freq="H")
        dur = self.durations
        L = self.cfg.sequence_length
        series = hourly['passenger_count_scaled']

        X_1h, X_1d, X_1w, X_1m, Y = [], [], [], [], []
        for s in s_list:
            if s not in idx:
                continue
            end_i = idx[s]           # 标签在 s
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

        if not X_1m:
            # 没有新增样本（例如数据缺失）
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

        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, new_until)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        """用保存的模型对 target_date 小时做预测（输入为 [target-720, target-1]）"""
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} 没有 checkpoint，请先训练")

        model, scaler = self._load(zone_id)
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=target_date)
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
                self.incremental_update(df, zone_id, prev_until=prev, new_until=target_date)
        return self.predict(df, zone_id, target_date)


# ============== 一个可直接运行的 main()（IDE 里点 ▶️ 即可） ==============

# 用户可改的配置
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
CHECKPOINT_DIR = "checkpoints_multiscale_inc"

START_TARGET = pd.Timestamp("2021-03-05 12:00")  # 第 721 小时
ROLLING_STEPS = 2                                 # 连续预测小时数
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]    # 过滤的区域
RETRAIN_EACH_HOUR = False                          # True = 每小时从零重训（仅调试用）


def _prepare_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("H")
    if EXCLUDED_ZONES:
        df = df[~df["PULocationID"].isin(EXCLUDED_ZONES)].copy()
    return df


def _get_true_counts(df: pd.DataFrame, hour: pd.Timestamp) -> pd.Series:
    return df[df["datetime"] == hour].groupby("PULocationID").size()


def run_rolling(df: pd.DataFrame):
    zones = sorted(df["PULocationID"].unique().tolist())
    cfg = ManagerConfig()
    base_mgr = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)

    rows, hours = [], []
    for i in range(ROLLING_STEPS):
        target = START_TARGET + pd.Timedelta(hours=i)
        print(f"\n🕒 Target hour: {target}")

        # 可选：每小时重训（独立目录，非增量，仅供对照/调试）
        mgr = base_mgr
        if RETRAIN_EACH_HOUR:
            mgr = MultiScaleModelManager(
                checkpoint_dir=str(Path(CHECKPOINT_DIR) / target.strftime("%Y%m%d_%H%M")),
                cfg=cfg
            )

        y_true_map = _get_true_counts(df, target)
        preds, trues = [], []
        for zid in zones:
            try:
                y_pred = mgr.train_and_predict_if_needed(df, zid, target, auto_train=True)
                y_true = float(y_true_map.get(zid, 0.0))
                rows.append({"target_hour": target, "PULocationID": zid, "y_pred": y_pred, "y_true": y_true})
                preds.append(y_pred); trues.append(y_true)
            except Exception as e:
                rows.append({"target_hour": target, "PULocationID": zid, "y_pred": np.nan, "y_true": np.nan, "error": str(e)})

        # 小时级指标
        preds = np.array(preds, dtype=float); trues = np.array(trues, dtype=float)
        m = ~np.isnan(preds) & ~np.isnan(trues)
        if m.any():
            mae = float(np.mean(np.abs(preds[m] - trues[m])))
            rmse = float(np.sqrt(np.mean((preds[m] - trues[m]) ** 2)))
            hours.append({"target_hour": target, "MAE": mae, "RMSE": rmse, "N": int(m.sum())})
            print(f"✅ Hourly MAE={mae:.3f}, RMSE={rmse:.3f}")
        else:
            hours.append({"target_hour": target, "MAE": np.nan, "RMSE": np.nan, "N": 0})
            print("⚠️ 无有效样本计算指标")

    # 保存
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv("predictions_rolling.csv", index=False)
    hour_df = pd.DataFrame(hours).sort_values("target_hour")
    hour_df.to_csv("hourly_metrics.csv", index=False)

    # Overall
    m = pred_df.dropna(subset=["y_pred", "y_true"])
    if not m.empty:
        overall_mae = float(np.mean(np.abs(m["y_pred"].values - m["y_true"].values)))
        overall_rmse = float(np.sqrt(np.mean((m["y_pred"].values - m["y_true"].values) ** 2)))
    else:
        overall_mae = overall_rmse = np.nan
    with open("overall_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall MAE: {overall_mae}\nOverall RMSE: {overall_rmse}\n")
    print(f"\n🎯 Overall MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    print("已保存：predictions_rolling.csv / hourly_metrics.csv / overall_metrics.txt")


def main():
    df = _prepare_df()
    run_rolling(df)


if __name__ == "__main__":
    main()
