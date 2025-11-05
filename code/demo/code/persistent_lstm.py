# persistent_lstm.py
# 一个可持久化、按区域增量训练的纯 LSTM 基线
# 用法与 MultiScaleModelManager 基本一致：train_and_predict_if_needed(df, zid, t, auto_train=True)

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def _to_hourly_counts(df: pd.DataFrame,
                      zid: int,
                      time_min: pd.Timestamp,
                      time_max: pd.Timestamp) -> pd.Series:
    """将给定区域在 [time_min, time_max] 内聚合为按小时计数的等间隔序列（缺失填 0）"""
    sub = df[df["PULocationID"] == zid].copy()
    if "datetime" not in sub.columns:
        sub["datetime"] = pd.to_datetime(sub["pickup_datetime"]).dt.floor("H")
    idx = pd.date_range(time_min, time_max, freq="h")
    counts = sub.groupby("datetime").size().reindex(idx, fill_value=0)
    counts.index.name = "datetime"
    return counts.astype(np.float32)


class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, history: int):
        """
        series: 1D array of length T
        history: window length (e.g., 720). For each i, x=series[i-history:i], y=series[i]
        需要满足 T > history
        """
        self.series = series
        self.history = history
        self.T = len(series)

    def __len__(self) -> int:
        return max(0, self.T - self.history)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx: idx + self.history]
        y = self.series[idx + self.history]
        # 形状：(history, 1) -> LSTM 的 (seq_len, features)
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)


class PureLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        out, _ = self.lstm(x)
        # 取最后时刻
        last = out[:, -1, :]
        y = self.head(last)
        return y  # (B, 1)


@dataclass
class ZoneState:
    model: PureLSTM
    scaler_mean: float
    scaler_std: float
    last_trained_upto: Optional[pd.Timestamp] = None  # 该时间点（含）之前的目标已用于训练


class PureLSTMModelManager:
    """
    纯 LSTM 的持久化对照基线。
    - 与原管理器保持相同接口：train_and_predict_if_needed(df, zid, t, auto_train=True)
    - 每个 zone 单独训练与持久化
    - 增量训练：只用「上次已训练到」之后的新窗口继续训练
    Checkpoint 内容：state_dict, scaler, last_trained_upto
    """

    def __init__(self,
                 checkpoint_dir: str = "checkpoints_lstm",
                 hidden_size: int = 64,
                 history_needed: int = 720,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 epochs_initial: int = 3,
                 epochs_incremental: int = 1,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 device: Optional[torch.device] = None,
                 verbose: bool = False):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.hidden_size = hidden_size
        self.history_needed = history_needed
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs_initial = epochs_initial
        self.epochs_incremental = epochs_incremental
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self._zones: Dict[int, ZoneState] = {}

    # ---------- 公共接口（与 MultiScaleModelManager 对齐） ----------
    def train_and_predict_if_needed(self, df: pd.DataFrame, zid: int, t: pd.Timestamp, auto_train: bool = True) -> float:
        """
        使用 [t-history_needed, ..., t-1] 的历史预测 t 的值。
        若 auto_train=True，则会在可用历史上训练/增量训练模型。
        """
        if "datetime" not in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("H")

        hist_start = t - pd.Timedelta(hours=self.history_needed)
        hist_end = t - pd.Timedelta(hours=1)

        # 准备该区域在 [global_min, hist_end] 的全部序列，便于形成多窗口训练
        global_min = df["datetime"].min()
        if hist_end <= global_min:
            raise RuntimeError(f"Not enough history to predict t={t} for zid={zid}.")

        series_full = _to_hourly_counts(df, zid, time_min=global_min, time_max=hist_end)

        # 缩放器用“已用于训练的全部窗口”的输入与目标估计（简单 z-score）
        # 训练/增量训练
        state = self._load_or_init_zone(zid)
        # 计算训练数据可用的“目标时间范围”
        train_last_target_time = hist_end  # 只训练到 t-1 的目标
        if auto_train:
            state = self._fit_zone_state(zid, state, series_full, train_last_target_time)

        # 预测窗口必须是最近 history_needed 小时
        input_window = _to_hourly_counts(df, zid, time_min=hist_start, time_max=hist_end).values.astype(np.float32)

        if len(input_window) < self.history_needed:
            raise RuntimeError(f"Not enough history window ({len(input_window)}) for zid={zid}, t={t}.")

        # 应用缩放
        x = (input_window - state.scaler_mean) / (state.scaler_std + 1e-6)
        x_tensor = torch.from_numpy(x).view(1, self.history_needed, 1).to(self.device)

        state.model.eval()
        with torch.no_grad():
            yhat_norm = state.model(x_tensor).squeeze().item()
        # 反缩放
        yhat = yhat_norm * (state.scaler_std + 1e-6) + state.scaler_mean
        return float(max(0.0, yhat))  # 计数下限为 0

    # ---------- 内部：训练 & 状态管理 ----------
    def _fit_zone_state(self,
                        zid: int,
                        state: ZoneState,
                        series_full: pd.Series,
                        train_last_target_time: pd.Timestamp) -> ZoneState:
        """
        在该区域的历史上训练/增量训练。
        series_full: [global_min .. hist_end] 的每小时序列
        train_last_target_time: 训练数据的最后一个“目标时刻”
        """
        # 可形成的窗口数量
        # 我们将 series_full 转为 numpy，然后构造 dataset：所有目标索引 i ∈ [history_needed, T-1]
        # 对应的目标时刻是 series_full.index[i]
        values = series_full.values.astype(np.float32)
        times = series_full.index

        if len(values) <= self.history_needed:
            # 历史长度不足，无法训练；缩放器用当前可用数据估计
            mean, std = float(np.mean(values)), float(np.std(values) + 1e-3)
            state.scaler_mean, state.scaler_std = mean, std
            return state

        # 选择需要用于（初次或增量）训练的目标索引范围
        target_mask = (times >= times[self.history_needed]) & (times <= train_last_target_time)
        if state.last_trained_upto is not None:
            target_mask &= (times > state.last_trained_upto)

        target_indices = np.nonzero(target_mask)[0]
        if target_indices.size == 0:
            # 没有新数据需要训练，仍然更新缩放器（便于长期适应）
            used = values[: max(self.history_needed, min(len(values), 5000))]
            state.scaler_mean = float(np.mean(used))
            state.scaler_std = float(np.std(used) + 1e-3)
            return state

        # 用“将要用于训练的窗口的输入与目标”估计缩放器
        # 为简单起见，这里用整段可训练区间的均值方差
        train_start_idx = target_indices.min() - self.history_needed
        arr_for_scaler = values[train_start_idx: target_indices.max() + 1]
        scaler_mean = float(np.mean(arr_for_scaler))
        scaler_std = float(np.std(arr_for_scaler) + 1e-3)
        state.scaler_mean, state.scaler_std = scaler_mean, scaler_std

        # 归一化后的序列
        norm_values = (values - scaler_mean) / scaler_std

        # 构造数据集（只包含要训练的那部分目标）
        # 我们用一个“索引映射”来只取需要的样本，避免全量 dataset 再采样
        class _SubsetSeqDataset(Dataset):
            def __init__(self, series: np.ndarray, history: int, target_idxs: np.ndarray):
                self.series = series
                self.history = history
                self.target_idxs = target_idxs

            def __len__(self):
                return len(self.target_idxs)

            def __getitem__(self, j):
                i = self.target_idxs[j]
                x = self.series[i - self.history: i]
                y = self.series[i]
                return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)

        ds = _SubsetSeqDataset(norm_values, self.history_needed, target_indices)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        model = state.model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # 训练轮数：首次更多，增量更少
        epochs = self.epochs_initial if state.last_trained_upto is None else self.epochs_incremental

        for ep in range(epochs):
            total = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(self.device)  # (B, T, 1)
                yb = yb.to(self.device)  # (B, 1)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item() * xb.size(0)
                n += xb.size(0)
            if self.verbose:
                print(f"[zid={zid}] epoch {ep+1}/{epochs} loss={total/max(1,n):.4f}")

        # 更新 “已训练到” 的时间并保存
        state.last_trained_upto = train_last_target_time
        self._zones[zid] = state
        self._save_zone_state(zid, state)
        return state

    def _zone_ckpt_path(self, zid: int) -> str:
        return os.path.join(self.checkpoint_dir, f"zone_{int(zid)}.pt")

    def _load_or_init_zone(self, zid: int) -> ZoneState:
        p = self._zone_ckpt_path(zid)
        if os.path.exists(p):
            blob = torch.load(p, map_location="cpu")
            model = PureLSTM(input_size=1,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout)
            model.load_state_dict(blob["state_dict"])
            return ZoneState(
                model=model.to(self.device),
                scaler_mean=float(blob["scaler_mean"]),
                scaler_std=float(blob["scaler_std"]),
                last_trained_upto=pd.Timestamp(blob["last_trained_upto"]) if blob["last_trained_upto"] is not None else None
            )
        # 初始化新模型
        model = PureLSTM(input_size=1,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout=self.dropout).to(self.device)
        return ZoneState(model=model, scaler_mean=0.0, scaler_std=1.0, last_trained_upto=None)

    def _save_zone_state(self, zid: int, state: ZoneState) -> None:
        p = self._zone_ckpt_path(zid)
        torch.save({
            "state_dict": state.model.state_dict(),
            "scaler_mean": state.scaler_mean,
            "scaler_std": state.scaler_std,
            "last_trained_upto": None if state.last_trained_upto is None else str(state.last_trained_upto)
        }, p)
