# persistent_transformer.py
# 一个可持久化、按区域增量训练的纯 Transformer 基线
# 接口与 PureLSTMModelManager 基本一致：train_and_predict_if_needed(df, zid, t, auto_train=True)

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============== 基础工具 ===============
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
        history: window length (e.g., 720). For each target index i (history..T-1),
                 x = series[i-history:i], y = series[i]
        """
        self.series = series
        self.history = history
        self.T = len(series)

    def __len__(self) -> int:
        return max(0, self.T - self.history)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.series[idx: idx + self.history]
        y = self.series[idx + self.history]
        # 返回形状：(history, 1) -> Transformer 的 (B, T, d) 中的 T 维
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)


# =============== 模型 ===============
class PositionalEncoding(nn.Module):
    """标准正弦位置编码，适配 (B, T, d_model) 的 batch_first 张量"""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (T, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # (1, T, d)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class PureTransformer(nn.Module):
    """
    纯 Transformer 编码器：
      - 输入 (B, T, 1) 先线性投影到 d_model
      - 加位置编码
      - 过 N 层 TransformerEncoder
      - 取最后时刻 token 的表示接回归头 -> (B, 1)
    """
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_len: int = 2048):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # 更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.posenc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        z = self.input_proj(x)                # (B, T, d)
        z = self.posenc(z)                    # (B, T, d)
        # 因为窗口内都是纯历史，不需要 causal mask；如需可加 src_mask
        enc = self.encoder(z)                 # (B, T, d)
        last = enc[:, -1, :]                  # (B, d)
        y = self.head(last)                   # (B, 1)
        return y


# =============== 状态与管理器 ===============
@dataclass
class ZoneState:
    model: PureTransformer
    scaler_mean: float
    scaler_std: float
    last_trained_upto: Optional[pd.Timestamp] = None  # 该时间点（含）之前的目标已用于训练


class PureTransformerModelManager:
    """
    纯 Transformer 的持久化对照基线。
    - 与 LSTM 管理器保持相同接口：train_and_predict_if_needed(df, zid, t, auto_train=True)
    - 每个 zone 单独训练与持久化
    - 增量训练：只用“上次已训练到”之后的新窗口继续训练
    Checkpoint 内容：state_dict, scaler(mean/std), last_trained_upto
    """

    def __init__(self,
                 checkpoint_dir: str = "checkpoints_transformer",
                 history_needed: int = 720,
                 hidden_size: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 epochs_initial: int = 3,
                 epochs_incremental: int = 1,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 device: Optional[torch.device] = None,
                 verbose: bool = False):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history_needed = history_needed
        self.d_model = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.epochs_initial = epochs_initial
        self.epochs_incremental = epochs_incremental
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self._zones: Dict[int, ZoneState] = {}

    # ---------- 公共接口 ----------
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

        # 载入/初始化 zone 状态
        state = self._load_or_init_zone(zid)

        # 训练（或增量训练）
        if auto_train:
            state = self._fit_zone_state(zid, state, series_full, train_last_target_time=hist_end)

        # 预测窗口（最近 720 小时）
        input_window = _to_hourly_counts(df, zid, time_min=hist_start, time_max=hist_end).values.astype(np.float32)
        if len(input_window) < self.history_needed:
            raise RuntimeError(f"Not enough history window ({len(input_window)}) for zid={zid}, t={t}.")

        # 归一化 + 推理
        x = (input_window - state.scaler_mean) / (state.scaler_std + 1e-6)
        x_tensor = torch.from_numpy(x).view(1, self.history_needed, 1).to(self.device)

        state.model.eval()
        with torch.no_grad():
            yhat_norm = state.model(x_tensor).squeeze().item()

        # 反缩放并做非负裁剪（计数）
        yhat = yhat_norm * (state.scaler_std + 1e-6) + state.scaler_mean
        return float(max(0.0, yhat))

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

        # 用将要用于训练的区间估计缩放器（简单 z-score）
        train_start_idx = target_indices.min() - self.history_needed
        arr_for_scaler = values[train_start_idx: target_indices.max() + 1]
        scaler_mean = float(np.mean(arr_for_scaler))
        scaler_std = float(np.std(arr_for_scaler) + 1e-3)
        state.scaler_mean, state.scaler_std = scaler_mean, scaler_std

        # 归一化后的序列
        norm_values = (values - scaler_mean) / scaler_std

        # 构造“仅包含目标索引”的子数据集
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
                print(f"[Transformer zid={zid}] epoch {ep+1}/{epochs} loss={total/max(1,n):.4f}")

        # 更新状态并保存
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
            model = PureTransformer(
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )
            model.load_state_dict(blob["state_dict"])
            return ZoneState(
                model=model.to(self.device),
                scaler_mean=float(blob["scaler_mean"]),
                scaler_std=float(blob["scaler_std"]),
                last_trained_upto=pd.Timestamp(blob["last_trained_upto"]) if blob["last_trained_upto"] is not None else None
            )
        # 初始化新模型
        model = PureTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)
        return ZoneState(model=model, scaler_mean=0.0, scaler_std=1.0, last_trained_upto=None)

    def _save_zone_state(self, zid: int, state: ZoneState) -> None:
        p = self._zone_ckpt_path(zid)
        torch.save({
            "state_dict": state.model.state_dict(),
            "scaler_mean": state.scaler_mean,
            "scaler_std": state.scaler_std,
            "last_trained_upto": None if state.last_trained_upto is None else str(state.last_trained_upto)
        }, p)
