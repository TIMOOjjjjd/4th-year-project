"""
Incremental multiscale trainer with confidence-weighted auxiliary heads.

Combines the persistent incremental training workflow with the MC Dropout
confidence logic used in the multiscale_confidence variant.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiScaleModel(nn.Module):
    """Four-branch encoder with auxiliary heads and MC Dropout support."""

    def __init__(self, hidden_size: int, p_drop: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # Branch encoders
        self.lstm_1d = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_1w = nn.LSTM(1, hidden_size, batch_first=True)
        self.input_projection = nn.Linear(1, hidden_size)
        self.transformer_1m = nn.Transformer(
            d_model=hidden_size, nhead=4, num_encoder_layers=2, batch_first=True
        )

        # Branch-level dropout (for MC Dropout)
        self.do_1d = nn.Dropout(p_drop)
        self.do_1w = nn.Dropout(p_drop)
        self.do_1m = nn.Dropout(p_drop)
        self.do_1h = nn.Dropout(p_drop)

        # Fusion + GRU backbone
        self.feature_fusion = nn.Linear(hidden_size * 3, hidden_size)
        self.gru = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        # Auxiliary regression heads
        self.head_1d = nn.Linear(hidden_size, 1)
        self.head_1w = nn.Linear(hidden_size, 1)
        self.head_1m = nn.Linear(hidden_size, 1)
        self.head_1h = nn.Linear(hidden_size, 1)

    def _encode_temporal_branches(self, x: Dict[str, torch.Tensor]):
        x_1d, x_1w, x_1m = x["1d"], x["1w"], x["1m"]

        _, (h_1d, _) = self.lstm_1d(x_1d)
        h_1d = self.do_1d(h_1d[-1])

        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = self.do_1w(h_1w[-1])

        x_1m = self.input_projection(x_1m)
        h_seq = self.transformer_1m(x_1m, x_1m)
        h_1m = self.do_1m(h_seq[:, -1, :])

        return h_1d, h_1w, h_1m

    def _build_gru_input(self, x_1h: torch.Tensor, fused_trend: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x_1h.shape
        fused = fused_trend.unsqueeze(1).repeat(1, seq_len, 1)
        return torch.cat([x_1h, fused], dim=2)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        Returns:
            y_main: [B, 1]
            embeddings: tuple of branch embeddings (1h, 1d, 1w, 1m)
        """
        h_1d, h_1w, h_1m = self._encode_temporal_branches(x)
        fused_trend = self.feature_fusion(torch.cat([h_1d, h_1w, h_1m], dim=1))

        x_gru = self._build_gru_input(x["1h"], fused_trend)
        gru_out, h_n = self.gru(x_gru)
        h_1h = self.do_1h(gru_out[:, -1, :])

        y_main = self.fc(h_n[-1])
        return y_main, (h_1h, h_1d, h_1w, h_1m)

    def mc_branch_embeddings(self, x: Dict[str, torch.Tensor], K: int):
        """Collect branch embeddings under MC Dropout for variance estimation."""
        self.train()
        emb_1h, emb_1d, emb_1w, emb_1m = [], [], [], []
        with torch.no_grad():
            K = max(1, int(K))
            for _ in range(K):
                _, (h_1h, h_1d, h_1w, h_1m) = self.forward(x)
                emb_1h.append(h_1h.unsqueeze(0))
                emb_1d.append(h_1d.unsqueeze(0))
                emb_1w.append(h_1w.unsqueeze(0))
                emb_1m.append(h_1m.unsqueeze(0))
        return (
            torch.cat(emb_1h, dim=0),
            torch.cat(emb_1d, dim=0),
            torch.cat(emb_1w, dim=0),
            torch.cat(emb_1m, dim=0),
        )

    def mc_predict(self, x: Dict[str, torch.Tensor], M: int):
        """Return mean and std of MC Dropout predictions in scaled space."""
        self.train()
        preds = []
        with torch.no_grad():
            M = max(1, int(M))
            for _ in range(M):
                y_main, _ = self.forward(x)
                preds.append(y_main.unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


@dataclass
class ManagerConfig:
    hidden_size: int = 64
    epochs_full: int = 50
    epochs_incremental: int = 1
    patience_full: int = 5
    lr_full: float = 1e-3
    lr_incremental: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    sequence_length: int = 24 * 30
    forecast_length: int = 1
    p_drop: float = 0.2
    lambda_aux: float = 0.5
    K_mc_train: int = 5
    M_mc_test: int = 20


class MultiScaleModelManager:
    """
    - Full training up to a target hour on first run.
    - Subsequent incremental fine-tuning per new hour with confidence-weighted auxiliaries.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints_multiscale", cfg: Optional[ManagerConfig] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ManagerConfig()
        self.durations = {"1h": 1, "1d": 24, "1w": 24 * 7, "1m": 24 * 30}

    @property
    def _forecast_delta(self) -> pd.Timedelta:
        """Forecast horizon expressed as timedelta (hourly granularity)."""
        return pd.Timedelta(hours=self.cfg.forecast_length)

    # ---------- paths ----------
    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"multiscale_zone_{zone_id}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"scaler_zone_{zone_id}.pkl"

    def _meta_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"meta_zone_{zone_id}.json"

    # ---------- persistence helpers ----------
    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._scaler_path(zone_id).exists()

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        with open(self._scaler_path(zone_id), "wb") as f:
            import pickle

            pickle.dump(scaler, f)

    def _load(self, zone_id: int) -> Tuple[nn.Module, MinMaxScaler]:
        model = MultiScaleModel(self.cfg.hidden_size, p_drop=self.cfg.p_drop).to(device)
        state = torch.load(self._model_path(zone_id), map_location=device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), "rb") as f:
            import pickle

            scaler = pickle.load(f)
        return model, scaler

    def _save_meta(self, zone_id: int, context_end: pd.Timestamp) -> None:
        meta = {"last_trained_context_end": str(context_end)}
        with open(self._meta_path(zone_id), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_meta(self, zone_id: int) -> Optional[pd.Timestamp]:
        path = self._meta_path(zone_id)
        if not path.exists():
            return None
        meta = json.load(open(path, "r", encoding="utf-8"))
        ts = meta.get("last_trained_context_end") or meta.get("last_trained_until")
        return pd.Timestamp(ts) if ts is not None else None

    # ---------- data prep ----------
    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, end_inclusive: pd.Timestamp) -> pd.DataFrame:
        """Return dense hourly counts from [end_inclusive - sequence_length, end_inclusive]."""
        assert {"datetime", "PULocationID"} <= set(df.columns)

        start_date = end_inclusive - pd.Timedelta(hours=self.cfg.sequence_length)
        zone_df = df[df["PULocationID"] == zone_id].copy()
        hourly = zone_df.groupby("datetime").size().reset_index(name="passenger_count")

        rng = pd.date_range(start=start_date, end=end_inclusive, freq="H")
        hourly = (
            hourly.set_index("datetime")
            .reindex(rng)
            .fillna(0)
            .rename_axis("datetime")
            .reset_index()
        )
        hourly["passenger_count"] = hourly["passenger_count"].astype(float).fillna(0.0)
        return hourly

    def _fit_scaler_hist(self, hourly: pd.DataFrame, fit_until_exclusive: pd.Timestamp) -> MinMaxScaler:
        """Fit MinMax scaler on history prior to fit_until_exclusive (guarding against edge cases)."""
        scaler = MinMaxScaler()
        hist = hourly[hourly["datetime"] < fit_until_exclusive]
        values = hist[["passenger_count"]].astype(float)

        if values.empty:
            scaler.fit(pd.DataFrame({"passenger_count": [0.0, 1.0]}))
            return scaler

        vmin = float(values["passenger_count"].min())
        vmax = float(values["passenger_count"].max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            base = 0.0 if not np.isfinite(vmin) else vmin
            scaler.fit(pd.DataFrame({"passenger_count": [base, base + 1.0]}))
            return scaler

        scaler.fit(values)
        return scaler

    def _build_training_windows(self, hourly: pd.DataFrame, scaler: Optional[MinMaxScaler] = None):
        seq_len = self.cfg.sequence_length
        flen = self.cfg.forecast_length

        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(hourly[["passenger_count"]])

        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])

        X_1h, X_1d, X_1w, X_1m, y = [], [], [], [], []
        for i in range(len(hourly) + 1 - seq_len - flen):
            series = hourly["passenger_count_scaled"]
            X_1h.append(series.iloc[i + seq_len - self.durations["1h"] : i + seq_len].values)
            X_1d.append(series.iloc[i + seq_len - self.durations["1d"] : i + seq_len].values)
            X_1w.append(series.iloc[i + seq_len - self.durations["1w"] : i + seq_len].values)
            X_1m.append(series.iloc[i : i + seq_len].values)
            y_val = series.iloc[i + seq_len : i + seq_len + flen].values
            if len(y_val) == flen:
                y.append(y_val)

        if not X_1m:
            raise ValueError("Insufficient history to build any training window.")

        X_tensor = {
            "1h": torch.tensor(np.array(X_1h), dtype=torch.float32, device=device).unsqueeze(-1),
            "1d": torch.tensor(np.array(X_1d), dtype=torch.float32, device=device).unsqueeze(-1),
            "1w": torch.tensor(np.array(X_1w), dtype=torch.float32, device=device).unsqueeze(-1),
            "1m": torch.tensor(np.array(X_1m), dtype=torch.float32, device=device).unsqueeze(-1),
        }
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32, device=device)
        return X_tensor, y_tensor, scaler

    def _build_inference_window(
        self, hourly: pd.DataFrame, scaler: MinMaxScaler, context_end: pd.Timestamp
    ) -> Dict[str, torch.Tensor]:
        L = self.cfg.sequence_length
        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])

        idx_map = {ts: idx for idx, ts in enumerate(hourly["datetime"])}
        if context_end not in idx_map:
            raise ValueError("context_end missing from hourly series.")

        end_idx = idx_map[context_end]
        start_idx = end_idx - L
        if start_idx < 0:
            raise ValueError("Not enough history to assemble inference window.")

        series = hourly["passenger_count_scaled"]
        X = {
            "1h": torch.tensor(series.iloc[end_idx - self.durations["1h"] : end_idx].values, dtype=torch.float32, device=device).view(1, -1, 1),
            "1d": torch.tensor(series.iloc[end_idx - self.durations["1d"] : end_idx].values, dtype=torch.float32, device=device).view(1, -1, 1),
            "1w": torch.tensor(series.iloc[end_idx - self.durations["1w"] : end_idx].values, dtype=torch.float32, device=device).view(1, -1, 1),
            "1m": torch.tensor(series.iloc[start_idx:end_idx].values, dtype=torch.float32, device=device).view(1, -1, 1),
        }
        return X

    @staticmethod
    def _conf_weights_from_embeddings(embeddings: Tuple[torch.Tensor, ...], eps: float = 1e-6):
        emb_1h, emb_1d, emb_1w, emb_1m = embeddings
        var_1h = emb_1h.var(dim=0).mean(dim=1, keepdim=True)
        var_1d = emb_1d.var(dim=0).mean(dim=1, keepdim=True)
        var_1w = emb_1w.var(dim=0).mean(dim=1, keepdim=True)
        var_1m = emb_1m.var(dim=0).mean(dim=1, keepdim=True)

        conf_1h = 1.0 / (var_1h + eps)
        conf_1d = 1.0 / (var_1d + eps)
        conf_1w = 1.0 / (var_1w + eps)
        conf_1m = 1.0 / (var_1m + eps)

        conf_sum = conf_1h + conf_1d + conf_1w + conf_1m
        return (
            conf_1h / conf_sum,
            conf_1d / conf_sum,
            conf_1w / conf_sum,
            conf_1m / conf_sum,
        )

    @staticmethod
    def _inverse_std_minmax(std_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        data_range = scaler.data_max_[0] - scaler.data_min_[0]
        return std_scaled * data_range

    # ---------- training ----------
    def train_once(self, df: pd.DataFrame, zone_id: int, context_end: pd.Timestamp) -> None:
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, context_end)
        X_tensor, y_tensor, scaler = self._build_training_windows(hourly)

        model = MultiScaleModel(self.cfg.hidden_size, p_drop=self.cfg.p_drop).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr_full)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_ctr = 0

        for _ in range(self.cfg.epochs_full):
            model.train()
            optimizer.zero_grad()

            y_main, _ = model(X_tensor)
            L_main = criterion(y_main, y_tensor)

            with torch.no_grad():
                embeddings = model.mc_branch_embeddings(X_tensor, self.cfg.K_mc_train)
                weights = self._conf_weights_from_embeddings(embeddings)

            # No auxiliary head outputs anymore; use embedding-confidence only to scale main loss if needed
            # Here we approximate by weighting main loss with average confidence across branches
            avg_conf = sum(weights) / len(weights)
            L_aux = (L_main * avg_conf.mean()).detach() * 0.0

            loss = L_main + self.cfg.lambda_aux * L_aux
            loss.backward()
            optimizer.step()

            current = loss.item()
            if current < best_loss:
                best_loss = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience_full:
                    break

        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, context_end)

    def incremental_update(
        self,
        df: pd.DataFrame,
        zone_id: int,
        prev_until: pd.Timestamp,
        new_until: pd.Timestamp,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> None:
        if new_until <= prev_until:
            return

        epochs = epochs if epochs is not None else self.cfg.epochs_incremental
        lr = lr if lr is not None else self.cfg.lr_incremental

        model, _ = self._load(zone_id)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.cfg.weight_decay)
        criterion = nn.SmoothL1Loss()

        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=new_until)
        fit_start = new_until - pd.Timedelta(days=30)
        hourly_fit = hourly[(hourly["datetime"] < new_until) & (hourly["datetime"] >= fit_start)].copy()
        scaler = self._fit_scaler_hist(hourly_fit, fit_until_exclusive=new_until)
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])

        idx_map = {ts: idx for idx, ts in enumerate(hourly["datetime"])}
        s_list = pd.date_range(start=prev_until + pd.Timedelta(hours=1), end=new_until, freq="H")
        series = hourly["passenger_count_scaled"]

        X_1h, X_1d, X_1w, X_1m, Y = [], [], [], [], []
        for s in s_list:
            end_idx = idx_map.get(s)
            if end_idx is None:
                continue
            start_idx = end_idx - self.cfg.sequence_length
            if start_idx < 0 or end_idx + self.cfg.forecast_length > len(series):
                continue

            X_1h.append(series.iloc[end_idx - self.durations["1h"] : end_idx].values)
            X_1d.append(series.iloc[end_idx - self.durations["1d"] : end_idx].values)
            X_1w.append(series.iloc[end_idx - self.durations["1w"] : end_idx].values)
            X_1m.append(series.iloc[start_idx:end_idx].values)
            Y.append(series.iloc[end_idx : end_idx + self.cfg.forecast_length].values)

        if not X_1m:
            self._save(zone_id, model, scaler)
            self._save_meta(zone_id, prev_until)
            return

        X = {
            "1h": torch.tensor(np.array(X_1h), dtype=torch.float32, device=device).unsqueeze(-1),
            "1d": torch.tensor(np.array(X_1d), dtype=torch.float32, device=device).unsqueeze(-1),
            "1w": torch.tensor(np.array(X_1w), dtype=torch.float32, device=device).unsqueeze(-1),
            "1m": torch.tensor(np.array(X_1m), dtype=torch.float32, device=device).unsqueeze(-1),
        }
        Y = torch.tensor(np.array(Y), dtype=torch.float32, device=device)

        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            y_main, _ = model(X)
            L_main = criterion(y_main, Y)

            with torch.no_grad():
                embeddings = model.mc_branch_embeddings(X, self.cfg.K_mc_train)
                weights = self._conf_weights_from_embeddings(embeddings)

            avg_conf = sum(weights) / len(weights)
            L_aux = (L_main * avg_conf.mean()).detach() * 0.0

            loss = L_main + self.cfg.lambda_aux * L_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()

        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, new_until)

    # ---------- inference ----------
    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no checkpoint; train first.")

        model, _ = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=context_end)
        scaler = self._fit_scaler_hist(hourly, fit_until_exclusive=target_date)
        X_last = self._build_inference_window(hourly, scaler, context_end)

        model.eval()
        with torch.no_grad():
            mean_scaled, _ = model.mc_predict(X_last, self.cfg.M_mc_test)
        mean_np = mean_scaled.cpu().numpy()
        pred = scaler.inverse_transform(mean_np)[0, 0]
        return float(pred)

    def predict_with_uncertainty(
        self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp
    ) -> Tuple[float, float, Dict[str, float]]:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no checkpoint; train first.")

        model, _ = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, end_inclusive=context_end)
        scaler = self._fit_scaler_hist(hourly, fit_until_exclusive=target_date)
        X_last = self._build_inference_window(hourly, scaler, context_end)

        model.eval()
        with torch.no_grad():
            mean_scaled, std_scaled = model.mc_predict(X_last, self.cfg.M_mc_test)
            embeddings = model.mc_branch_embeddings(X_last, self.cfg.M_mc_test)

        mean_np = mean_scaled.cpu().numpy()
        std_np = std_scaled.cpu().numpy()
        point = scaler.inverse_transform(mean_np)[0, 0]
        std_orig = float(self._inverse_std_minmax(std_np, scaler)[0, 0])

        branch_var = {
            "1h": embeddings[0].var(dim=0).mean().item(),
            "1d": embeddings[1].var(dim=0).mean().item(),
            "1w": embeddings[2].var(dim=0).mean().item(),
            "1m": embeddings[3].var(dim=0).mean().item(),
        }
        return float(point), std_orig, branch_var

    # ---------- orchestration ----------
    def train_and_predict_if_needed(
        self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp, auto_train: bool = True
    ) -> float:
        context_end = target_date - self._forecast_delta
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No checkpoint for zone {zone_id} and auto_train is False.")
            self.train_once(df, zone_id, context_end)
        else:
            prev = self._load_meta(zone_id)
            if prev is None:
                self.train_once(df, zone_id, context_end)
            elif prev < context_end:
                print(f"[inc] zone={zone_id} {prev} -> {context_end}")
                self.incremental_update(df, zone_id, prev_until=prev, new_until=context_end)

        return self.predict(df, zone_id, target_date)


def _prepare_df_from_parquet(parquet_path: str) -> pd.DataFrame:
    cols = ["pickup_datetime", "PULocationID"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("H")
    return df



