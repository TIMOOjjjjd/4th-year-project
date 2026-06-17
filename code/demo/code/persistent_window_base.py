"""Shared train-once/reuse manager for single-window forecasting backends."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class PersistentWindowModelManager:
    """Base manager for per-zone sequence models with a common public API."""

    model_name = "WindowModel"
    file_prefix = "window"

    def __init__(
        self,
        checkpoint_dir: str,
        cfg: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if cfg is None:
            raise ValueError("cfg must be provided by the concrete backend manager.")
        if hidden_size is not None and hasattr(cfg, "hidden_size"):
            cfg = replace(cfg, hidden_size=hidden_size)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def _forecast_delta(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.cfg.forecast_length)

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"{self.file_prefix}_zone_{int(zone_id)}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"{self.file_prefix}_scaler_zone_{int(zone_id)}.pkl"

    def _meta_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"{self.file_prefix}_meta_zone_{int(zone_id)}.json"

    def has_checkpoint(self, zone_id: int) -> bool:
        return (
            self._model_path(zone_id).exists()
            and self._scaler_path(zone_id).exists()
            and self._meta_path(zone_id).exists()
        )

    def _new_model(self) -> nn.Module:
        raise NotImplementedError

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        with open(self._scaler_path(zone_id), "wb") as file_handle:
            pickle.dump(scaler, file_handle)

    def _load(self, zone_id: int) -> Tuple[nn.Module, MinMaxScaler]:
        model = self._new_model()
        state = torch.load(self._model_path(zone_id), map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), "rb") as file_handle:
            scaler = pickle.load(file_handle)
        return model, scaler

    def _save_meta(self, zone_id: int, context_end: pd.Timestamp) -> None:
        meta = {"last_trained_context_end": str(context_end)}
        with open(self._meta_path(zone_id), "w", encoding="utf-8") as file_handle:
            json.dump(meta, file_handle)

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
        if "datetime" in df.columns:
            return df
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")
        return df

    def _prepare_zone_series(
        self,
        df: pd.DataFrame,
        zone_id: int,
        end_inclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        df = self._ensure_datetime(df)
        context_hours = max(
            self.cfg.context_hours,
            self.cfg.sequence_length + self.cfg.forecast_length + 1,
        )
        start_date = end_inclusive - pd.Timedelta(hours=context_hours)
        mask = (
            (df["PULocationID"] == zone_id)
            & (df["datetime"] >= start_date)
            & (df["datetime"] <= end_inclusive)
        )
        hourly_counts = df.loc[mask].groupby("datetime").size()
        hourly_index = pd.date_range(start=start_date, end=end_inclusive, freq="h")
        hourly = (
            hourly_counts.reindex(hourly_index, fill_value=0)
            .rename("passenger_count")
            .rename_axis("datetime")
            .reset_index()
        )
        hourly["passenger_count"] = hourly["passenger_count"].astype(float)
        return hourly

    @staticmethod
    def _fit_scaler_hist(
        hourly: pd.DataFrame,
        fit_until_exclusive: pd.Timestamp,
    ) -> MinMaxScaler:
        scaler = MinMaxScaler()
        hist = hourly[hourly["datetime"] < fit_until_exclusive]
        values = hist[["passenger_count"]].astype(float)

        if values.empty:
            scaler.fit(pd.DataFrame({"passenger_count": [0.0, 1.0]}))
            return scaler

        min_value = float(values["passenger_count"].min())
        max_value = float(values["passenger_count"].max())
        if (
            not np.isfinite(min_value)
            or not np.isfinite(max_value)
            or np.isclose(min_value, max_value)
        ):
            base = 0.0 if not np.isfinite(min_value) else min_value
            scaler.fit(pd.DataFrame({"passenger_count": [base, base + 1.0]}))
            return scaler

        scaler.fit(values)
        return scaler

    def _build_training_arrays(
        self,
        hourly: pd.DataFrame,
        scaler: MinMaxScaler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sequence_length = self.cfg.sequence_length
        forecast_length = self.cfg.forecast_length
        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])
        series = hourly["passenger_count_scaled"]

        input_values, target_values = [], []
        for start_idx in range(len(hourly) + 1 - sequence_length - forecast_length):
            target_start = start_idx + sequence_length
            target_end = target_start + forecast_length
            input_values.append(series.iloc[start_idx:target_start].values)
            target_values.append(series.iloc[target_start:target_end].values)

        if not input_values:
            raise ValueError(
                f"Insufficient history to build {self.model_name} training windows."
            )

        input_array = np.asarray(input_values, dtype=np.float32)[..., None]
        target_array = np.asarray(target_values, dtype=np.float32)
        return input_array, target_array

    def _build_inference_window(
        self,
        hourly: pd.DataFrame,
        scaler: MinMaxScaler,
        context_end: pd.Timestamp,
    ) -> torch.Tensor:
        sequence_length = self.cfg.sequence_length
        hourly = hourly.copy()
        hourly["passenger_count_scaled"] = scaler.transform(hourly[["passenger_count"]])
        idx_map = {ts: idx for idx, ts in enumerate(hourly["datetime"])}
        if context_end not in idx_map:
            raise ValueError("context_end missing from hourly series.")

        end_idx = idx_map[context_end]
        start_idx = end_idx - sequence_length + 1
        if start_idx < 0:
            raise ValueError(
                f"Not enough history to assemble {self.model_name} inference window."
            )

        values = hourly["passenger_count_scaled"].iloc[start_idx : end_idx + 1].values
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, -1, 1)

    def _train_arrays(
        self,
        model: nn.Module,
        input_array: np.ndarray,
        target_array: np.ndarray,
    ) -> None:
        dataset = TensorDataset(torch.from_numpy(input_array), torch.from_numpy(target_array))
        loader = DataLoader(
            dataset,
            batch_size=min(self.cfg.batch_size, max(1, len(dataset))),
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.lr_full,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.MSELoss()
        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        for _ in range(max(1, int(self.cfg.epochs_full))):
            model.train()
            total_loss = 0.0
            sample_count = 0
            for input_batch, target_batch in loader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(input_batch), target_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()
                total_loss += float(loss.item()) * input_batch.size(0)
                sample_count += input_batch.size(0)

            epoch_loss = total_loss / max(1, sample_count)
            if epoch_loss < best_loss - self.cfg.min_delta:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience_full:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

    @staticmethod
    def _inverse_values(values_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        original_shape = values_scaled.shape
        values = np.asarray(values_scaled, dtype=np.float32).reshape(-1, 1)
        return scaler.inverse_transform(values).reshape(original_shape)

    @staticmethod
    def _inverse_std_minmax(std_scaled: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        data_range = float(scaler.data_max_[0] - scaler.data_min_[0])
        return std_scaled * data_range

    def train_once(self, df: pd.DataFrame, zone_id: int, context_end: pd.Timestamp) -> None:
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, context_end)
        scaler = self._fit_scaler_hist(
            hourly,
            fit_until_exclusive=context_end + pd.Timedelta(hours=1),
        )
        input_array, target_array = self._build_training_arrays(hourly, scaler)
        model = self._new_model()
        self._train_arrays(model, input_array, target_array)
        self._save(zone_id, model, scaler)
        self._save_meta(zone_id, context_end)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no {self.model_name} checkpoint.")

        model, scaler = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, context_end)
        input_last = self._build_inference_window(hourly, scaler, context_end)

        model.eval()
        with torch.no_grad():
            prediction_scaled = model(input_last).cpu().numpy()
        prediction = self._inverse_values(prediction_scaled, scaler)[0, 0]
        return float(max(0.0, prediction))

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        zone_id: int,
        target_date: pd.Timestamp,
    ) -> Tuple[float, float, Dict[str, float]]:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no {self.model_name} checkpoint.")

        model, scaler = self._load(zone_id)
        context_end = target_date - self._forecast_delta
        hourly = self._prepare_zone_series(df, zone_id, context_end)
        input_last = self._build_inference_window(hourly, scaler, context_end)

        mean_scaled, std_scaled = model.mc_predict(input_last, self.cfg.M_mc_test)
        mean_values = mean_scaled.cpu().numpy()
        std_values = std_scaled.cpu().numpy()
        point = self._inverse_values(mean_values, scaler)[0, 0]
        std_orig = self._inverse_std_minmax(std_values, scaler)[0, 0]
        diagnostics = {f"{self.file_prefix}_mc_variance": float(std_orig**2)}
        return float(max(0.0, point)), float(max(0.0, std_orig)), diagnostics

    def train_and_predict_if_needed(
        self,
        df: pd.DataFrame,
        zone_id: int,
        target_date: pd.Timestamp,
        auto_train: bool = True,
    ) -> float:
        context_end = target_date - self._forecast_delta
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No {self.model_name} checkpoint for zone {zone_id}.")
            self.train_once(df, zone_id, context_end)

        return self.predict(df, zone_id, target_date)
