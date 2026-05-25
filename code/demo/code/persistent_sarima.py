"""Persistent SARIMA baseline with train-once/reuse behavior.

The public manager interface mirrors the neural baselines used by demo_test.py:
train_and_predict_if_needed(df, zone_id, target_date) returns one hourly demand
forecast for a pickup zone and saves per-zone checkpoints.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ManagerConfig:
    order: Tuple[int, int, int] = (1, 0, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 24)
    trend: Optional[str] = None
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 50
    context_hours: int = 24 * 60
    forecast_length: int = 1
    min_fit_points: int = 48


class SARIMAModelManager:
    """Per-zone persistent SARIMA manager.

    Statsmodels is imported lazily so other demo_test.py backends can still run
    when statsmodels is not installed. Install it before selecting
    MODEL_BACKEND = "sarima".
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints_sarima",
        cfg: Optional[ManagerConfig] = None,
        hidden_size: Optional[int] = None,
        device: object = None,
    ) -> None:
        del hidden_size, device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or ManagerConfig()

    @property
    def _forecast_delta(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.cfg.forecast_length)

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"sarima_zone_{int(zone_id)}.pkl"

    def _meta_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"sarima_meta_zone_{int(zone_id)}.json"

    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._meta_path(zone_id).exists()

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
    ) -> pd.Series:
        df = self._ensure_datetime(df)
        context_hours = max(
            self.cfg.context_hours,
            self.cfg.min_fit_points + self.cfg.forecast_length + 1,
        )
        start_date = end_inclusive - pd.Timedelta(hours=context_hours)
        mask = (
            (df["PULocationID"] == zone_id)
            & (df["datetime"] >= start_date)
            & (df["datetime"] <= end_inclusive)
        )
        hourly_counts = df.loc[mask].groupby("datetime").size()
        hourly_index = pd.date_range(start=start_date, end=end_inclusive, freq="h")
        series = hourly_counts.reindex(hourly_index, fill_value=0).astype(float)
        series.index.freq = "h"
        return series

    def _save_meta(self, zone_id: int, context_end: pd.Timestamp, kind: str) -> None:
        meta = {
            "last_trained_context_end": str(context_end),
            "kind": kind,
            "order": list(self.cfg.order),
            "seasonal_order": list(self.cfg.seasonal_order),
        }
        with open(self._meta_path(zone_id), "w", encoding="utf-8") as file_handle:
            json.dump(meta, file_handle)

    def _load_meta(self, zone_id: int) -> Dict[str, object]:
        with open(self._meta_path(zone_id), "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def _save_payload(self, zone_id: int, payload: Dict[str, object]) -> None:
        with open(self._model_path(zone_id), "wb") as file_handle:
            pickle.dump(payload, file_handle)

    def _load_payload(self, zone_id: int) -> Dict[str, object]:
        with open(self._model_path(zone_id), "rb") as file_handle:
            return pickle.load(file_handle)

    @staticmethod
    def _seasonal_naive_forecast(history: np.ndarray, horizon: int, season_length: int) -> float:
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            return 0.0
        if season_length > 0 and values.size >= season_length:
            idx = values.size - season_length + ((horizon - 1) % season_length)
            return float(values[idx])
        return float(values[-min(values.size, max(1, season_length)) :].mean())

    def train_once(self, df: pd.DataFrame, zone_id: int, context_end: pd.Timestamp) -> None:
        if self.has_checkpoint(zone_id):
            return

        series = self._prepare_zone_series(df, zone_id, context_end)
        values = series.to_numpy(dtype=float)
        season_length = int(self.cfg.seasonal_order[3])

        if len(values) < self.cfg.min_fit_points or np.isclose(values.std(), 0.0):
            payload = {"kind": "seasonal_naive", "history": values, "season_length": season_length}
            self._save_payload(zone_id, payload)
            self._save_meta(zone_id, context_end, kind="seasonal_naive")
            return

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            warnings.warn(
                'statsmodels is not installed; using seasonal naive fallback for '
                'MODEL_BACKEND = "sarima". Install statsmodels for SARIMAX fitting.',
                RuntimeWarning,
                stacklevel=2,
            )
            payload = {"kind": "seasonal_naive", "history": values, "season_length": season_length}
            self._save_payload(zone_id, payload)
            self._save_meta(zone_id, context_end, kind="seasonal_naive")
            return

        try:
            model = SARIMAX(
                series,
                order=self.cfg.order,
                seasonal_order=self.cfg.seasonal_order,
                trend=self.cfg.trend,
                enforce_stationarity=self.cfg.enforce_stationarity,
                enforce_invertibility=self.cfg.enforce_invertibility,
            )
            result = model.fit(disp=False, maxiter=self.cfg.maxiter)
            payload = {"kind": "sarima", "result": result, "history": values}
            kind = "sarima"
        except Exception:
            payload = {"kind": "seasonal_naive", "history": values, "season_length": season_length}
            kind = "seasonal_naive"

        self._save_payload(zone_id, payload)
        self._save_meta(zone_id, context_end, kind=kind)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        del df
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(f"Zone {zone_id} has no SARIMA checkpoint.")

        meta = self._load_meta(zone_id)
        context_end = pd.Timestamp(meta["last_trained_context_end"])
        horizon = int((pd.Timestamp(target_date) - context_end) / pd.Timedelta(hours=1))
        if horizon < 1:
            horizon = 1

        payload = self._load_payload(zone_id)
        if payload.get("kind") == "sarima":
            result = payload["result"]
            forecast = result.forecast(steps=horizon)
            prediction = float(np.asarray(forecast, dtype=float)[-1])
        else:
            prediction = self._seasonal_naive_forecast(
                history=np.asarray(payload.get("history", []), dtype=float),
                horizon=horizon,
                season_length=int(payload.get("season_length", self.cfg.seasonal_order[3])),
            )

        return float(max(0.0, prediction))

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        zone_id: int,
        target_date: pd.Timestamp,
    ) -> Tuple[float, float, Dict[str, float]]:
        point = self.predict(df, zone_id, target_date)
        payload = self._load_payload(zone_id)
        history = np.asarray(payload.get("history", []), dtype=float)
        std = float(np.std(history[-24:])) if history.size else 0.0
        diagnostics = {"sarima_recent_variance": float(std**2)}
        return point, std, diagnostics

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
                raise FileNotFoundError(f"No SARIMA checkpoint for zone {zone_id}.")
            self.train_once(df, zone_id, context_end)

        return self.predict(df, zone_id, target_date)


MultiScaleModelManager = SARIMAModelManager
