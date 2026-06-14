"""Persistent DLinear baseline with train-once/reuse behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from persistent_window_base import PersistentWindowModelManager


@dataclass
class ManagerConfig:
    hidden_size: int = 64
    moving_avg_kernel: int = 25
    epochs_full: int = 50
    patience_full: int = 5
    lr_full: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    sequence_length: int = 24 * 30
    forecast_length: int = 1
    p_drop: float = 0.2
    batch_size: int = 64
    context_hours: int = 800
    lambda_aux: float = 0.0
    K_mc_train: int = 5
    M_mc_test: int = 20
    min_delta: float = 1e-6


class MovingAverage(nn.Module):
    """Centered moving average used by the DLinear decomposition."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("moving_avg_kernel must be positive.")
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return values

        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = values[:, 0:1, :].repeat(1, pad_left, 1)
        end = values[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, values, end], dim=1)
        trend = self.avg(padded.transpose(1, 2)).transpose(1, 2)
        return trend


class DLinear(nn.Module):
    """Decomposition-Linear forecaster for univariate hourly demand."""

    def __init__(
        self,
        sequence_length: int,
        forecast_length: int,
        moving_avg_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decomposition = MovingAverage(moving_avg_kernel)
        self.dropout = nn.Dropout(dropout)
        self.seasonal_linear = nn.Linear(sequence_length, forecast_length)
        self.trend_linear = nn.Linear(sequence_length, forecast_length)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(values)
        seasonal = values - trend
        seasonal = self.dropout(seasonal.squeeze(-1))
        trend = self.dropout(trend.squeeze(-1))
        return self.seasonal_linear(seasonal) + self.trend_linear(trend)

    def mc_predict(self, values: torch.Tensor, samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(max(1, int(samples))):
                predictions.append(self.forward(values).unsqueeze(0))
        prediction_stack = torch.cat(predictions, dim=0)
        return prediction_stack.mean(dim=0), prediction_stack.std(dim=0, unbiased=False)


class DLinearModelManager(PersistentWindowModelManager):
    """Per-zone persistent DLinear manager."""

    model_name = "DLinear"
    file_prefix = "dlinear"

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints_dlinear",
        cfg: Optional[ManagerConfig] = None,
        hidden_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            cfg=cfg or ManagerConfig(),
            hidden_size=hidden_size,
            device=device,
        )

    def _new_model(self) -> DLinear:
        return DLinear(
            sequence_length=self.cfg.sequence_length,
            forecast_length=self.cfg.forecast_length,
            moving_avg_kernel=self.cfg.moving_avg_kernel,
            dropout=self.cfg.p_drop,
        ).to(self.device)


MultiScaleModelManager = DLinearModelManager
