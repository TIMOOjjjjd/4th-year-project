"""Persistent PatchTST baseline with train-once/reuse behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from persistent_window_base import PersistentWindowModelManager


@dataclass
class ManagerConfig:
    hidden_size: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    patch_length: int = 24
    patch_stride: int = 12
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


class PatchTST(nn.Module):
    """Patch-based Transformer encoder for univariate hourly demand."""

    def __init__(
        self,
        sequence_length: int,
        forecast_length: int,
        patch_length: int,
        patch_stride: int,
        hidden_size: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_length <= 0:
            raise ValueError("patch_length must be positive.")
        if patch_stride <= 0:
            raise ValueError("patch_stride must be positive.")
        if sequence_length < patch_length:
            raise ValueError("sequence_length must be at least patch_length.")
        if hidden_size % nhead != 0:
            raise ValueError("hidden_size must be divisible by nhead.")

        self.patch_length = int(patch_length)
        self.patch_stride = int(patch_stride)
        self.num_patches = 1 + (sequence_length - patch_length) // patch_stride

        self.patch_projection = nn.Linear(patch_length, hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, forecast_length)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = values.squeeze(-1)
        patches = sequence.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        if patches.size(1) != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {patches.size(1)}."
            )

        tokens = self.patch_projection(patches)
        tokens = self.dropout(tokens + self.positional_embedding)
        encoded = self.encoder(tokens)
        last_token = self.dropout(encoded[:, -1, :])
        return self.head(last_token)

    def mc_predict(self, values: torch.Tensor, samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(max(1, int(samples))):
                predictions.append(self.forward(values).unsqueeze(0))
        prediction_stack = torch.cat(predictions, dim=0)
        return prediction_stack.mean(dim=0), prediction_stack.std(dim=0, unbiased=False)


class PatchTSTModelManager(PersistentWindowModelManager):
    """Per-zone persistent PatchTST manager."""

    model_name = "PatchTST"
    file_prefix = "patchtst"

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints_patchtst",
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

    def _new_model(self) -> PatchTST:
        return PatchTST(
            sequence_length=self.cfg.sequence_length,
            forecast_length=self.cfg.forecast_length,
            patch_length=self.cfg.patch_length,
            patch_stride=self.cfg.patch_stride,
            hidden_size=self.cfg.hidden_size,
            nhead=self.cfg.nhead,
            num_layers=self.cfg.num_layers,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.p_drop,
        ).to(self.device)


MultiScaleModelManager = PatchTSTModelManager
