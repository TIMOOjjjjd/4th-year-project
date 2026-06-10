from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Union

import numpy as np
import torch


CONFIDENCE_COMPONENT_NAMES = ("prior", "stability", "history_consistency")
DEFAULT_CONFIDENCE_WEIGHTS = np.array([0.3, 0.4, 0.3], dtype=np.float32)


def normalize_confidence_weights(
    weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
) -> np.ndarray:
    if weights is None:
        values = DEFAULT_CONFIDENCE_WEIGHTS.copy()
    elif isinstance(weights, Mapping):
        values = np.array(
            [float(weights[name]) for name in CONFIDENCE_COMPONENT_NAMES],
            dtype=np.float32,
        )
    else:
        values = np.array(list(weights), dtype=np.float32)

    if values.shape != DEFAULT_CONFIDENCE_WEIGHTS.shape:
        raise ValueError(
            f"Expected {len(DEFAULT_CONFIDENCE_WEIGHTS)} confidence weights, got {len(values)}."
        )
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Confidence weights must have a positive finite sum.")
    return values / total


def combine_confidence_components(
    prior: float,
    stability: float,
    history_consistency: float,
    weights: Optional[Union[Sequence[float], Mapping[str, float]]] = None,
) -> float:
    weight_values = normalize_confidence_weights(weights)
    components = np.array(
        [prior, stability, history_consistency],
        dtype=np.float32,
    )
    components = np.clip(components, 0.05, 1.0)
    log_score = float((np.log(components) * weight_values).sum())
    return float(np.clip(np.exp(log_score), 0.05, 1.0))


def component_tensor(data) -> torch.Tensor:
    return torch.stack(
        [
            data.prior_score.float(),
            data.stability_score.float(),
            data.history_consistency_score.float(),
        ],
        dim=1,
    ).clamp(min=0.05, max=1.0)


def confidence_from_component_weights(data, weights: torch.Tensor) -> torch.Tensor:
    components = component_tensor(data)
    weights = weights.float().to(components.device)
    weights = weights / weights.sum().clamp(min=1e-6)
    log_confidence = (torch.log(components) * weights.unsqueeze(0)).sum(dim=1)
    return torch.exp(log_confidence).clamp(min=0.05, max=1.0)
