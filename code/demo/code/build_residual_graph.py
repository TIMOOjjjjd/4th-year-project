from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _as_frame(
    values: ArrayLike,
    timestamps: Optional[Sequence[object]],
    zone_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        frame = values.copy()
        if timestamps is not None:
            frame.index = pd.to_datetime(timestamps)
        if zone_names is not None:
            if len(zone_names) != frame.shape[1]:
                raise ValueError("zone_names length must match the number of columns.")
            frame.columns = [str(zone) for zone in zone_names]
        return frame

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays shaped [time, zone].")

    columns = (
        [str(zone) for zone in zone_names]
        if zone_names is not None
        else [str(idx) for idx in range(array.shape[1])]
    )
    if len(columns) != array.shape[1]:
        raise ValueError("zone_names length must match the number of columns.")

    index = pd.to_datetime(timestamps) if timestamps is not None else None
    return pd.DataFrame(array, index=index, columns=columns)


def _filter_history(
    residuals: pd.DataFrame,
    current_t: Optional[object],
    window_hours: int,
) -> pd.DataFrame:
    if current_t is None:
        return residuals
    if window_hours <= 0:
        raise ValueError("window_hours must be > 0 when current_t is provided.")

    if isinstance(residuals.index, pd.DatetimeIndex):
        end_time = pd.Timestamp(current_t)
        start_time = end_time - pd.Timedelta(hours=window_hours)
        return residuals[(residuals.index >= start_time) & (residuals.index < end_time)]

    current_idx = int(current_t)
    start_idx = max(0, current_idx - window_hours)
    return residuals.iloc[start_idx:current_idx]


def _safe_corr(residuals: pd.DataFrame) -> pd.DataFrame:
    corr = residuals.astype(float).corr(method="pearson")
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return corr.clip(lower=-1.0, upper=1.0)


def _random_pair_mean(
    score_matrix: np.ndarray,
    random_seed: int,
    max_pairs: int = 10_000,
) -> float:
    node_count = score_matrix.shape[0]
    if node_count < 2:
        return 0.0

    rows, cols = np.where(~np.eye(node_count, dtype=bool))
    if rows.size == 0:
        return 0.0

    rng = np.random.default_rng(random_seed)
    sample_size = min(max_pairs, rows.size)
    selected = rng.choice(rows.size, size=sample_size, replace=False)
    return float(np.mean(score_matrix[rows[selected], cols[selected]]))


def _summarize_graph(
    adjacency: np.ndarray,
    score_matrix: np.ndarray,
    top_k: int,
    window_hours: int,
    random_seed: int,
) -> Dict[str, float]:
    edge_values = adjacency[adjacency != 0.0]

    if edge_values.size == 0:
        mean_edge = median_edge = max_edge = min_edge = 0.0
        mean_selected_corr = 0.0
    else:
        mean_edge = float(np.mean(edge_values))
        median_edge = float(np.median(edge_values))
        max_edge = float(np.max(edge_values))
        min_edge = float(np.min(edge_values))
        mean_selected_corr = float(np.mean(np.abs(edge_values)))

    return {
        "num_nodes": int(adjacency.shape[0]),
        "num_edges": int(edge_values.size),
        "top_k": int(top_k),
        "window_hours": int(window_hours),
        "mean_edge_weight": mean_edge,
        "median_edge_weight": median_edge,
        "max_edge_weight": max_edge,
        "min_edge_weight": min_edge,
        "mean_selected_top_k_residual_corr": mean_selected_corr,
        "mean_random_pair_corr": _random_pair_mean(
            score_matrix=score_matrix,
            random_seed=random_seed,
        ),
    }


def build_residual_correlation_graph(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    timestamps: Optional[Sequence[object]] = None,
    current_t: Optional[object] = None,
    window_hours: int = 720,
    top_k: int = 10,
    use_abs_corr: bool = True,
    min_corr: float = 0.0,
    symmetrize: bool = False,
    save_path: Optional[Path] = None,
    zone_names: Optional[Sequence[str]] = None,
    random_seed: int = 42,
    summary_path: Optional[Path] = None,
    return_summary: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]:
    """Build a task-aligned graph from historical temporal-model residuals.

    The residual graph is intended for residual refinement: zones are connected
    when their historical base-model errors move together. When current_t is
    provided, only residuals before current_t are used, avoiding target-hour
    true-value leakage.
    """

    if top_k < 0:
        raise ValueError("top_k must be >= 0.")
    if min_corr < 0.0:
        raise ValueError("min_corr must be >= 0.")

    true_frame = _as_frame(y_true, timestamps=timestamps, zone_names=zone_names)
    pred_frame = _as_frame(y_pred, timestamps=timestamps, zone_names=true_frame.columns)
    pred_frame = pred_frame.reindex(index=true_frame.index, columns=true_frame.columns)

    residuals = true_frame.astype(float) - pred_frame.astype(float)
    residuals = _filter_history(
        residuals=residuals,
        current_t=current_t,
        window_hours=window_hours,
    )
    if residuals.empty:
        raise ValueError("No historical residuals are available for graph construction.")

    corr = _safe_corr(residuals)
    corr_values = corr.to_numpy(dtype=float)
    rank_scores = np.abs(corr_values)
    np.fill_diagonal(rank_scores, -np.inf)

    node_count = corr_values.shape[0]
    adjacency = np.zeros((node_count, node_count), dtype=float)
    for row_idx in range(node_count):
        candidate_scores = rank_scores[row_idx]
        candidate_indices = np.where(candidate_scores >= min_corr)[0]
        if candidate_indices.size == 0:
            continue

        order = np.lexsort((candidate_indices, -candidate_scores[candidate_indices]))
        ranked = candidate_indices[order]
        selected = ranked if top_k == 0 else ranked[:top_k]
        if use_abs_corr:
            adjacency[row_idx, selected] = np.abs(corr_values[row_idx, selected])
        else:
            adjacency[row_idx, selected] = corr_values[row_idx, selected]

    if symmetrize:
        if use_abs_corr:
            adjacency = np.maximum(adjacency, adjacency.T)
        else:
            transpose = adjacency.T
            use_transpose = np.abs(transpose) > np.abs(adjacency)
            adjacency = np.where(use_transpose, transpose, adjacency)

    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = pd.DataFrame(adjacency, index=corr.index.tolist(), columns=corr.columns.tolist())
    summary = _summarize_graph(
        adjacency=adjacency,
        score_matrix=np.where(np.isfinite(rank_scores), rank_scores, 0.0),
        top_k=top_k,
        window_hours=window_hours,
        random_seed=random_seed,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(save_path, encoding="utf-8")

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_path, index=False)

    if verbose:
        print(
            "Residual graph diagnostics: "
            f"nodes={summary['num_nodes']}, edges={summary['num_edges']}, "
            f"mean_edge_weight={summary['mean_edge_weight']:.4f}, "
            f"median_edge_weight={summary['median_edge_weight']:.4f}, "
            f"max_edge_weight={summary['max_edge_weight']:.4f}, "
            f"min_edge_weight={summary['min_edge_weight']:.4f}, "
            "mean_selected_top_k_residual_corr="
            f"{summary['mean_selected_top_k_residual_corr']:.4f}, "
            f"mean_random_pair_corr={summary['mean_random_pair_corr']:.4f}"
        )

    if return_summary:
        return matrix, summary
    return matrix
