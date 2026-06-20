"""Utilities for dynamic residual-dependency graph construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from build_residual_graph import build_residual_correlation_graph


GraphContextT = TypeVar("GraphContextT")


def residual_graph_dir(args) -> Path:
    return (
        args.residual_graph_dir
        if args.residual_graph_dir is not None
        else args.results_dir / "residual_graphs"
    )


def residual_graph_path(args, target_hour: pd.Timestamp) -> Path:
    timestamp = pd.Timestamp(target_hour).strftime("%Y%m%d_%H%M")
    return residual_graph_dir(args) / f"edge_weight_matrix_residual_until_{timestamp}.csv"


def residual_graph_summary_path(args) -> Path:
    return residual_graph_dir(args) / "residual_graph_summary.csv"


def get_cached_base_prediction_sequence(
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
    manager,
    df: pd.DataFrame,
    zone_id: int,
    target_hours: Sequence[pd.Timestamp],
) -> Tuple[list[float], int]:
    failures = 0
    normalized_hours = [pd.Timestamp(hour) for hour in target_hours]
    for hour in normalized_hours:
        key = (int(zone_id), hour)
        if key in prediction_cache:
            continue
        try:
            prediction_cache[key] = float(manager.predict(df, int(zone_id), hour))
        except Exception:  # noqa: BLE001
            failures += 1
            prediction_cache[key] = np.nan

    return [prediction_cache[(int(zone_id), hour)] for hour in normalized_hours], failures


def build_residual_history_frames(
    args,
    df: pd.DataFrame,
    manager,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    graph_context,
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build historical true/base-pred matrices for residual graph construction."""

    context_end = pd.Timestamp(target_hour) - manager._forecast_delta
    for zone_id in zones:
        trained_until = manager._load_meta(int(zone_id))
        if trained_until is not None and trained_until > context_end:
            raise RuntimeError(
                f"Checkpoint for zone {zone_id} was trained through {trained_until}, "
                f"which is after residual graph context_end={context_end}."
            )

    window_start = pd.Timestamp(target_hour) - pd.Timedelta(hours=args.residual_window_hours)
    history_hours = pd.date_range(
        start=window_start,
        periods=args.residual_window_hours,
        freq="h",
    )
    zone_names = list(graph_context.zone_names)
    y_true = pd.DataFrame(0.0, index=history_hours, columns=zone_names)
    y_pred = pd.DataFrame(np.nan, index=history_hours, columns=zone_names)

    prediction_failures = 0
    for zone_id in zones:
        zone_int = int(zone_id)
        zone_name = graph_context.location_to_zone.get(zone_int)
        if zone_name is None or zone_name not in y_true.columns:
            continue

        try:
            true_series = zone_hourly_counts.loc[zone_int]
            y_true[zone_name] = true_series.reindex(history_hours, fill_value=0.0).astype(float)
        except KeyError:
            y_true[zone_name] = 0.0

        predictions, failures = get_cached_base_prediction_sequence(
            prediction_cache=prediction_cache,
            manager=manager,
            df=df,
            zone_id=zone_int,
            target_hours=history_hours,
        )
        prediction_failures += failures
        y_pred[zone_name] = predictions

    return y_true, y_pred, prediction_failures


def build_dynamic_residual_graph_context(
    args,
    df: pd.DataFrame,
    manager,
    lookup_df: pd.DataFrame,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    graph_context: GraphContextT,
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
    build_graph_context_fn: Callable[[Path, pd.DataFrame], GraphContextT],
) -> Tuple[GraphContextT, Dict[str, object]]:
    """Construct a residual-dependency graph using history before target_hour."""

    y_true, y_pred, prediction_failures = build_residual_history_frames(
        args=args,
        df=df,
        manager=manager,
        target_hour=target_hour,
        zones=zones,
        zone_hourly_counts=zone_hourly_counts,
        graph_context=graph_context,
        prediction_cache=prediction_cache,
    )

    output_path = residual_graph_path(args=args, target_hour=target_hour)
    _matrix, summary = build_residual_correlation_graph(
        y_true=y_true,
        y_pred=y_pred,
        timestamps=y_true.index,
        current_t=target_hour,
        window_hours=args.residual_window_hours,
        top_k=args.residual_top_k,
        use_abs_corr=not args.residual_use_signed_corr,
        min_corr=args.residual_min_corr,
        symmetrize=args.residual_symmetrize,
        save_path=output_path,
        zone_names=y_true.columns.tolist(),
        random_seed=args.seed,
        return_summary=True,
        verbose=True,
    )

    window_start = pd.Timestamp(target_hour) - pd.Timedelta(hours=args.residual_window_hours)
    summary.update(
        {
            "target_hour": pd.Timestamp(target_hour),
            "window_start": window_start,
            "window_end_exclusive": pd.Timestamp(target_hour),
            "prediction_failures": int(prediction_failures),
            "graph_csv": str(output_path),
        }
    )
    print(
        f"Generated residual graph [{window_start}, {target_hour}) at {output_path} "
        f"with {summary['num_edges']} nonzero edges."
    )
    return build_graph_context_fn(output_path, lookup_df), summary
