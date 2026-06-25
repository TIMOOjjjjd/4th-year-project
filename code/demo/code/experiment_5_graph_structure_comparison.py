"""Experiment 5: compare OD, residual, geographic, random, and no-graph baselines.

This experiment reuses Experiment 3's confidence-weighted residual GraphSAGE
pipeline. The only variable is the graph structure:

    no_graph: temporal baseline only, no residual GNN
    od:       residual GraphSAGE on the rolling 30-day OD-flow graph
    residual: residual GraphSAGE on historical base-model residual correlation
    geo:      residual GraphSAGE on the geographic graph
    random:   residual GraphSAGE on random graph baselines across fixed seeds

All graph variants use Experiment 3's learned-softmax confidence-weighted
residual loss.
For fairness, each rolling hour shares the same temporal predictions and the
same location-based split across all graph structures. The GNN train/validation
loss is computed only on historical residual snapshots before the target hour;
target-hour labels are used only for post-inference test metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from build_od_graph import (
    _build_matrix,
    _build_random_matrix,
    _load_location_lookup,
    _load_zone_names,
    _retain_top_k,
)
from build_residual_graph import build_residual_correlation_graph
from experiment_3_confidence_weighted_gnn_ablation import (
    BASE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RESULTS_DIR,
    EXCLUDED_ZONES,
    ROLLING_STEPS,
    START_TARGET,
    GNNTrainingConfig,
    GraphContext,
    ManagerConfig,
    MultiScaleModelManager,
    SplitMasks,
    build_historical_gnn_training_data,
    build_gnn_features,
    build_graph_context,
    build_zone_hourly_counts,
    clean_checkpoint_dir,
    compute_confidence_components,
    compute_prior_scores_from_zone_hourly_counts,
    evaluate_refined_predictions,
    filter_history_frames_before,
    resolve_window_starts,
    run_multiscale_temporal_baseline,
    run_single_ablation,
    set_random_seed,
    validate_windows,
)


GRAPH_CONFIDENCE_MODE = "learned_softmax"
DEFAULT_OD_EDGE_MATRIX = BASE_DIR / "edge_weight_matrix_od.csv"
DEFAULT_GEO_EDGE_MATRIX = BASE_DIR / "edge_weight_matrix_geo.csv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_tcn_shared_v1"
DEFAULT_RANDOM_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_OD_LOOKBACK_DAYS = 30
DEFAULT_RESIDUAL_WINDOW_HOURS = 24 * 30


@dataclass(frozen=True)
class GraphSpec:
    key: str
    model_name: str
    description: str
    edge_csv: Optional[Path]
    random_seed: Optional[int] = None


@dataclass(frozen=True)
class LocationSplitSets:
    train: Set[int]
    val: Set[int]
    test: Set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 5: OD vs residual vs geographic vs random vs no graph."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument(
        "--od-edge-csv",
        type=Path,
        default=DEFAULT_OD_EDGE_MATRIX,
        help="Static OD graph CSV used only when --od-lookback-days is 0.",
    )
    parser.add_argument("--geo-edge-csv", type=Path, default=DEFAULT_GEO_EDGE_MATRIX)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-target", type=pd.Timestamp, default=START_TARGET)
    parser.add_argument(
        "--window-starts",
        nargs="*",
        default=None,
        help=(
            "Optional explicit 24-hour window starts. By default, uses the "
            "same twelve windows as demo_test.py. If omitted with a custom "
            "--start-target, a single window is run for backward compatibility."
        ),
    )
    parser.add_argument("--rolling-steps", type=int, default=ROLLING_STEPS)
    parser.add_argument("--excluded-zones", type=int, nargs="*", default=EXCLUDED_ZONES)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--gnn-epochs", type=int, default=300)
    parser.add_argument("--gnn-hidden-dim", type=int, default=256)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-patience", type=int, default=40)
    parser.add_argument(
        "--use-edge-weight",
        action="store_true",
        help=(
            "Use nonzero CSV values as weighted GraphSAGE aggregation weights. "
            "By default only the graph topology is compared."
        ),
    )
    parser.add_argument(
        "--graph-types",
        nargs="*",
        choices=["no_graph", "od", "residual", "geo", "random"],
        default=["no_graph", "od", "residual", "geo", "random"],
        help="Subset and order of graph variants to evaluate.",
    )
    parser.add_argument(
        "--od-lookback-days",
        type=int,
        default=DEFAULT_OD_LOOKBACK_DAYS,
        help=(
            "Build each OD graph from trips in [target_hour - N days, target_hour). "
            "Use 0 to load the static --od-edge-csv instead."
        ),
    )
    parser.add_argument(
        "--od-graph-dir",
        type=Path,
        default=None,
        help="Directory for generated rolling OD graph CSVs. Defaults to results/od_graphs.",
    )
    parser.add_argument(
        "--od-top-k",
        type=int,
        default=10,
        help="Keep the top K destination zones per origin when generating rolling OD graphs.",
    )
    parser.add_argument(
        "--od-min-flow",
        type=int,
        default=1,
        help="Drop rolling OD edges with fewer than this many trips before top-k filtering.",
    )
    parser.add_argument(
        "--od-weight-mode",
        choices=["row_share", "retained_share", "count", "binary", "log_count"],
        default="row_share",
        help="How to write retained rolling OD edge weights.",
    )
    parser.add_argument(
        "--od-symmetrize",
        choices=["none", "max", "sum", "mean"],
        default="none",
        help="Optionally convert each rolling OD graph to an undirected matrix.",
    )
    parser.add_argument(
        "--od-include-self",
        action="store_true",
        help="Keep trips where pickup and dropoff map to the same zone in rolling OD graphs.",
    )
    parser.add_argument(
        "--residual-window-hours",
        type=int,
        default=DEFAULT_RESIDUAL_WINDOW_HOURS,
        help=(
            "Historical base-residual window used to build each residual graph. "
            "Default is 720 hours, matching 30 days."
        ),
    )
    parser.add_argument(
        "--residual-top-k",
        type=int,
        default=10,
        help="Keep the top K residual-correlated destination zones per origin.",
    )
    parser.add_argument(
        "--residual-min-corr",
        type=float,
        default=0.0,
        help="Drop residual graph edges whose absolute Pearson correlation is below this.",
    )
    parser.add_argument(
        "--residual-use-signed-corr",
        action="store_true",
        help="Use signed Pearson correlation as edge weight; ranking still uses abs(corr).",
    )
    parser.add_argument(
        "--residual-symmetrize",
        action="store_true",
        help="Symmetrize the residual graph after top-k selection.",
    )
    parser.add_argument(
        "--residual-graph-dir",
        type=Path,
        default=None,
        help="Directory for generated residual graph CSVs. Defaults to results/residual_graphs.",
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="*",
        default=DEFAULT_RANDOM_SEEDS,
        help="Fixed seeds used to generate random graph baselines.",
    )
    parser.add_argument(
        "--random-mode",
        choices=["edge_count", "per_origin_top_k"],
        default="per_origin_top_k",
        help="Random graph construction mode reused from build_od_graph.py.",
    )
    parser.add_argument(
        "--random-top-k",
        type=int,
        default=10,
        help="Random outgoing destinations per node, or edge-count fallback multiplier.",
    )
    parser.add_argument(
        "--random-edge-count",
        type=int,
        default=None,
        help="Total random edges for --random-mode edge_count.",
    )
    parser.add_argument(
        "--random-weight-mode",
        choices=["binary", "uniform"],
        default="binary",
        help="Random edge weights. GNN currently uses nonzero topology, not edge weights.",
    )
    parser.add_argument(
        "--random-graph-dir",
        type=Path,
        default=None,
        help="Directory for generated random graph CSVs. Defaults to results/random_graphs.",
    )
    parser.add_argument(
        "--clean-checkpoints",
        action="store_true",
        help="Delete this experiment's temporal checkpoint directory before running.",
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=None,
        help="Optional smoke-test limit. Default evaluates all shared compatible zones.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing experiment_5_graph_structure_comparison_metrics.png.",
    )
    args = parser.parse_args()
    if not args.graph_types:
        parser.error("--graph-types requires at least one graph type when provided.")
    if args.od_lookback_days < 0:
        parser.error("--od-lookback-days must be >= 0.")
    if args.od_top_k < 0:
        parser.error("--od-top-k must be >= 0.")
    if args.od_min_flow < 1:
        parser.error("--od-min-flow must be >= 1.")
    if args.residual_window_hours <= 0:
        parser.error("--residual-window-hours must be > 0.")
    if args.residual_top_k < 0:
        parser.error("--residual-top-k must be >= 0.")
    if args.residual_min_corr < 0.0:
        parser.error("--residual-min-corr must be >= 0.")
    if "random" in args.graph_types and not args.random_seeds:
        parser.error("--random-seeds requires at least one seed when random is selected.")
    if args.random_top_k < 0:
        parser.error("--random-top-k must be >= 0.")
    if args.random_edge_count is not None and args.random_edge_count < 0:
        parser.error("--random-edge-count must be >= 0.")
    return args


def build_graph_specs(args: argparse.Namespace) -> Tuple[Dict[str, GraphSpec], List[str]]:
    random_graph_dir = (
        args.random_graph_dir
        if args.random_graph_dir is not None
        else args.results_dir / "random_graphs"
    )
    od_model_name = (
        f"OD Graph ({args.od_lookback_days}d)"
        if args.od_lookback_days > 0
        else "OD Graph"
    )
    od_description = (
        "Experiment 3 learned-softmax GraphSAGE on rolling "
        f"{args.od_lookback_days}-day OD-flow graphs"
        if args.od_lookback_days > 0
        else "Experiment 3 learned-softmax GraphSAGE on a static OD-flow graph"
    )
    base_specs = {
        "no_graph": GraphSpec(
            key="no_graph",
            model_name="No Graph Temporal",
            description="temporal baseline only; no residual GraphSAGE",
            edge_csv=None,
        ),
        "od": GraphSpec(
            key="od",
            model_name=od_model_name,
            description=od_description,
            edge_csv=args.od_edge_csv,
        ),
        "residual": GraphSpec(
            key="residual",
            model_name="GNN + Residual Graph",
            description=(
                "Experiment 3 learned-softmax GraphSAGE on historical "
                "base-model residual correlation graph"
            ),
            edge_csv=None,
        ),
        "geo": GraphSpec(
            key="geo",
            model_name="Geographic Graph",
            description="Experiment 3 learned-softmax GraphSAGE on geographic graph",
            edge_csv=args.geo_edge_csv,
        ),
    }
    specs: Dict[str, GraphSpec] = {}
    graph_order: List[str] = []
    for graph_type in dict.fromkeys(args.graph_types):
        if graph_type != "random":
            specs[graph_type] = base_specs[graph_type]
            graph_order.append(graph_type)
            continue

        for seed in args.random_seeds:
            key = f"random_seed_{int(seed)}"
            specs[key] = GraphSpec(
                key=key,
                model_name=f"Random Graph seed={int(seed)}",
                description=(
                    "Experiment 3 learned-softmax GraphSAGE on a generated "
                    f"random graph, seed={int(seed)}"
                ),
                edge_csv=random_graph_dir / f"edge_weight_matrix_random_seed_{int(seed)}.csv",
                random_seed=int(seed),
            )
            graph_order.append(key)
    return specs, graph_order


def generate_random_graphs(
    args: argparse.Namespace,
    specs: Dict[str, GraphSpec],
    graph_order: Sequence[str],
) -> None:
    random_specs = [specs[key] for key in graph_order if specs[key].random_seed is not None]
    if not random_specs:
        return

    zone_names = _load_zone_names(args.lookup, template_path=None)
    for spec in random_specs:
        if spec.edge_csv is None or spec.random_seed is None:
            continue
        spec.edge_csv.parent.mkdir(parents=True, exist_ok=True)
        matrix = _build_random_matrix(
            zone_names=zone_names,
            random_mode=args.random_mode,
            random_edge_count=args.random_edge_count,
            top_k=args.random_top_k,
            include_self=False,
            random_seed=spec.random_seed,
            random_weight_mode=args.random_weight_mode,
            symmetrize="none",
            reference=None,
        )
        matrix.to_csv(spec.edge_csv, encoding="utf-8")
        edge_count = int((matrix.to_numpy(dtype=float) > 0.0).sum())
        print(
            f"Generated {spec.key} at {spec.edge_csv} "
            f"with {edge_count} nonzero edges."
        )


def uses_rolling_od_graph(args: argparse.Namespace, graph_order: Sequence[str]) -> bool:
    return "od" in graph_order and args.od_lookback_days > 0


def uses_residual_graph(graph_order: Sequence[str]) -> bool:
    return "residual" in graph_order


def od_graph_dir(args: argparse.Namespace) -> Path:
    return args.od_graph_dir if args.od_graph_dir is not None else args.results_dir / "od_graphs"


def residual_graph_dir(args: argparse.Namespace) -> Path:
    return (
        args.residual_graph_dir
        if args.residual_graph_dir is not None
        else args.results_dir / "residual_graphs"
    )


def od_graph_path(args: argparse.Namespace, target_hour: pd.Timestamp) -> Path:
    timestamp = pd.Timestamp(target_hour).strftime("%Y%m%d_%H%M%S")
    return od_graph_dir(args) / (
        f"edge_weight_matrix_od_last_{args.od_lookback_days}d_until_{timestamp}.csv"
    )


def residual_graph_path(args: argparse.Namespace, target_hour: pd.Timestamp) -> Path:
    timestamp = pd.Timestamp(target_hour).strftime("%Y%m%d_%H%M%S")
    return residual_graph_dir(args) / f"edge_weight_matrix_residual_until_{timestamp}.csv"


def residual_graph_summary_path(args: argparse.Namespace) -> Path:
    return residual_graph_dir(args) / "residual_graph_summary.csv"


def load_od_zone_names(args: argparse.Namespace) -> List[str]:
    template_path = args.geo_edge_csv if args.geo_edge_csv.exists() else None
    return _load_zone_names(args.lookup, template_path=template_path)


def build_zone_only_graph_context(zone_names: List[str], lookup_df: pd.DataFrame) -> GraphContext:
    lookup_by_zone = lookup_df.drop_duplicates(subset="Zone")
    zone_to_location = {
        str(row.Zone): int(row.LocationID)
        for row in lookup_by_zone.itertuples()
        if pd.notna(row.LocationID)
    }
    location_to_zone = {
        int(row.LocationID): str(row.Zone)
        for row in lookup_df.itertuples()
        if pd.notna(row.LocationID)
    }
    return GraphContext(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,), dtype=torch.float32),
        zone_names=zone_names,
        zone_idx_map={zone_name: idx for idx, zone_name in enumerate(zone_names)},
        location_to_zone=location_to_zone,
        zone_to_location=zone_to_location,
    )


def aggregate_od_flows_from_frame(
    frame: pd.DataFrame,
    location_to_zone: Dict[int, str],
    excluded_locations: Set[int],
    include_self: bool,
) -> Tuple[Counter[Tuple[str, str]], Counter[str], int]:
    flows: Counter[Tuple[str, str]] = Counter()
    origin_totals: Counter[str] = Counter()

    od_frame = frame[["PULocationID", "DOLocationID"]].dropna().copy()
    if od_frame.empty:
        return flows, origin_totals, 0

    od_frame["PULocationID"] = od_frame["PULocationID"].astype(int)
    od_frame["DOLocationID"] = od_frame["DOLocationID"].astype(int)

    if excluded_locations:
        od_frame = od_frame[
            ~od_frame["PULocationID"].isin(excluded_locations)
            & ~od_frame["DOLocationID"].isin(excluded_locations)
        ]
    if od_frame.empty:
        return flows, origin_totals, 0

    od_frame["origin_zone"] = od_frame["PULocationID"].map(location_to_zone)
    od_frame["dest_zone"] = od_frame["DOLocationID"].map(location_to_zone)
    od_frame = od_frame.dropna(subset=["origin_zone", "dest_zone"])
    if not include_self:
        od_frame = od_frame[od_frame["origin_zone"] != od_frame["dest_zone"]]
    if od_frame.empty:
        return flows, origin_totals, 0

    grouped = od_frame.groupby(["origin_zone", "dest_zone"], sort=False).size()
    for (origin, dest), count in grouped.items():
        count_int = int(count)
        flows[(str(origin), str(dest))] += count_int
        origin_totals[str(origin)] += count_int

    return flows, origin_totals, len(od_frame)


def build_rolling_od_graph_context(
    args: argparse.Namespace,
    df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    zone_names: List[str],
    location_to_zone: Dict[int, str],
    target_hour: pd.Timestamp,
) -> GraphContext:
    if "DOLocationID" not in df.columns:
        raise ValueError("Rolling OD graph generation requires DOLocationID in taxi data.")

    end_time = pd.Timestamp(target_hour)
    start_time = end_time - pd.Timedelta(days=args.od_lookback_days)
    window_mask = (
        (df["pickup_datetime"] >= start_time)
        & (df["pickup_datetime"] < end_time)
    )
    window_df = df.loc[window_mask, ["PULocationID", "DOLocationID"]]
    flows, origin_totals, kept_rows = aggregate_od_flows_from_frame(
        frame=window_df,
        location_to_zone=location_to_zone,
        excluded_locations=set(args.excluded_zones),
        include_self=args.od_include_self,
    )
    retained = _retain_top_k(
        flows=flows,
        top_k=args.od_top_k,
        min_flow=args.od_min_flow,
    )
    matrix = _build_matrix(
        zone_names=zone_names,
        retained=retained,
        origin_totals=origin_totals,
        weight_mode=args.od_weight_mode,
        symmetrize=args.od_symmetrize,
    )

    output_path = od_graph_path(args=args, target_hour=end_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path, encoding="utf-8")

    edge_count = int((matrix.to_numpy(dtype=float) > 0.0).sum())
    print(
        "Generated rolling OD graph "
        f"[{start_time}, {end_time}) at {output_path} "
        f"from {kept_rows} retained trips with {edge_count} nonzero edges."
    )
    return build_graph_context(edge_csv=output_path, lookup_df=lookup_df)


def cache_baseline_predictions(
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
    baseline_df: pd.DataFrame,
) -> None:
    for row in baseline_df.itertuples():
        target_hour = pd.Timestamp(row.target_hour)
        prediction_cache[(int(row.PULocationID), target_hour)] = float(row.base_pred)


def get_cached_base_prediction_sequence(
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
    manager: MultiScaleModelManager,
    df: pd.DataFrame,
    zone_id: int,
    target_hours: Sequence[pd.Timestamp],
) -> Tuple[List[float], int]:
    """Predict temporal/base demand for many hours while loading the model once."""

    zone_int = int(zone_id)
    normalized_hours = [pd.Timestamp(hour) for hour in target_hours]
    missing_hours = [
        hour for hour in normalized_hours if (zone_int, hour) not in prediction_cache
    ]
    failures = 0

    if missing_hours:
        model, _ = manager._load(zone_int)
        model.eval()
        for hour in missing_hours:
            key = (zone_int, hour)
            try:
                context_end = hour - manager._forecast_delta
                hourly = manager._prepare_zone_series(
                    df=df,
                    zone_id=zone_int,
                    end_inclusive=context_end,
                )
                scaler = manager._fit_scaler_hist(hourly, fit_until_exclusive=hour)
                x_last = manager._build_inference_window(
                    hourly=hourly,
                    scaler=scaler,
                    context_end=context_end,
                )
                with torch.no_grad():
                    mean_scaled, _ = model.mc_predict(x_last, manager.cfg.M_mc_test)
                prediction = scaler.inverse_transform(mean_scaled.cpu().numpy())[0, 0]
                prediction_cache[key] = float(prediction)
            except Exception:  # noqa: BLE001
                failures += 1
                prediction_cache[key] = np.nan

    values = [prediction_cache[(zone_int, hour)] for hour in normalized_hours]
    return values, failures


def build_residual_history_frames(
    args: argparse.Namespace,
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    graph_context: GraphContext,
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build historical true/base-pred matrices for residual graph construction.

    Rows are historical hours strictly before target_hour. Predictions are from
    the temporal/base model only; no GNN-refined predictions are used.
    """

    context_end = pd.Timestamp(target_hour) - manager._forecast_delta
    # A checkpoint trained through target_hour - 1 is allowed here because the
    # graph is built for target_hour and uses only information already historical
    # at that point. The target_hour true value itself is never included.
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


def build_residual_graph_context(
    args: argparse.Namespace,
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    lookup_df: pd.DataFrame,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    graph_context: GraphContext,
    prediction_cache: Dict[Tuple[int, pd.Timestamp], float],
) -> Tuple[GraphContext, Dict[str, object]]:
    # The residual graph is aligned with residual refinement: it connects zones
    # whose historical temporal-model errors co-move before the target hour.
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
    matrix, summary = build_residual_correlation_graph(
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
    del matrix

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
    return build_graph_context(edge_csv=output_path, lookup_df=lookup_df), summary


def load_taxi_data(
    data_path: Path,
    excluded_zones: Sequence[int],
    include_dropoff: bool,
) -> pd.DataFrame:
    columns = ["pickup_datetime", "PULocationID"]
    if include_dropoff:
        columns.append("DOLocationID")
    df = pd.read_parquet(data_path, columns=columns)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(excluded_zones)].copy()

    print("Earliest timestamp:", df["datetime"].min())
    print("Latest timestamp:", df["datetime"].max())
    print("Total hours:", df["datetime"].nunique())
    print("Total non-excluded zones:", df["PULocationID"].nunique())
    return df


def load_graph_contexts(
    specs: Dict[str, GraphSpec],
    requested_graphs: Sequence[str],
    lookup_df: pd.DataFrame,
    skip_graphs: Optional[Set[str]] = None,
) -> Dict[str, GraphContext]:
    skip_graphs = skip_graphs or set()
    graphs: Dict[str, GraphContext] = {}
    for graph_key in requested_graphs:
        if graph_key in skip_graphs:
            continue
        spec = specs[graph_key]
        if spec.edge_csv is None:
            continue
        if not spec.edge_csv.exists():
            raise FileNotFoundError(
                f"Missing {graph_key} graph matrix: {spec.edge_csv}. "
                "Generate it first or pass the corresponding --*-edge-csv path."
            )
        graph = build_graph_context(edge_csv=spec.edge_csv, lookup_df=lookup_df)
        graphs[graph_key] = graph
        print(
            f"Loaded {graph_key} graph from {spec.edge_csv} "
            f"with {len(graph.zone_names)} zones and {graph.edge_index.shape[1]} edges."
        )
    return graphs


def compatible_zones_for_graph(df: pd.DataFrame, graph: GraphContext) -> Set[int]:
    zones: Set[int] = set()
    for zone_id in df["PULocationID"].dropna().unique():
        zone_int = int(zone_id)
        zone_name = graph.location_to_zone.get(zone_int)
        if zone_name is not None and zone_name in graph.zone_idx_map:
            zones.add(zone_int)
    return zones


def select_shared_zones(
    df: pd.DataFrame,
    graphs: Dict[str, GraphContext],
    max_zones: Optional[int],
) -> List[int]:
    shared = set(int(zone_id) for zone_id in df["PULocationID"].dropna().unique())
    for graph_key, graph in graphs.items():
        graph_zones = compatible_zones_for_graph(df, graph)
        shared &= graph_zones
        print(f"{graph_key} graph-compatible zones:", len(graph_zones))

    zones = sorted(shared)
    if max_zones is not None:
        zones = zones[: max(1, max_zones)]
        print(f"Using first {len(zones)} shared zones for smoke-test run.")
    if len(zones) < 3:
        raise ValueError("At least three shared valid zones are required for this experiment.")
    print("Shared zones selected:", len(zones))
    return zones


def finite_step_zone_ids(step_df: pd.DataFrame, candidate_zones: Sequence[int]) -> List[int]:
    candidate_set = set(int(zone_id) for zone_id in candidate_zones)
    mask = (
        step_df["PULocationID"].isin(candidate_set)
        & np.isfinite(step_df["base_pred"].to_numpy(dtype=float))
        & np.isfinite(step_df["true_value"].to_numpy(dtype=float))
    )
    zone_ids = sorted(int(zone_id) for zone_id in step_df.loc[mask, "PULocationID"])
    if len(zone_ids) < 3:
        raise ValueError("At least three finite baseline predictions are required.")
    return zone_ids


def make_location_split_sets(
    location_ids: Sequence[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> LocationSplitSets:
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train + val < 1.")

    unique_ids = np.array(sorted(set(int(loc_id) for loc_id in location_ids)), dtype=int)
    if unique_ids.size < 3:
        raise ValueError("At least three locations are required for train/val/test splits.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_ids)
    train_count = max(1, int(round(unique_ids.size * train_ratio)))
    val_count = max(1, int(round(unique_ids.size * val_ratio)))
    if train_count + val_count >= unique_ids.size:
        val_count = max(1, unique_ids.size - train_count - 1)
    test_count = unique_ids.size - train_count - val_count
    if test_count <= 0:
        train_count = max(1, unique_ids.size - val_count - 1)
        test_count = unique_ids.size - train_count - val_count
    if test_count <= 0:
        raise ValueError("Unable to create a non-empty test split.")

    return LocationSplitSets(
        train=set(int(loc_id) for loc_id in perm[:train_count]),
        val=set(int(loc_id) for loc_id in perm[train_count : train_count + val_count]),
        test=set(int(loc_id) for loc_id in perm[train_count + val_count :]),
    )


def masks_from_location_splits(
    data,
    split_sets: LocationSplitSets,
    require_train_val: bool = True,
    require_test: bool = False,
) -> SplitMasks:
    location_ids = data.location_id.cpu().numpy().astype(int)
    train = torch.tensor([loc_id in split_sets.train for loc_id in location_ids])
    val = torch.tensor([loc_id in split_sets.val for loc_id in location_ids])
    test = torch.tensor([loc_id in split_sets.test for loc_id in location_ids])
    if require_train_val and (int(train.sum()) == 0 or int(val.sum()) == 0):
        raise ValueError("Graph data does not contain non-empty train/val splits.")
    if require_test and int(test.sum()) == 0:
        raise ValueError("Graph data does not contain a non-empty test split.")
    return SplitMasks(train=train, val=val, test=test)


def split_label_for_location(location_id: int, split_sets: LocationSplitSets) -> str:
    if location_id in split_sets.train:
        return "train"
    if location_id in split_sets.val:
        return "val"
    if location_id in split_sets.test:
        return "test"
    return "unused"


def collect_no_graph_predictions(
    target_hour: pd.Timestamp,
    step_df: pd.DataFrame,
    split_sets: LocationSplitSets,
    spec: GraphSpec,
) -> pd.DataFrame:
    rows = step_df.copy()
    rows = rows[
        np.isfinite(rows["base_pred"].to_numpy(dtype=float))
        & np.isfinite(rows["true_value"].to_numpy(dtype=float))
    ].copy()
    rows["split"] = rows["PULocationID"].map(
        lambda zone_id: split_label_for_location(int(zone_id), split_sets)
    )
    rows = rows[rows["split"] != "unused"].copy()
    rows["is_test"] = rows["split"] == "test"
    rows["graph_type"] = spec.key
    rows["mode"] = spec.key
    rows["experiment_3_mode"] = "none"
    rows["Model"] = spec.model_name
    rows["random_seed"] = np.nan
    rows["edge_weight_used"] = False
    rows["residual_target"] = rows["true_value"] - rows["base_pred"]
    rows["residual_pred"] = 0.0
    rows["refined_pred"] = rows["base_pred"]
    rows["sample_weight"] = np.nan
    rows["best_epoch"] = 0
    rows["best_val_loss"] = np.nan
    rows["target_hour"] = target_hour
    return rows[
        [
            "target_hour",
            "graph_type",
            "mode",
            "experiment_3_mode",
            "Model",
            "random_seed",
            "edge_weight_used",
            "PULocationID",
            "split",
            "is_test",
            "base_pred",
            "true_value",
            "residual_target",
            "residual_pred",
            "refined_pred",
            "sample_weight",
            "prior_score",
            "stability_score",
            "history_consistency_score",
            "full_confidence",
            "best_epoch",
            "best_val_loss",
        ]
    ]


def hourly_record_from_predictions(
    target_hour: pd.Timestamp,
    predictions: pd.DataFrame,
    spec: GraphSpec,
) -> Dict[str, object]:
    test_df = predictions[predictions["is_test"]]
    metrics = evaluate_refined_predictions(
        y_true=test_df["true_value"].to_numpy(dtype=float),
        y_pred=test_df["refined_pred"].to_numpy(dtype=float),
    )
    return {
        "target_hour": target_hour,
        "graph_type": spec.key,
        "mode": spec.key,
        "experiment_3_mode": predictions["experiment_3_mode"].iloc[0],
        "Model": spec.model_name,
        "random_seed": spec.random_seed,
        "edge_weight_used": False,
        "hourly_mae": metrics["MAE"],
        "hourly_rmse": metrics["RMSE"],
        "hourly_mse": metrics["MSE"],
        "n_train": int((predictions["split"] == "train").sum()),
        "n_val": int((predictions["split"] == "val").sum()),
        "n_test": int((predictions["split"] == "test").sum()),
        "best_epoch": int(predictions["best_epoch"].max()),
        "best_val_loss": float(predictions["best_val_loss"].max(skipna=True)),
    }


def run_graph_variant(
    target_hour: pd.Timestamp,
    graph_key: str,
    graph: GraphContext,
    step_df: pd.DataFrame,
    history_frames: Sequence[pd.DataFrame],
    split_sets: LocationSplitSets,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
    spec: GraphSpec,
    use_edge_weight: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    inference_data = build_gnn_features(
        step_df=step_df,
        graph=graph,
        use_edge_weight=use_edge_weight,
    )
    inference_splits = masks_from_location_splits(
        data=inference_data,
        split_sets=split_sets,
        require_train_val=False,
        require_test=True,
    )
    train_data = build_historical_gnn_training_data(
        history_frames=history_frames,
        graph=graph,
        use_edge_weight=use_edge_weight,
    )
    train_splits: Optional[SplitMasks] = None
    if train_data is not None:
        try:
            train_splits = masks_from_location_splits(
                data=train_data,
                split_sets=split_sets,
                require_train_val=True,
                require_test=False,
            )
        except ValueError as exc:
            print(f"[{target_hour}] {graph_key} historical GNN training skipped: {exc}")
            train_data = None

    prediction_df, hourly_record = run_single_ablation(
        target_hour=target_hour,
        mode=GRAPH_CONFIDENCE_MODE,
        train_data=train_data,
        train_splits=train_splits,
        inference_data=inference_data,
        inference_splits=inference_splits,
        device=device,
        cfg=cfg,
        seed=seed,
    )

    prediction_df["graph_type"] = graph_key
    prediction_df["mode"] = graph_key
    prediction_df["experiment_3_mode"] = GRAPH_CONFIDENCE_MODE
    prediction_df["Model"] = spec.model_name
    prediction_df["random_seed"] = spec.random_seed
    prediction_df["edge_weight_used"] = bool(use_edge_weight)

    hourly_record["graph_type"] = graph_key
    hourly_record["mode"] = graph_key
    hourly_record["experiment_3_mode"] = GRAPH_CONFIDENCE_MODE
    hourly_record["Model"] = spec.model_name
    hourly_record["random_seed"] = spec.random_seed
    hourly_record["edge_weight_used"] = bool(use_edge_weight)
    return prediction_df, hourly_record


def summarize_results(
    detailed_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    specs: Dict[str, GraphSpec],
    graph_order: Sequence[str],
) -> pd.DataFrame:
    no_graph_mae = np.nan
    if "no_graph" in set(graph_order):
        no_graph_df = detailed_df[
            (detailed_df["graph_type"] == "no_graph") & detailed_df["is_test"]
        ]
        no_graph_metrics = evaluate_refined_predictions(
            y_true=no_graph_df["true_value"].to_numpy(dtype=float),
            y_pred=no_graph_df["refined_pred"].to_numpy(dtype=float),
        )
        no_graph_mae = no_graph_metrics["MAE"]

    rows: List[Dict[str, object]] = []
    for graph_key in graph_order:
        spec = specs[graph_key]
        test_df = detailed_df[
            (detailed_df["graph_type"] == graph_key) & detailed_df["is_test"]
        ]
        metrics = evaluate_refined_predictions(
            y_true=test_df["true_value"].to_numpy(dtype=float),
            y_pred=test_df["refined_pred"].to_numpy(dtype=float),
        )
        hourly_values = hourly_df.loc[
            hourly_df["graph_type"] == graph_key,
            "hourly_mae",
        ].dropna()
        mae_delta = metrics["MAE"] - no_graph_mae
        if np.isfinite(no_graph_mae) and not np.isclose(no_graph_mae, 0.0):
            mae_improvement_pct = (no_graph_mae - metrics["MAE"]) / no_graph_mae * 100.0
        else:
            mae_improvement_pct = np.nan

        rows.append(
            {
                "graph_type": graph_key,
                "Model": spec.model_name,
                "Description": spec.description,
                "random_seed": spec.random_seed,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MSE": metrics["MSE"],
                "Std of hourly MAE": (
                    float(np.std(hourly_values.to_numpy(dtype=float), ddof=0))
                    if not hourly_values.empty
                    else np.nan
                ),
                "Random seed MAE std": np.nan,
                "MAE delta vs no_graph": mae_delta,
                "MAE improvement vs no_graph (%)": mae_improvement_pct,
            }
        )
    summary_df = pd.DataFrame(rows)
    random_df = summary_df[summary_df["random_seed"].notna()]
    if len(random_df) > 1:
        mean_mae = float(random_df["MAE"].mean())
        mean_rmse = float(random_df["RMSE"].mean())
        mean_mse = float(random_df["MSE"].mean())
        if np.isfinite(no_graph_mae) and not np.isclose(no_graph_mae, 0.0):
            improvement_pct = (no_graph_mae - mean_mae) / no_graph_mae * 100.0
        else:
            improvement_pct = np.nan
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "graph_type": "random_mean",
                            "Model": "Random Graph mean over seeds",
                            "Description": "Mean over generated random graph seeds",
                            "random_seed": np.nan,
                            "MAE": mean_mae,
                            "RMSE": mean_rmse,
                            "MSE": mean_mse,
                            "Std of hourly MAE": float(random_df["Std of hourly MAE"].mean()),
                            "Random seed MAE std": float(random_df["MAE"].std(ddof=0)),
                            "MAE delta vs no_graph": mean_mae - no_graph_mae,
                            "MAE improvement vs no_graph (%)": improvement_pct,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return summary_df


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_random_seed(args.seed)
    if args.clean_checkpoints:
        clean_checkpoint_dir(args.checkpoint_dir)

    specs, requested_graphs = build_graph_specs(args)
    rolling_od = uses_rolling_od_graph(args=args, graph_order=requested_graphs)
    residual_graph_enabled = uses_residual_graph(graph_order=requested_graphs)
    generate_random_graphs(args=args, specs=specs, graph_order=requested_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("Graph variants:", ", ".join(requested_graphs))
    print("Use graph edge weights:", bool(args.use_edge_weight))
    if rolling_od:
        print(
            "Rolling OD graph window: "
            f"{args.od_lookback_days} days before each target hour"
        )
    if residual_graph_enabled:
        print(
            "Residual graph window: "
            f"{args.residual_window_hours} hours before each target hour"
        )
    if rolling_od and residual_graph_enabled:
        od_window_hours = int(args.od_lookback_days * 24)
        if od_window_hours == int(args.residual_window_hours):
            print(f"OD/residual graph history windows match: {od_window_hours} hours.")
        else:
            print(
                "WARNING: OD/residual graph history windows differ: "
                f"OD={od_window_hours} hours, "
                f"residual={args.residual_window_hours} hours."
            )

    df = load_taxi_data(
        data_path=args.data,
        excluded_zones=args.excluded_zones,
        include_dropoff=rolling_od,
    )
    window_starts = resolve_window_starts(args)
    validate_windows(df, window_starts, args.rolling_steps)
    print(
        "Window starts:",
        ", ".join(str(start) for start in window_starts),
    )
    lookup_df = pd.read_csv(args.lookup).drop_duplicates(subset="LocationID")
    graphs = load_graph_contexts(
        specs=specs,
        requested_graphs=requested_graphs,
        lookup_df=lookup_df,
        skip_graphs={"od"} if rolling_od else None,
    )
    rolling_od_cache: Dict[pd.Timestamp, GraphContext] = {}
    od_zone_names: List[str] = []
    od_location_to_zone: Dict[int, str] = {}
    if rolling_od:
        od_zone_names = load_od_zone_names(args)
        od_location_to_zone = _load_location_lookup(args.lookup)
        first_target_hour = window_starts[0]
        rolling_od_cache[first_target_hour] = build_rolling_od_graph_context(
            args=args,
            df=df,
            lookup_df=lookup_df,
            zone_names=od_zone_names,
            location_to_zone=od_location_to_zone,
            target_hour=first_target_hour,
        )
        graphs["od"] = rolling_od_cache[first_target_hour]
    residual_base_context: Optional[GraphContext] = None
    if residual_graph_enabled:
        residual_zone_names = load_od_zone_names(args)
        residual_base_context = build_zone_only_graph_context(
            zone_names=residual_zone_names,
            lookup_df=lookup_df,
        )
        graphs["residual"] = residual_base_context
    zones = select_shared_zones(df=df, graphs=graphs, max_zones=args.max_zones)
    zone_hourly_counts = build_zone_hourly_counts(df)

    manager_cfg = ManagerConfig(M_mc_test=args.mc_samples)
    manager = MultiScaleModelManager(checkpoint_dir=str(args.checkpoint_dir), cfg=manager_cfg)
    gnn_cfg = GNNTrainingConfig(
        hidden_dim=args.gnn_hidden_dim,
        dropout=args.gnn_dropout,
        learning_rate=args.gnn_lr,
        epochs=args.gnn_epochs,
        patience=args.gnn_patience,
    )

    detailed_frames: List[pd.DataFrame] = []
    hourly_records: List[Dict[str, object]] = []
    residual_prediction_cache: Dict[Tuple[int, pd.Timestamp], float] = {}
    residual_summary_records: List[Dict[str, object]] = []

    for window_idx, window_start in enumerate(window_starts, start=1):
        # Keep GNN residual-training snapshots local to this evaluation window.
        historical_step_frames: List[pd.DataFrame] = []
        print(
            f"\n===== Experiment 5 window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            print(
                f"\n///// Experiment 5 target hour: {target_hour} "
                f"window {window_idx} step {step} /////"
            )

            set_random_seed(args.seed + global_step)
            if rolling_od:
                if target_hour not in rolling_od_cache:
                    rolling_od_cache[target_hour] = build_rolling_od_graph_context(
                        args=args,
                        df=df,
                        lookup_df=lookup_df,
                        zone_names=od_zone_names,
                        location_to_zone=od_location_to_zone,
                        target_hour=target_hour,
                    )
                graphs["od"] = rolling_od_cache[target_hour]

            baseline_df = run_multiscale_temporal_baseline(
                df=df,
                manager=manager,
                target_hour=target_hour,
                zones=zones,
                zone_hourly_counts=zone_hourly_counts,
            )
            if residual_graph_enabled:
                cache_baseline_predictions(
                    prediction_cache=residual_prediction_cache,
                    baseline_df=baseline_df,
                )
                if residual_base_context is None:
                    raise RuntimeError("Residual graph context was not initialized.")
                residual_context, residual_summary = build_residual_graph_context(
                    args=args,
                    df=df,
                    manager=manager,
                    lookup_df=lookup_df,
                    target_hour=target_hour,
                    zones=zones,
                    zone_hourly_counts=zone_hourly_counts,
                    graph_context=residual_base_context,
                    prediction_cache=residual_prediction_cache,
                )
                residual_summary["window_id"] = window_idx
                residual_summary["window_start"] = window_start
                residual_summary["window_step"] = step
                graphs["residual"] = residual_context
                residual_summary_records.append(residual_summary)
                summary_log_path = residual_graph_summary_path(args)
                summary_log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(residual_summary_records).to_csv(summary_log_path, index=False)

            prior_scores = compute_prior_scores_from_zone_hourly_counts(
                zone_hourly_counts=zone_hourly_counts,
                target_hour=target_hour,
            )
            step_df = compute_confidence_components(
                step_df=baseline_df,
                prior_scores=prior_scores,
            )
            step_df["window_id"] = window_idx
            step_df["window_start"] = window_start
            step_df["window_step"] = step
            valid_locations = finite_step_zone_ids(step_df=step_df, candidate_zones=zones)
            split_sets = make_location_split_sets(
                location_ids=valid_locations,
                seed=args.seed + global_step,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            history_frames = filter_history_frames_before(historical_step_frames, target_hour)

            hour_messages: List[str] = []
            for graph_key in requested_graphs:
                spec = specs[graph_key]
                if graph_key == "no_graph":
                    prediction_df = collect_no_graph_predictions(
                        target_hour=target_hour,
                        step_df=step_df,
                        split_sets=split_sets,
                        spec=spec,
                    )
                    hourly_record = hourly_record_from_predictions(
                        target_hour=target_hour,
                        predictions=prediction_df,
                        spec=spec,
                    )
                else:
                    prediction_df, hourly_record = run_graph_variant(
                        target_hour=target_hour,
                        graph_key=graph_key,
                        graph=graphs[graph_key],
                        step_df=step_df,
                        history_frames=history_frames,
                        split_sets=split_sets,
                        device=device,
                        cfg=gnn_cfg,
                        seed=args.seed + global_step,
                        spec=spec,
                        use_edge_weight=args.use_edge_weight,
                    )

                prediction_df["window_id"] = window_idx
                prediction_df["window_start"] = window_start
                prediction_df["window_step"] = step
                hourly_record.update(
                    {
                        "window_id": window_idx,
                        "window_start": window_start,
                        "window_step": step,
                    }
                )
                detailed_frames.append(prediction_df)
                hourly_records.append(hourly_record)
                hour_messages.append(f"{graph_key} MAE={hourly_record['hourly_mae']:.4f}")

            print(f"[{target_hour}] " + ", ".join(hour_messages))
            historical_step_frames.append(step_df.copy())

    detailed_df = pd.concat(detailed_frames, ignore_index=True)
    hourly_df = pd.DataFrame(hourly_records)
    summary_df = summarize_results(
        detailed_df=detailed_df,
        hourly_df=hourly_df,
        specs=specs,
        graph_order=requested_graphs,
    )
    return summary_df, detailed_df, hourly_df


def save_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    results_dir: Path,
    make_plot: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "experiment_5_graph_structure_comparison_summary.csv"
    detailed_path = results_dir / "experiment_5_graph_structure_comparison_detailed.csv"
    hourly_path = results_dir / "experiment_5_graph_structure_comparison_hourly_mae.csv"
    plot_path = results_dir / "experiment_5_graph_structure_comparison_metrics.png"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)

    plot_written = False
    if make_plot:
        try:
            plot_metrics(
                summary_df=summary_df,
                hourly_df=hourly_df,
                plot_path=plot_path,
            )
            plot_written = True
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed predictions to {detailed_path}")
    print(f"Saved hourly MAE to {hourly_path}")
    if plot_written:
        print(f"Saved plot to {plot_path}")


def plot_metrics(summary_df: pd.DataFrame, hourly_df: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = summary_df["Model"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(x - width, summary_df["MAE"], width, label="MAE")
    axes[0].bar(x, summary_df["RMSE"], width, label="RMSE")
    axes[0].bar(x + width, summary_df["MSE"], width, label="MSE")
    axes[0].set_title("Overall Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].bar(labels, summary_df["Std of hourly MAE"])
    axes[1].set_title("Std of Hourly MAE")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for graph_key in summary_df["graph_type"].tolist():
        graph_hourly = hourly_df[hourly_df["graph_type"] == graph_key].sort_values(
            "target_hour"
        )
        if graph_hourly.empty:
            continue
        model_name = summary_df.loc[summary_df["graph_type"] == graph_key, "Model"].iloc[0]
        axes[2].plot(
            pd.to_datetime(graph_hourly["target_hour"]),
            graph_hourly["hourly_mae"],
            marker="o",
            linewidth=1.5,
            label=model_name,
        )
    axes[2].set_title("Hourly MAE Over Rolling Hours")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8)

    fig.suptitle("Experiment 5: Graph Structure Comparison")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    columns = [
        "Model",
        "MAE",
        "RMSE",
        "MSE",
        "Std of hourly MAE",
        "Random seed MAE std",
        "MAE improvement vs no_graph (%)",
    ]
    display_df = summary_df[columns].copy()
    for column in columns[1:]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "nan"
        )
    print("\nFinal Summary Table")
    print(display_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    summary_df, detailed_df, hourly_df = run_experiment(args)
    save_results(
        summary_df=summary_df,
        detailed_df=detailed_df,
        hourly_df=hourly_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
