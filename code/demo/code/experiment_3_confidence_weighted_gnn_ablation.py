"""Experiment 3: confidence-weighted GNN ablation.

This experiment keeps the residual refinement setting from Experiment 2:

    residual_target = true_value - base_pred
    residual_pred = GraphSAGE(features, edge_index)
    refined_pred = base_pred + residual_pred

The ablation changes only the residual loss sample weights. All modes use the
same temporal baseline, graph, node features, GNN architecture, splits, epochs,
learning rate, and initialization seed.

For each rolling target hour, the GNN is trained only on residual snapshots from
hours strictly before the target. Target-hour labels are retained only for
post-inference metrics.

Compared modes:

    none: no confidence weighting
    prior_only: historical-data prior only
    stability_only: MC-variance stability only
    history_consistency_only: 24h/168h/720h historical mean consistency only
    full: fixed prior/stability/history-consistency weights from Experiment 2
    learned_softmax: trainable softmax weights over the three confidence components
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

from confidence_softmax import (
    CONFIDENCE_COMPONENT_NAMES,
    DEFAULT_CONFIDENCE_WEIGHTS,
    combine_confidence_components,
    confidence_from_component_weights,
)
from gnn_model import MultiScaleGraphSAGE
from persistent_tcn import ManagerConfig, MultiScaleModelManager
from residual_graph_utils import (
    build_dynamic_residual_graph_context,
    residual_graph_summary_path,
)
from rolling_od_graph_utils import (
    add_rolling_od_args,
    build_rolling_od_graph_context,
    load_od_location_lookup,
    load_od_zone_names,
    uses_rolling_od_graph,
    validate_rolling_od_args,
)


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="Incremental updates are disabled; using checkpoint trained through*",
    category=RuntimeWarning,
)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PATH = BASE_DIR / "data.parquet"
DEFAULT_LOOKUP_PATH = BASE_DIR / "taxi-zone-lookup.csv"
DEFAULT_EDGE_WEIGHT_MATRIX = BASE_DIR / "edge_weight_matrix_od.csv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_tcn_shared_v1"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_GRAPH_TYPE = "od"
DEFAULT_RESIDUAL_WINDOW_HOURS = 168

START_TARGET = pd.Timestamp("2021-07-05 00:00")
ROLLING_STEPS = 24
DEFAULT_WINDOW_STARTS = (
    "2021-03-01 00:00",
    "2021-03-29 00:00",
    "2021-04-26 00:00",
    "2021-05-24 00:00",
    "2021-06-21 00:00",
    "2021-07-05 00:00",
    "2021-07-19 00:00",
    "2021-08-16 00:00",
    "2021-09-13 00:00",
    "2021-10-11 00:00",
    "2021-11-08 00:00",
    "2021-12-06 00:00",
)
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]

HISTORY_WINDOWS = {
    "mean_24h": 24,
    "mean_168h": 24 * 7,
    "mean_720h": 720,
}
HISTORY_FEATURES = list(HISTORY_WINDOWS.keys())
HISTORY_CONSISTENCY_TAU = 1.0

CONFIDENCE_WEIGHTS = {
    name: float(weight)
    for name, weight in zip(CONFIDENCE_COMPONENT_NAMES, DEFAULT_CONFIDENCE_WEIGHTS)
}

ABLATION_MODES = {
    "none": {
        "model": "GNN without confidence",
        "description": "sample weight = 1 for all nodes / samples",
    },
    "prior_only": {
        "model": "GNN + prior only",
        "description": "only use prior score as sample weight",
    },
    "stability_only": {
        "model": "GNN + stability only",
        "description": "only use MC Dropout stability score as sample weight",
    },
    "history_consistency_only": {
        "model": "GNN + history consistency only",
        "description": "only use 24h/168h/720h consistency score as sample weight",
    },
    "full": {
        "model": "GNN + fixed full confidence",
        "description": (
            "use fixed log-space prior/stability/history-consistency weights "
            "0.3/0.4/0.3"
        ),
    },
    "learned_softmax": {
        "model": "GNN + learned softmax confidence",
        "description": "learn prior/stability/history-consistency weights jointly with GraphSAGE",
    },
}
MODE_ORDER = [
    "none",
    "prior_only",
    "stability_only",
    "history_consistency_only",
    "full",
    "learned_softmax",
]


@dataclass(frozen=True)
class GraphContext:
    """Static graph metadata shared by all rolling hours."""

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    zone_names: List[str]
    zone_idx_map: Dict[str, int]
    location_to_zone: Dict[int, str]
    zone_to_location: Dict[str, int]


@dataclass(frozen=True)
class SplitMasks:
    """Node-level train/validation/test split for one rolling graph snapshot."""

    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


@dataclass(frozen=True)
class LocationSplitSets:
    """Location-level split reused across historical training and target inference."""

    train: set[int]
    val: set[int]
    test: set[int]


@dataclass(frozen=True)
class GNNTrainingConfig:
    """GraphSAGE hyperparameters shared by every ablation mode."""

    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 0.01
    epochs: int = 300
    patience: int = 40


@dataclass
class GNNResult:
    """Predictions and validation diagnostics for one ablation mode."""

    residual_pred: np.ndarray
    refined_pred: np.ndarray
    best_epoch: int
    best_val_loss: float
    confidence_weights: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 3: confidence-weighted GNN ablation."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_WEIGHT_MATRIX)
    add_rolling_od_args(parser)
    parser.add_argument(
        "--graph-type",
        choices=["residual", "od"],
        default=DEFAULT_GRAPH_TYPE,
        help=(
            "Graph used by residual GraphSAGE. Default uses rolling OD graphs; "
            "set --od-lookback-days 0 to use the static --edge-csv matrix."
        ),
    )
    parser.add_argument(
        "--residual-window-hours",
        type=int,
        default=DEFAULT_RESIDUAL_WINDOW_HOURS,
        help="Historical base-residual window used to build each residual graph.",
    )
    parser.add_argument("--residual-top-k", type=int, default=10)
    parser.add_argument("--residual-min-corr", type=float, default=0.0)
    parser.add_argument("--residual-use-signed-corr", action="store_true")
    parser.add_argument("--residual-symmetrize", action="store_true")
    parser.add_argument(
        "--residual-graph-dir",
        type=Path,
        default=None,
        help="Directory for generated residual graph CSVs. Defaults to results/residual_graphs.",
    )
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
        "--clean-checkpoints",
        action="store_true",
        help="Delete this experiment's checkpoint directory before running.",
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=None,
        help="Optional smoke-test limit. Default evaluates all graph-compatible zones.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing experiment_3_confidence_weighted_gnn_ablation_metrics.png.",
    )
    args = parser.parse_args()
    if args.residual_window_hours <= 0:
        parser.error("--residual-window-hours must be > 0.")
    if args.residual_top_k < 0:
        parser.error("--residual-top-k must be >= 0.")
    if args.residual_min_corr < 0:
        parser.error("--residual-min-corr must be >= 0.")
    validate_rolling_od_args(parser, args)
    return args


def normalize_window_starts(values: Optional[Sequence[str]]) -> List[pd.Timestamp]:
    raw_values = list(values) if values else list(DEFAULT_WINDOW_STARTS)
    starts = sorted({pd.Timestamp(value).floor("h") for value in raw_values})
    if not starts:
        raise ValueError("At least one window start is required.")
    return starts


def resolve_window_starts(args: argparse.Namespace) -> List[pd.Timestamp]:
    if args.window_starts is not None:
        return normalize_window_starts(args.window_starts)
    if pd.Timestamp(args.start_target) != START_TARGET:
        return [pd.Timestamp(args.start_target).floor("h")]
    return normalize_window_starts(None)


def validate_windows(
    df: pd.DataFrame,
    window_starts: Sequence[pd.Timestamp],
    rolling_steps: int,
) -> None:
    if rolling_steps <= 0:
        raise ValueError("--rolling-steps must be positive.")

    min_ts = df["datetime"].min()
    max_ts = df["datetime"].max()
    for start in window_starts:
        end = start + pd.Timedelta(hours=rolling_steps - 1)
        if start < min_ts or end > max_ts:
            raise ValueError(
                f"Window {start} to {end} is outside data range {min_ts} to {max_ts}."
            )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_required_data(
    data_path: Path,
    lookup_path: Path,
    edge_csv: Path,
    excluded_zones: Sequence[int],
    include_dropoff: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, GraphContext]:
    """Load taxi trips, lookup metadata, and the OD-flow graph."""

    columns = ["pickup_datetime", "PULocationID"]
    if include_dropoff:
        columns.append("DOLocationID")
    df = pd.read_parquet(data_path, columns=columns)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(excluded_zones)].copy()

    lookup_df = pd.read_csv(lookup_path).drop_duplicates(subset="LocationID")
    graph = build_graph_context(edge_csv=edge_csv, lookup_df=lookup_df)

    print("Earliest timestamp:", df["datetime"].min())
    print("Latest timestamp:", df["datetime"].max())
    print("Total hours:", df["datetime"].nunique())
    print("Total non-excluded zones:", df["PULocationID"].nunique())
    return df, lookup_df, graph


def build_graph_context(edge_csv: Path, lookup_df: pd.DataFrame) -> GraphContext:
    """Build graph tensors and zone mappings using the same OD matrix as the demo."""

    df_adj = pd.read_csv(edge_csv, index_col=0)
    df_adj.index = [str(idx).lstrip("\ufeff") for idx in df_adj.index]
    df_adj.columns = [str(col).lstrip("\ufeff") for col in df_adj.columns]
    df_adj = df_adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)
    edge_index, edge_weight = dense_to_sparse(adj_matrix)

    zone_names = df_adj.index.tolist()
    zone_idx_map = {zone_name: idx for idx, zone_name in enumerate(zone_names)}

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
        edge_index=edge_index,
        edge_weight=edge_weight.float(),
        zone_names=zone_names,
        zone_idx_map=zone_idx_map,
        location_to_zone=location_to_zone,
        zone_to_location=zone_to_location,
    )


def select_zones(df: pd.DataFrame, graph: GraphContext, max_zones: Optional[int]) -> List[int]:
    """Select zones that can be mapped into the OD-flow graph."""

    zones: List[int] = []
    for zone_id in sorted(int(zone_id) for zone_id in df["PULocationID"].unique()):
        zone_name = graph.location_to_zone.get(zone_id)
        if zone_name is None:
            continue
        if zone_name not in graph.zone_idx_map:
            continue
        zones.append(zone_id)

    if max_zones is not None:
        zones = zones[: max(1, max_zones)]
        print(f"Using first {len(zones)} graph-compatible zones for smoke-test run.")
    return zones


def get_true_counts(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def build_zone_hourly_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["PULocationID", "datetime"]).size().rename("count")


def compute_history_means(
    zone_hourly_counts: pd.Series,
    zone_id: int,
    target_hour: pd.Timestamp,
) -> Dict[str, float]:
    """Compute historical means strictly before target_hour.

    Missing hours are treated as zero by dividing by the full window length. This
    is a conservative fallback for sparse hourly count data and uses no future
    observations.
    """

    means = {name: 0.0 for name in HISTORY_FEATURES}
    try:
        zone_series = zone_hourly_counts.loc[zone_id]
    except KeyError:
        return means

    for feature_name, hours in HISTORY_WINDOWS.items():
        start = target_hour - pd.Timedelta(hours=hours)
        window = zone_series[(zone_series.index >= start) & (zone_series.index < target_hour)]
        total = float(window.sum()) if not window.empty else 0.0
        means[feature_name] = total / float(hours)
    return means


def run_multiscale_temporal_baseline(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
) -> pd.DataFrame:
    """Generate leakage-free base_pred from persistent_tcn.py."""

    y_true_dict = get_true_counts(df, target_hour)
    records: List[Dict[str, object]] = []

    total_zones = len(zones)
    for zone_pos, zone_id in enumerate(zones, start=1):
        try:
            zone_int = int(zone_id)
            print(
                f"[{target_hour}] baseline zone {zone_pos}/{total_zones}: {zone_int}",
                flush=True,
            )
            guard_against_future_checkpoint(manager, zone_int, target_hour)
            context_end = target_hour - manager._forecast_delta
            if not manager.has_checkpoint(zone_int):
                print(
                    f"[{target_hour}] training missing TCN checkpoint for zone {zone_int}",
                    flush=True,
                )
                manager.train_once(df, zone_int, context_end)
            point, std, _ = manager.predict_with_uncertainty(df, zone_int, target_hour)
            true_value = float(y_true_dict.get(zone_id, 0.0))
            history_means = compute_history_means(zone_hourly_counts, zone_int, target_hour)
            records.append(
                {
                    "target_hour": target_hour,
                    "PULocationID": zone_int,
                    "base_pred": float(point),
                    "true_value": true_value,
                    "mc_std": float(std),
                    "variance": float(std**2),
                    "error": "",
                    **history_means,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[diag] zone={zone_id} target={target_hour} "
                f"error={type(exc).__name__}: {exc}"
            )
            records.append(
                {
                    "target_hour": target_hour,
                    "PULocationID": int(zone_id),
                    "base_pred": np.nan,
                    "true_value": np.nan,
                    "mc_std": np.nan,
                    "variance": np.nan,
                    "error": str(exc),
                    **{feature: np.nan for feature in HISTORY_FEATURES},
                }
            )

    return pd.DataFrame(records)


def guard_against_future_checkpoint(
    manager: MultiScaleModelManager,
    zone_id: int,
    target_hour: pd.Timestamp,
) -> None:
    """Prevent reusing a temporal checkpoint trained beyond the target context."""

    if not manager.has_checkpoint(zone_id):
        return

    context_end = target_hour - manager._forecast_delta
    trained_until = manager._load_meta(zone_id)
    if trained_until is not None and trained_until > context_end:
        raise RuntimeError(
            f"Checkpoint for zone {zone_id} was trained through {trained_until}, "
            f"which is after context_end={context_end}. Re-run with "
            "--clean-checkpoints or a fresh --checkpoint-dir."
        )


def clamp_score(value: float, default: float) -> float:
    if not np.isfinite(value):
        value = default
    return float(np.clip(value, 0.05, 1.0))


def compute_prior_scores(history_df: pd.DataFrame) -> Dict[int, float]:
    """Prior confidence: zones with richer historical samples get higher weight."""

    counts = history_df.groupby("PULocationID").size().astype(float)
    return normalize_prior_counts(counts)


def compute_prior_scores_from_zone_hourly_counts(
    zone_hourly_counts: pd.Series,
    target_hour: pd.Timestamp,
) -> Dict[int, float]:
    """Compute prior confidence without repeatedly filtering the trip dataframe."""

    if zone_hourly_counts.empty:
        return {}
    datetime_index = zone_hourly_counts.index.get_level_values("datetime")
    counts = (
        zone_hourly_counts[datetime_index < pd.Timestamp(target_hour)]
        .groupby(level=0)
        .sum()
        .astype(float)
    )
    return normalize_prior_counts(counts)


def normalize_prior_counts(counts: pd.Series) -> Dict[int, float]:
    if counts.empty:
        return {}

    log_counts = np.log1p(counts)
    vmin, vmax = float(log_counts.min()), float(log_counts.max())
    if np.isclose(vmin, vmax):
        normalized = pd.Series(1.0, index=log_counts.index)
    else:
        normalized = (log_counts - vmin) / (vmax - vmin)

    scaled = 0.2 + 0.8 * normalized
    return {int(idx): clamp_score(float(val), default=0.4) for idx, val in scaled.items()}


def compute_stability_scores(step_df: pd.DataFrame) -> Dict[int, float]:
    """Stability confidence: lower MC Dropout variance means higher confidence."""

    finite_variance = step_df["variance"].dropna()
    if finite_variance.empty:
        return {int(row.PULocationID): 1.0 for row in step_df.itertuples()}

    scale = max(float(np.median(finite_variance)), 1e-3)
    scores: Dict[int, float] = {}
    for row in step_df.itertuples():
        zone_id = int(row.PULocationID)
        variance = float(row.variance)
        if np.isfinite(variance):
            raw = float(np.exp(-variance / (3.0 * scale)))
            scores[zone_id] = clamp_score(raw, default=0.5)
        else:
            scores[zone_id] = 0.2
    return scores


def compute_history_consistency_scores(step_df: pd.DataFrame) -> Dict[int, float]:
    """History confidence: higher when 24h/168h/720h means agree."""
    scores: Dict[int, float] = {}
    for row in step_df.itertuples():
        values = []
        for feature_name in HISTORY_FEATURES:
            value = float(getattr(row, feature_name, np.nan))
            if np.isfinite(value):
                values.append(max(value, 0.0))

        zone_id = int(row.PULocationID)
        if len(values) < 2:
            scores[zone_id] = 0.5
            continue

        log_values = np.log1p(np.array(values, dtype=np.float32))
        dispersion = float(log_values.std())
        raw = float(np.exp(-dispersion / HISTORY_CONSISTENCY_TAU))
        scores[zone_id] = clamp_score(raw, default=0.6)
    return scores


def compute_confidence_components(
    step_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
    prior_scores: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Attach confidence components needed by the ablation.

    persistent_tcn.py in this repo exposes MC uncertainty but does
    not expose _assign_confidence_scores(). Importing demo_rolling_GNN.py would
    execute top-level checkpoint cleanup, so this wrapper mirrors the component
    definitions locally and returns each component separately for ablation.
    """

    step_df = step_df.copy()
    if prior_scores is None:
        if history_df is None:
            raise ValueError("Either history_df or prior_scores is required.")
        prior_scores = compute_prior_scores(history_df)
    stability_scores = compute_stability_scores(step_df)
    history_scores = compute_history_consistency_scores(step_df)

    step_df["prior_score"] = step_df["PULocationID"].map(
        lambda zone_id: clamp_score(prior_scores.get(int(zone_id), 0.4), default=0.4)
    )
    step_df["stability_score"] = step_df["PULocationID"].map(
        lambda zone_id: clamp_score(stability_scores.get(int(zone_id), 0.5), default=0.5)
    )
    step_df["history_consistency_score"] = step_df["PULocationID"].map(
        lambda zone_id: clamp_score(
            history_scores.get(int(zone_id), 0.6),
            default=0.6,
        )
    )
    step_df["full_confidence"] = step_df.apply(
        lambda row: combine_confidence_components(
            prior=float(row.prior_score),
            stability=float(row.stability_score),
            history_consistency=float(row.history_consistency_score),
            weights=CONFIDENCE_WEIGHTS,
        ),
        axis=1,
    )
    return step_df


def build_gnn_features(
    step_df: pd.DataFrame,
    graph: GraphContext,
    use_edge_weight: bool = False,
) -> Data:
    """Build mode-invariant node features and residual targets.

    For a fair confidence ablation, every mode uses exactly the same node
    features:

        base_pred, full_confidence

    full_confidence already includes the history-consistency component computed
    from mean_24h/mean_168h/mean_720h, so the raw historical means are not
    duplicated in x.
    """

    node_count = len(graph.zone_names)
    node_pred = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_label = torch.full((node_count,), float("nan"), dtype=torch.float32)
    prior = torch.full((node_count,), 0.4, dtype=torch.float32)
    stability = torch.full((node_count,), 0.5, dtype=torch.float32)
    history_consistency = torch.full((node_count,), 0.6, dtype=torch.float32)
    full_confidence = torch.full((node_count,), 0.5, dtype=torch.float32)
    location_id = torch.full((node_count,), -1, dtype=torch.long)

    for row in step_df.itertuples():
        loc_id = int(row.PULocationID)
        zone_name = graph.location_to_zone.get(loc_id)
        if zone_name is None:
            continue
        node_idx = graph.zone_idx_map.get(zone_name)
        if node_idx is None:
            continue

        node_pred[node_idx] = float(row.base_pred)
        node_label[node_idx] = float(row.true_value)
        location_id[node_idx] = loc_id
        prior[node_idx] = clamp_score(float(row.prior_score), default=0.4)
        stability[node_idx] = clamp_score(float(row.stability_score), default=0.5)
        history_consistency[node_idx] = clamp_score(
            float(row.history_consistency_score),
            default=0.6,
        )
        full_confidence[node_idx] = clamp_score(float(row.full_confidence), default=0.5)

    valid_indices = torch.where(~torch.isnan(node_pred) & ~torch.isnan(node_label))[0]
    if valid_indices.numel() < 3:
        raise ValueError("Not enough valid graph nodes to train/evaluate residual GNN.")

    edge_index, edge_weight = remap_edges_and_weights_to_valid_nodes(
        edge_index=graph.edge_index,
        valid_indices=valid_indices,
        edge_weight=graph.edge_weight if use_edge_weight else None,
    )
    node_pred = node_pred[valid_indices]
    node_label = node_label[valid_indices]
    prior = prior[valid_indices]
    stability = stability[valid_indices]
    history_consistency = history_consistency[valid_indices]
    full_confidence = full_confidence[valid_indices]
    location_id = location_id[valid_indices]

    x_feat = torch.cat(
        [
            node_pred.unsqueeze(1),
            full_confidence.unsqueeze(1),
        ],
        dim=1,
    )
    residual_target = node_label - node_pred

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        y=residual_target,
        base_pred=node_pred,
        true_value=node_label,
        location_id=location_id,
        prior_score=prior,
        stability_score=stability,
        history_consistency_score=history_consistency,
        full_confidence=full_confidence,
    )
    if edge_weight is not None:
        data.edge_weight = edge_weight
    return data


def build_historical_gnn_training_data(
    history_frames: Sequence[pd.DataFrame],
    graph: GraphContext,
    use_edge_weight: bool = False,
) -> Optional[Data]:
    """Concatenate historical residual snapshots into one leakage-free train graph."""

    snapshots: List[Data] = []
    for frame in history_frames:
        if frame.empty:
            continue
        try:
            snapshots.append(
                build_gnn_features(
                    step_df=frame,
                    graph=graph,
                    use_edge_weight=use_edge_weight,
                )
            )
        except ValueError:
            continue

    if not snapshots:
        return None

    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    true_parts: List[torch.Tensor] = []
    location_parts: List[torch.Tensor] = []
    prior_parts: List[torch.Tensor] = []
    stability_parts: List[torch.Tensor] = []
    history_consistency_parts: List[torch.Tensor] = []
    full_confidence_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    edge_weight_parts: List[torch.Tensor] = []

    offset = 0
    for snapshot in snapshots:
        x_parts.append(snapshot.x)
        y_parts.append(snapshot.y)
        base_parts.append(snapshot.base_pred)
        true_parts.append(snapshot.true_value)
        location_parts.append(snapshot.location_id)
        prior_parts.append(snapshot.prior_score)
        stability_parts.append(snapshot.stability_score)
        history_consistency_parts.append(snapshot.history_consistency_score)
        full_confidence_parts.append(snapshot.full_confidence)

        if snapshot.edge_index.numel() > 0:
            edge_parts.append(snapshot.edge_index + offset)
            if use_edge_weight and hasattr(snapshot, "edge_weight"):
                edge_weight_parts.append(snapshot.edge_weight)
        offset += int(snapshot.num_nodes)

    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=edge_index,
        y=torch.cat(y_parts, dim=0),
        base_pred=torch.cat(base_parts, dim=0),
        true_value=torch.cat(true_parts, dim=0),
        location_id=torch.cat(location_parts, dim=0),
        prior_score=torch.cat(prior_parts, dim=0),
        stability_score=torch.cat(stability_parts, dim=0),
        history_consistency_score=torch.cat(history_consistency_parts, dim=0),
        full_confidence=torch.cat(full_confidence_parts, dim=0),
    )
    if use_edge_weight:
        data.edge_weight = (
            torch.cat(edge_weight_parts, dim=0)
            if edge_weight_parts
            else torch.empty((0,), dtype=torch.float32)
        )
    return data


def filter_history_frames_before(
    history_frames: Sequence[pd.DataFrame],
    target_hour: pd.Timestamp,
) -> List[pd.DataFrame]:
    """Return only cached snapshots whose label hour is strictly before target_hour."""

    filtered: List[pd.DataFrame] = []
    target_ts = pd.Timestamp(target_hour)
    for frame in history_frames:
        if frame.empty:
            continue
        if "target_hour" not in frame.columns:
            filtered.append(frame)
            continue
        frame_hours = pd.to_datetime(frame["target_hour"])
        if frame_hours.max() < target_ts:
            filtered.append(frame)
    return filtered


def remap_edges_and_weights_to_valid_nodes(
    edge_index: torch.Tensor,
    valid_indices: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    valid_old = [int(idx) for idx in valid_indices.cpu().tolist()]
    valid_set = set(valid_old)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_old)}

    remapped_edges: List[Tuple[int, int]] = []
    remapped_weights: List[float] = []
    for edge_pos, (src, dst) in enumerate(edge_index.t().tolist()):
        src_i, dst_i = int(src), int(dst)
        if src_i in valid_set and dst_i in valid_set:
            remapped_edges.append((old_to_new[src_i], old_to_new[dst_i]))
            if edge_weight is not None:
                remapped_weights.append(float(edge_weight[edge_pos].item()))

    if not remapped_edges:
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_edge_weight = (
            torch.empty((0,), dtype=torch.float32) if edge_weight is not None else None
        )
        return empty_edge_index, empty_edge_weight

    remapped_edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
    if edge_weight is None:
        return remapped_edge_index, None
    return remapped_edge_index, torch.tensor(remapped_weights, dtype=torch.float32)


def remap_edges_to_valid_nodes(
    edge_index: torch.Tensor,
    valid_indices: torch.Tensor,
) -> torch.Tensor:
    remapped_edge_index, _ = remap_edges_and_weights_to_valid_nodes(
        edge_index=edge_index,
        valid_indices=valid_indices,
    )
    return remapped_edge_index


def get_sample_weights(mode: str, data: Data) -> torch.Tensor:
    """Return sample weights for one ablation mode, clamped to [0.05, 1.0]."""

    if mode == "none":
        weights = torch.ones_like(data.y)
    elif mode == "prior_only":
        weights = data.prior_score
    elif mode == "stability_only":
        weights = data.stability_score
    elif mode == "history_consistency_only":
        weights = data.history_consistency_score
    elif mode == "full":
        weights = data.full_confidence
    elif mode == "learned_softmax":
        raise ValueError("learned_softmax weights are created during GNN training.")
    else:
        raise ValueError(f"Unsupported confidence weighting mode: {mode}")
    return weights.float().clamp(min=0.05, max=1.0)


def make_node_splits(
    node_count: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> SplitMasks:
    if node_count < 3:
        raise ValueError("At least three nodes are required for train/val/test splits.")
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio >= 0, and train + val < 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(node_count)

    train_count = max(1, int(round(node_count * train_ratio)))
    val_count = max(1, int(round(node_count * val_ratio)))
    if train_count + val_count >= node_count:
        val_count = max(1, node_count - train_count - 1)
    test_count = node_count - train_count - val_count
    if test_count <= 0:
        train_count = max(1, node_count - val_count - 1)
        test_count = node_count - train_count - val_count
    if test_count <= 0:
        raise ValueError("Unable to create a non-empty test split.")

    train_idx = perm[:train_count]
    val_idx = perm[train_count : train_count + val_count]
    test_idx = perm[train_count + val_count :]

    train_mask = torch.zeros(node_count, dtype=torch.bool)
    val_mask = torch.zeros(node_count, dtype=torch.bool)
    test_mask = torch.zeros(node_count, dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True
    return SplitMasks(train=train_mask, val=val_mask, test=test_mask)


def make_location_split_sets(
    location_ids: Sequence[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> LocationSplitSets:
    """Create stable location cohorts without reading target-hour labels."""

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
    data: Data,
    split_sets: LocationSplitSets,
    require_train_val: bool = True,
    require_test: bool = False,
) -> SplitMasks:
    """Map location cohorts onto a graph snapshot or concatenated history graph."""

    location_ids = data.location_id.cpu().numpy().astype(int)
    train = torch.tensor([loc_id in split_sets.train for loc_id in location_ids])
    val = torch.tensor([loc_id in split_sets.val for loc_id in location_ids])
    test = torch.tensor([loc_id in split_sets.test for loc_id in location_ids])

    if require_train_val and (int(train.sum()) == 0 or int(val.sum()) == 0):
        raise ValueError("Graph data does not contain non-empty train/val splits.")
    if require_test and int(test.sum()) == 0:
        raise ValueError("Graph data does not contain a non-empty test split.")
    return SplitMasks(train=train, val=val, test=test)


def train_residual_gnn_with_weighted_loss(
    train_data: Optional[Data],
    train_splits: Optional[SplitMasks],
    inference_data: Data,
    mode: str,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> GNNResult:
    """Train GraphSAGE with confidence-weighted residual SmoothL1 loss.

    Only historical residual labels on train nodes contribute to optimization:

        loss_per_sample = smooth_l1(residual_pred, residual_target)
        weighted_loss = mean(sample_weight * loss_per_sample)

    The target-hour graph is passed separately as inference_data and is never
    used to compute the training or validation loss.
    """

    set_random_seed(seed)

    if (
        train_data is None
        or train_splits is None
        or int(train_splits.train.sum()) == 0
        or int(train_splits.val.sum()) == 0
    ):
        confidence_weights = (
            np.array(DEFAULT_CONFIDENCE_WEIGHTS, dtype=np.float32)
            if mode == "learned_softmax"
            else None
        )
        base_pred = inference_data.base_pred.detach().cpu().numpy()
        return GNNResult(
            residual_pred=np.zeros_like(base_pred, dtype=np.float32),
            refined_pred=base_pred.astype(np.float32),
            best_epoch=0,
            best_val_loss=float("nan"),
            confidence_weights=confidence_weights,
        )

    model = MultiScaleGraphSAGE(
        in_dim=int(train_data.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    train_data_device = copy.deepcopy(train_data).to(device)
    train_mask = train_splits.train.to(device)
    val_mask = train_splits.val.to(device)
    sample_weight = (
        torch.ones_like(train_data_device.y)
        if mode == "learned_softmax"
        else get_sample_weights(mode, train_data_device)
    )

    logits: Optional[nn.Parameter] = None
    optimizer_params = list(model.parameters())
    if mode == "learned_softmax":
        logits = nn.Parameter(
            torch.log(
                torch.tensor(DEFAULT_CONFIDENCE_WEIGHTS, dtype=torch.float32)
            ).to(device)
        )
        optimizer_params.append(logits)

    optimizer = torch.optim.Adam(optimizer_params, lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_func = nn.SmoothL1Loss(reduction="none")

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_logits: Optional[torch.Tensor] = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        residual_pred, _ = model(train_data_device)

        loss_per_node = loss_func(
            residual_pred[train_mask],
            train_data_device.y[train_mask],
        )
        if logits is not None:
            current_weights = torch.softmax(logits, dim=0)
            sample_weight = confidence_from_component_weights(
                train_data_device,
                current_weights,
            )
        train_weight = sample_weight[train_mask]
        train_weight = train_weight / train_weight.mean().clamp(min=1e-6)
        loss = (train_weight * loss_per_node).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_residual_pred, _ = model(train_data_device)
            val_loss = loss_func(
                val_residual_pred[val_mask],
                train_data_device.y[val_mask],
            ).mean()
            val_loss_value = float(val_loss.item())

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if logits is not None:
                best_logits = logits.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if logits is not None and best_logits is not None:
        logits.data.copy_(best_logits)

    learned_weights = (
        torch.softmax(logits.detach(), dim=0).cpu().numpy()
        if logits is not None
        else None
    )

    model.eval()
    inference_data_device = copy.deepcopy(inference_data).to(device)
    with torch.no_grad():
        residual_pred, refined_pred = model(inference_data_device)

    return GNNResult(
        residual_pred=residual_pred.detach().cpu().numpy(),
        refined_pred=refined_pred.detach().cpu().numpy(),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        confidence_weights=learned_weights,
    )


def evaluate_refined_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}

    errors = y_pred[mask] - y_true[mask]
    mse = float(np.mean(errors**2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
    }


def compute_hourly_mae_std(hourly_mae_df: pd.DataFrame) -> Dict[str, float]:
    std_by_mode: Dict[str, float] = {}
    for mode in MODE_ORDER:
        mode_df = hourly_mae_df[hourly_mae_df["mode"] == mode]
        values = mode_df["hourly_mae"].dropna().to_numpy(dtype=float)
        std_by_mode[mode] = float(np.std(values, ddof=0)) if values.size else float("nan")
    return std_by_mode


def split_labels(splits: SplitMasks) -> np.ndarray:
    labels = np.full(splits.train.numel(), "test", dtype=object)
    labels[splits.train.cpu().numpy()] = "train"
    labels[splits.val.cpu().numpy()] = "val"
    labels[splits.test.cpu().numpy()] = "test"
    return labels


def collect_node_predictions(
    target_hour: pd.Timestamp,
    mode: str,
    data: Data,
    splits: SplitMasks,
    sample_weight: torch.Tensor,
    result: GNNResult,
) -> pd.DataFrame:
    split_label = split_labels(splits)
    learned_weights = (
        result.confidence_weights
        if result.confidence_weights is not None
        else np.full(len(CONFIDENCE_COMPONENT_NAMES), np.nan, dtype=np.float32)
    )
    return pd.DataFrame(
        {
            "target_hour": target_hour,
            "mode": mode,
            "Model": ABLATION_MODES[mode]["model"],
            "PULocationID": data.location_id.cpu().numpy().astype(int),
            "split": split_label,
            "is_test": split_label == "test",
            "base_pred": data.base_pred.cpu().numpy(),
            "true_value": data.true_value.cpu().numpy(),
            "residual_target": data.y.cpu().numpy(),
            "residual_pred": result.residual_pred,
            "refined_pred": result.refined_pred,
            "sample_weight": sample_weight.cpu().numpy(),
            "prior_score": data.prior_score.cpu().numpy(),
            "stability_score": data.stability_score.cpu().numpy(),
            "history_consistency_score": data.history_consistency_score.cpu().numpy(),
            "full_confidence": data.full_confidence.cpu().numpy(),
            "w_prior": float(learned_weights[0]),
            "w_stability": float(learned_weights[1]),
            "w_history_consistency": float(learned_weights[2]),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        }
    )


def run_single_ablation(
    target_hour: pd.Timestamp,
    mode: str,
    train_data: Optional[Data],
    train_splits: Optional[SplitMasks],
    inference_data: Data,
    inference_splits: SplitMasks,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Train on historical snapshots and evaluate one target-hour inference graph."""

    result = train_residual_gnn_with_weighted_loss(
        train_data=train_data,
        train_splits=train_splits,
        inference_data=inference_data,
        mode=mode,
        device=device,
        cfg=cfg,
        seed=seed,
    )
    if result.confidence_weights is not None:
        sample_weight = confidence_from_component_weights(
            inference_data,
            torch.tensor(result.confidence_weights, dtype=torch.float32),
        )
    else:
        sample_weight = get_sample_weights(mode, inference_data)
    prediction_df = collect_node_predictions(
        target_hour=target_hour,
        mode=mode,
        data=inference_data,
        splits=inference_splits,
        sample_weight=sample_weight,
        result=result,
    )

    test_df = prediction_df[prediction_df["is_test"]]
    metrics = evaluate_refined_predictions(
        y_true=test_df["true_value"].to_numpy(dtype=float),
        y_pred=test_df["refined_pred"].to_numpy(dtype=float),
    )
    hourly_record = {
        "target_hour": target_hour,
        "mode": mode,
        "Model": ABLATION_MODES[mode]["model"],
        "hourly_mae": metrics["MAE"],
        "hourly_rmse": metrics["RMSE"],
        "hourly_mse": metrics["MSE"],
        "n_train": (
            int(train_splits.train.sum().item()) if train_splits is not None else 0
        ),
        "n_val": int(train_splits.val.sum().item()) if train_splits is not None else 0,
        "n_test": int(inference_splits.test.sum().item()),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "w_prior": (
            float(result.confidence_weights[0])
            if result.confidence_weights is not None
            else np.nan
        ),
        "w_stability": (
            float(result.confidence_weights[1])
            if result.confidence_weights is not None
            else np.nan
        ),
        "w_history_consistency": (
            float(result.confidence_weights[2])
            if result.confidence_weights is not None
            else np.nan
        ),
    }
    return prediction_df, hourly_record


def summarize_results(
    detailed_df: pd.DataFrame,
    hourly_mae_df: pd.DataFrame,
) -> pd.DataFrame:
    std_by_mode = compute_hourly_mae_std(hourly_mae_df)
    rows: List[Dict[str, object]] = []

    for mode in MODE_ORDER:
        mode_df = detailed_df[(detailed_df["mode"] == mode) & detailed_df["is_test"]]
        metrics = evaluate_refined_predictions(
            y_true=mode_df["true_value"].to_numpy(dtype=float),
            y_pred=mode_df["refined_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "Model": ABLATION_MODES[mode]["model"],
                "mode": mode,
                "Description": ABLATION_MODES[mode]["description"],
                "MAE ↓": metrics["MAE"],
                "RMSE ↓": metrics["RMSE"],
                "MSE ↓": metrics["MSE"],
                "Std of hourly MAE ↓": std_by_mode[mode],
            }
        )
    return pd.DataFrame(rows)


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_random_seed(args.seed)
    if args.clean_checkpoints:
        clean_checkpoint_dir(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)
    rolling_od = uses_rolling_od_graph(args)
    if rolling_od:
        print(
            "Rolling OD graph window: "
            f"{args.od_lookback_days} days before each target hour"
        )

    df, lookup_df, graph_template = load_required_data(
        data_path=args.data,
        lookup_path=args.lookup,
        edge_csv=args.edge_csv,
        excluded_zones=args.excluded_zones,
        include_dropoff=rolling_od,
    )
    window_starts = resolve_window_starts(args)
    validate_windows(df, window_starts, args.rolling_steps)
    print(
        "Window starts:",
        ", ".join(str(start) for start in window_starts),
    )
    rolling_od_cache: Dict[pd.Timestamp, GraphContext] = {}
    od_zone_names: List[str] = []
    od_location_to_zone: Dict[int, str] = {}
    if rolling_od:
        od_zone_names = load_od_zone_names(args)
        od_location_to_zone = load_od_location_lookup(args)
        first_target_hour = window_starts[0]
        rolling_od_cache[first_target_hour] = build_rolling_od_graph_context(
            args=args,
            df=df,
            lookup_df=lookup_df,
            zone_names=od_zone_names,
            location_to_zone=od_location_to_zone,
            target_hour=first_target_hour,
            build_graph_context_fn=build_graph_context,
        )
        graph_template = rolling_od_cache[first_target_hour]
    zones = select_zones(df, graph_template, args.max_zones)
    print("Graph-compatible zones selected:", len(zones))

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
            f"\n===== Experiment 3 window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            print(
                f"\n///// Experiment 3 target hour: {target_hour} "
                f"window {window_idx} step {step} /////"
            )

            set_random_seed(args.seed + global_step)
            if rolling_od:
                if target_hour not in rolling_od_cache:
                    print(f"[{target_hour}] start rolling OD graph", flush=True)
                    rolling_od_cache[target_hour] = build_rolling_od_graph_context(
                        args=args,
                        df=df,
                        lookup_df=lookup_df,
                        zone_names=od_zone_names,
                        location_to_zone=od_location_to_zone,
                        target_hour=target_hour,
                        build_graph_context_fn=build_graph_context,
                    )
                    print(f"[{target_hour}] done rolling OD graph", flush=True)
                graph_template = rolling_od_cache[target_hour]
            baseline_df = run_multiscale_temporal_baseline(
                df=df,
                manager=manager,
                target_hour=target_hour,
                zones=zones,
                zone_hourly_counts=zone_hourly_counts,
            )
            if args.graph_type == "residual":
                graph, residual_summary = build_dynamic_residual_graph_context(
                    args=args,
                    df=df,
                    manager=manager,
                    lookup_df=lookup_df,
                    target_hour=target_hour,
                    zones=zones,
                    zone_hourly_counts=zone_hourly_counts,
                    graph_context=graph_template,
                    prediction_cache=residual_prediction_cache,
                    build_graph_context_fn=build_graph_context,
                )
                residual_summary.update(
                    {
                        "window_id": window_idx,
                        "window_start": window_start,
                        "window_step": step,
                    }
                )
                residual_summary_records.append(residual_summary)
                summary_log_path = residual_graph_summary_path(args)
                summary_log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(residual_summary_records).to_csv(summary_log_path, index=False)
            else:
                graph = graph_template
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
            inference_data = build_gnn_features(step_df=step_df, graph=graph)
            split_sets = make_location_split_sets(
                location_ids=inference_data.location_id.cpu().numpy().astype(int),
                seed=args.seed + global_step,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            inference_splits = masks_from_location_splits(
                data=inference_data,
                split_sets=split_sets,
                require_train_val=False,
                require_test=True,
            )

            history_frames = filter_history_frames_before(historical_step_frames, target_hour)
            train_data = build_historical_gnn_training_data(
                history_frames=history_frames,
                graph=graph,
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
                    print(f"[{target_hour}] historical GNN training skipped: {exc}")
                    train_data = None

            hour_messages: List[str] = []
            for mode in MODE_ORDER:
                prediction_df, hourly_record = run_single_ablation(
                    target_hour=target_hour,
                    mode=mode,
                    train_data=train_data,
                    train_splits=train_splits,
                    inference_data=inference_data,
                    inference_splits=inference_splits,
                    device=device,
                    cfg=gnn_cfg,
                    seed=args.seed + global_step,
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
                hour_messages.append(f"{mode} MAE={hourly_record['hourly_mae']:.4f}")

            print(f"[{target_hour}] " + ", ".join(hour_messages))
            historical_step_frames.append(step_df.copy())

    detailed_df = pd.concat(detailed_frames, ignore_index=True)
    hourly_mae_df = pd.DataFrame(hourly_records)
    summary_df = summarize_results(detailed_df=detailed_df, hourly_mae_df=hourly_mae_df)
    return summary_df, detailed_df, hourly_mae_df


def save_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    hourly_mae_df: pd.DataFrame,
    results_dir: Path,
    make_plot: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "experiment_3_confidence_weighted_gnn_ablation_summary.csv"
    detailed_path = results_dir / "experiment_3_confidence_weighted_gnn_ablation_detailed.csv"
    hourly_path = results_dir / "experiment_3_confidence_weighted_gnn_ablation_hourly_mae.csv"
    plot_path = results_dir / "experiment_3_confidence_weighted_gnn_ablation_metrics.png"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    hourly_mae_df.to_csv(hourly_path, index=False)

    plot_written = False
    if make_plot:
        try:
            plot_metrics(summary_df=summary_df, hourly_mae_df=hourly_mae_df, plot_path=plot_path)
            plot_written = True
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed predictions to {detailed_path}")
    print(f"Saved hourly MAE to {hourly_path}")
    if plot_written:
        print(f"Saved plot to {plot_path}")


def plot_metrics(summary_df: pd.DataFrame, hourly_mae_df: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    models = summary_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(x - width, summary_df["MAE ↓"], width, label="MAE")
    axes[0].bar(x, summary_df["RMSE ↓"], width, label="RMSE")
    axes[0].bar(x + width, summary_df["MSE ↓"], width, label="MSE")
    axes[0].set_title("Overall Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].bar(models, summary_df["Std of hourly MAE ↓"])
    axes[1].set_title("Std of Hourly MAE")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for mode in MODE_ORDER:
        mode_df = hourly_mae_df[hourly_mae_df["mode"] == mode].sort_values("target_hour")
        axes[2].plot(
            pd.to_datetime(mode_df["target_hour"]),
            mode_df["hourly_mae"],
            marker="o",
            linewidth=1.5,
            label=ABLATION_MODES[mode]["model"],
        )
    axes[2].set_title("Hourly MAE Over Rolling Hours")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8)

    fig.suptitle("Experiment 3: Confidence-Weighted GNN Ablation")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    display_df = summary_df[
        ["Model", "MAE ↓", "RMSE ↓", "MSE ↓", "Std of hourly MAE ↓"]
    ].copy()
    for column in ["MAE ↓", "RMSE ↓", "MSE ↓", "Std of hourly MAE ↓"]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "nan"
        )
    print("\nFinal Summary Table")
    print(display_df.to_string(index=False))


def clean_checkpoint_dir(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"[debug] deleted checkpoint directory: {checkpoint_dir}")


def main() -> None:
    args = parse_args()
    summary_df, detailed_df, hourly_mae_df = run_experiment(args)
    save_results(
        summary_df=summary_df,
        detailed_df=detailed_df,
        hourly_mae_df=hourly_mae_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
