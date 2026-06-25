"""Experiment 2: effect of GraphSAGE residual graph refinement.

This script keeps the temporal baseline from persistent_tcn.py and
compares residual GNN feature variants on top:

T4: TCN temporal only
G0: TCN + GraphSAGE residual correction with base prediction only
G1: TCN + GraphSAGE residual correction with confidence feature
G2: TCN + GraphSAGE residual correction with historical mean features
G3: TCN + GraphSAGE residual correction with confidence + history

All GNN variants use learned-softmax confidence weighting over prior, stability, and
history-consistency components in the residual loss.

For each rolling target hour, GNN variants train only on residual snapshots from hours
strictly before the target. Target-hour labels are used only for post-inference
test metrics.
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
DEFAULT_GRAPH_TYPE = "residual"
DEFAULT_RESIDUAL_WINDOW_HOURS = 24 * 30

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

MODEL_SPECS = {
    "T4": {
        "model": "TCN Temporal",
        "description": "Only base prediction",
    },
    "G0": {
        "model": "TCN + GNN (Base Feature)",
        "description": "GNN uses base prediction only as node feature",
    },
    "G1": {
        "model": "TCN + GNN + Confidence Feature",
        "description": "GNN uses base prediction + confidence as node features",
    },
    "G2": {
        "model": "TCN + GNN + History Features",
        "description": "GNN uses base prediction + historical mean features",
    },
    "G3": {
        "model": "TCN + GNN + Confidence + History",
        "description": "GNN uses base prediction + confidence + historical mean features",
    },
}
MODEL_ORDER = ("T4", "G0", "G1", "G2", "G3")
GNN_MODEL_ORDER = ("G0", "G1", "G2", "G3")
FEATURE_COLUMNS_BY_MODEL = {
    "G0": ["base_pred"],
    "G1": ["base_pred", "confidence"],
    "G2": ["base_pred", *HISTORY_FEATURES],
    "G3": ["base_pred", "confidence", *HISTORY_FEATURES],
}


@dataclass(frozen=True)
class GraphContext:
    """Static graph metadata shared by all rolling hours."""

    edge_index: torch.Tensor
    zone_names: List[str]
    zone_idx_map: Dict[str, int]
    location_to_zone: Dict[int, str]
    zone_to_location: Dict[str, int]


@dataclass(frozen=True)
class SplitMasks:
    """Node-level train/validation/test split for one graph snapshot."""

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
    """Hyperparameters for residual GraphSAGE training."""

    hidden_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 0.01
    epochs: int = 300
    patience: int = 40


@dataclass
class GNNResult:
    """Predictions and diagnostics returned by residual GraphSAGE."""

    residual_pred: np.ndarray
    refined_pred: np.ndarray
    best_epoch: int
    best_val_loss: float
    confidence_weights: Optional[np.ndarray] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: effect of GraphSAGE residual refinement."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_WEIGHT_MATRIX)
    parser.add_argument(
        "--graph-type",
        choices=["residual", "od"],
        default=DEFAULT_GRAPH_TYPE,
        help="Graph used by residual GraphSAGE. Default builds dynamic residual graphs.",
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
        help="Optional smoke-test limit. Default evaluates all non-excluded zones.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing experiment_2_graph_refinement_metrics.png.",
    )
    args = parser.parse_args()
    if args.residual_window_hours <= 0:
        parser.error("--residual-window-hours must be > 0.")
    if args.residual_top_k < 0:
        parser.error("--residual-top-k must be >= 0.")
    if args.residual_min_corr < 0:
        parser.error("--residual-min-corr must be >= 0.")
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
) -> Tuple[pd.DataFrame, pd.DataFrame, GraphContext]:
    """Load taxi trips, zone lookup, and the OD-flow graph."""

    df = pd.read_parquet(data_path, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(excluded_zones)].copy()

    lookup_df = pd.read_csv(lookup_path).drop_duplicates(subset="LocationID")
    graph_context = build_graph_context(edge_csv, lookup_df)

    print("Earliest timestamp:", df["datetime"].min())
    print("Latest timestamp:", df["datetime"].max())
    print("Total hours:", df["datetime"].nunique())
    print("Total non-excluded zones:", df["PULocationID"].nunique())
    return df, lookup_df, graph_context


def build_graph_context(edge_csv: Path, lookup_df: pd.DataFrame) -> GraphContext:
    """Build graph tensors and mappings using the same OD matrix as demo_rolling_GNN."""

    df_adj = pd.read_csv(edge_csv, index_col=0)
    df_adj.index = [str(idx).lstrip("\ufeff") for idx in df_adj.index]
    df_adj.columns = [str(col).lstrip("\ufeff") for col in df_adj.columns]
    df_adj = df_adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj_matrix)

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
        zone_names=zone_names,
        zone_idx_map=zone_idx_map,
        location_to_zone=location_to_zone,
        zone_to_location=zone_to_location,
    )


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

    Missing hours are treated as zero by dividing by the full requested window.
    This matches sparse taxi count semantics and avoids using future values.
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


def compute_prior_scores(history_df: pd.DataFrame) -> Dict[int, float]:
    """Higher scores for zones with richer past history."""

    counts = history_df.groupby("PULocationID").size().astype(float)
    return normalize_prior_counts(counts)


def compute_prior_scores_from_zone_hourly_counts(
    zone_hourly_counts: pd.Series,
    target_hour: pd.Timestamp,
) -> Dict[int, float]:
    """Higher scores for zones with richer past history, using precomputed counts."""

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
    return {int(idx): float(val) for idx, val in scaled.items()}


def compute_stability_scores(step_df: pd.DataFrame) -> Dict[int, float]:
    """Use MC variance as an inverse uncertainty score."""

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
            scores[zone_id] = float(np.clip(raw, 0.05, 1.0))
        else:
            scores[zone_id] = 0.2
    return scores


def compute_history_consistency_scores(step_df: pd.DataFrame) -> Dict[int, float]:
    """Higher scores when 24h/168h/720h historical means agree."""
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
        scores[zone_id] = float(np.clip(raw, 0.05, 1.0))
    return scores


def assign_confidence_scores(
    step_df: pd.DataFrame,
    prior_scores: Dict[int, float],
) -> Dict[int, float]:
    stability_scores = compute_stability_scores(step_df)
    history_scores = compute_history_consistency_scores(step_df)

    zone_confidence: Dict[int, float] = {}
    for row in step_df.itertuples():
        zone_id = int(row.PULocationID)
        if not np.isfinite(row.base_pred):
            zone_confidence[zone_id] = 0.0
            continue

        combined = combine_confidence_components(
            prior=prior_scores.get(zone_id, 0.4),
            stability=stability_scores.get(zone_id, 0.5),
            history_consistency=history_scores.get(zone_id, 0.6),
            weights=DEFAULT_CONFIDENCE_WEIGHTS,
        )
        zone_confidence[zone_id] = combined

    step_df["prior_score"] = step_df["PULocationID"].map(
        lambda zone_id: prior_scores.get(int(zone_id), 0.4)
    )
    step_df["stability_score"] = step_df["PULocationID"].map(
        lambda zone_id: stability_scores.get(int(zone_id), 0.5)
    )
    step_df["history_consistency_score"] = step_df["PULocationID"].map(
        lambda zone_id: history_scores.get(int(zone_id), 0.6)
    )
    step_df["confidence"] = step_df["PULocationID"].map(
        lambda zone_id: zone_confidence.get(int(zone_id), 0.2)
    )
    return zone_confidence


def run_multiscale_temporal_baseline(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    target_hour: pd.Timestamp,
    zones: Sequence[int],
    zone_hourly_counts: pd.Series,
    prior_scores: Dict[int, float],
) -> pd.DataFrame:
    """Generate base_pred using persistent_tcn.py only."""

    y_true_dict = get_true_counts(df, target_hour)
    records: List[Dict[str, object]] = []

    for zone_id in zones:
        try:
            zone_int = int(zone_id)
            context_end = target_hour - manager._forecast_delta
            if not manager.has_checkpoint(zone_int):
                manager.train_once(df, zone_int, context_end)
            point, std, _ = manager.predict_with_uncertainty(df, int(zone_id), target_hour)
            true_value = float(y_true_dict.get(zone_id, 0.0))
            history_means = compute_history_means(zone_hourly_counts, int(zone_id), target_hour)
            records.append(
                {
                    "target_hour": target_hour,
                    "PULocationID": int(zone_id),
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

    step_df = pd.DataFrame(records)
    assign_confidence_scores(step_df, prior_scores)
    return step_df


def build_gnn_features_g0(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(step_df, graph, feature_columns=["base_pred"])


def build_gnn_features_g1(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(step_df, graph, feature_columns=["base_pred", "confidence"])


def build_gnn_features_g2(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(
        step_df,
        graph,
        feature_columns=["base_pred", *HISTORY_FEATURES],
    )


def build_gnn_features_g3(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(
        step_df,
        graph,
        feature_columns=["base_pred", "confidence", *HISTORY_FEATURES],
    )


def build_gnn_features_for_model(
    model_id: str,
    step_df: pd.DataFrame,
    graph: GraphContext,
) -> Data:
    try:
        feature_columns = FEATURE_COLUMNS_BY_MODEL[model_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported GNN model id: {model_id}") from exc
    return build_gnn_data(step_df=step_df, graph=graph, feature_columns=feature_columns)


def build_gnn_data(
    step_df: pd.DataFrame,
    graph: GraphContext,
    feature_columns: Sequence[str],
) -> Data:
    """Map zone predictions into graph order and build residual targets.

    The supervised target is residual_target = true_value - base_pred. Historical
    mean NaNs are filled with 0 before log1p; this means missing history is treated
    as zero demand, consistent with sparse hourly count construction.
    """

    node_count = len(graph.zone_names)
    node_pred = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_label = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_confidence = torch.full((node_count,), 0.2, dtype=torch.float32)
    prior_score = torch.full((node_count,), 0.4, dtype=torch.float32)
    stability_score = torch.full((node_count,), 0.5, dtype=torch.float32)
    history_score = torch.full((node_count,), 0.6, dtype=torch.float32)
    history_tensor = torch.full((node_count, len(HISTORY_FEATURES)), 0.0, dtype=torch.float32)

    for row in step_df.itertuples():
        location_id = int(row.PULocationID)
        zone_name = graph.location_to_zone.get(location_id)
        if zone_name is None:
            continue
        node_idx = graph.zone_idx_map.get(zone_name)
        if node_idx is None:
            continue

        node_pred[node_idx] = float(row.base_pred)
        node_label[node_idx] = float(row.true_value)
        node_confidence[node_idx] = float(np.clip(row.confidence, 0.0, 1.0))
        prior_score[node_idx] = float(np.clip(row.prior_score, 0.05, 1.0))
        stability_score[node_idx] = float(np.clip(row.stability_score, 0.05, 1.0))
        history_score[node_idx] = float(np.clip(row.history_consistency_score, 0.05, 1.0))
        for feature_idx, feature_name in enumerate(HISTORY_FEATURES):
            value = getattr(row, feature_name, 0.0)
            history_tensor[node_idx, feature_idx] = 0.0 if pd.isna(value) else float(value)

    valid_indices = torch.where(~torch.isnan(node_pred) & ~torch.isnan(node_label))[0]
    if valid_indices.numel() < 3:
        raise ValueError("Not enough valid graph nodes to train/evaluate residual GNN.")

    edge_index = remap_edges_to_valid_nodes(graph.edge_index, valid_indices)

    node_pred = node_pred[valid_indices]
    node_label = node_label[valid_indices]
    node_confidence = node_confidence[valid_indices]
    prior_score = prior_score[valid_indices]
    stability_score = stability_score[valid_indices]
    history_score = history_score[valid_indices]
    history_tensor = history_tensor[valid_indices]

    feature_tensors = []
    for column in feature_columns:
        if column == "base_pred":
            feature_tensors.append(node_pred.unsqueeze(1))
        elif column == "confidence":
            feature_tensors.append(node_confidence.unsqueeze(1))
        elif column in HISTORY_FEATURES:
            idx = HISTORY_FEATURES.index(column)
            feature_tensors.append(torch.log1p(history_tensor[:, idx]).unsqueeze(1))
        else:
            raise ValueError(f"Unsupported GNN feature column: {column}")

    x_feat = torch.cat(feature_tensors, dim=1)
    residual_target = node_label - node_pred
    valid_old_indices = [int(idx) for idx in valid_indices.cpu().tolist()]
    zone_names: List[str] = []
    location_ids: List[int] = []
    for old_idx in valid_old_indices:
        zone_name = str(graph.zone_names[old_idx])
        zone_names.append(zone_name)
        location_ids.append(int(graph.zone_to_location.get(zone_name, -1)))

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        y=residual_target,
        base_pred=node_pred,
        true_value=node_label,
        confidence=node_confidence,
        prior_score=prior_score,
        stability_score=stability_score,
        history_consistency_score=history_score,
    )
    data.zone_names = tuple(zone_names)
    data.location_ids = tuple(location_ids)
    return data


def build_historical_gnn_training_data(
    history_frames: Sequence[pd.DataFrame],
    graph: GraphContext,
    feature_columns: Sequence[str],
) -> Optional[Data]:
    """Concatenate historical residual snapshots into one train graph."""

    snapshots: List[Data] = []
    for frame in history_frames:
        if frame.empty:
            continue
        try:
            snapshots.append(
                build_gnn_data(
                    step_df=frame,
                    graph=graph,
                    feature_columns=feature_columns,
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
    confidence_parts: List[torch.Tensor] = []
    prior_parts: List[torch.Tensor] = []
    stability_parts: List[torch.Tensor] = []
    history_score_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    zone_names: List[str] = []
    location_ids: List[int] = []

    offset = 0
    for snapshot in snapshots:
        x_parts.append(snapshot.x)
        y_parts.append(snapshot.y)
        base_parts.append(snapshot.base_pred)
        true_parts.append(snapshot.true_value)
        confidence_parts.append(snapshot.confidence)
        prior_parts.append(snapshot.prior_score)
        stability_parts.append(snapshot.stability_score)
        history_score_parts.append(snapshot.history_consistency_score)
        zone_names.extend(str(name) for name in snapshot.zone_names)
        location_ids.extend(int(loc_id) for loc_id in snapshot.location_ids)

        if snapshot.edge_index.numel() > 0:
            edge_parts.append(snapshot.edge_index + offset)
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
        confidence=torch.cat(confidence_parts, dim=0),
        prior_score=torch.cat(prior_parts, dim=0),
        stability_score=torch.cat(stability_parts, dim=0),
        history_consistency_score=torch.cat(history_score_parts, dim=0),
    )
    data.zone_names = tuple(zone_names)
    data.location_ids = tuple(location_ids)
    return data


def filter_history_frames_before(
    history_frames: Sequence[pd.DataFrame],
    target_hour: pd.Timestamp,
) -> List[pd.DataFrame]:
    """Return cached residual snapshots whose label hour is before target_hour."""

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


def remap_edges_to_valid_nodes(
    edge_index: torch.Tensor,
    valid_indices: torch.Tensor,
) -> torch.Tensor:
    valid_old = [int(idx) for idx in valid_indices.tolist()]
    valid_set = set(valid_old)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_old)}

    remapped_edges: List[Tuple[int, int]] = []
    for src, dst in edge_index.t().tolist():
        src_i, dst_i = int(src), int(dst)
        if src_i in valid_set and dst_i in valid_set:
            remapped_edges.append((old_to_new[src_i], old_to_new[dst_i]))

    if not remapped_edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()


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
    location_ids = [int(loc_id) for loc_id in data.location_ids]
    train = torch.tensor([loc_id in split_sets.train for loc_id in location_ids])
    val = torch.tensor([loc_id in split_sets.val for loc_id in location_ids])
    test = torch.tensor([loc_id in split_sets.test for loc_id in location_ids])

    if require_train_val and (int(train.sum()) == 0 or int(val.sum()) == 0):
        raise ValueError("Graph data does not contain non-empty train/val splits.")
    if require_test and int(test.sum()) == 0:
        raise ValueError("Graph data does not contain a non-empty test split.")
    return SplitMasks(train=train, val=val, test=test)


def train_residual_gnn(
    train_data: Optional[Data],
    train_splits: Optional[SplitMasks],
    inference_data: Data,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> GNNResult:
    """Train GraphSAGE to predict residual_target, then refine base_pred.

    The model never predicts the true count directly. Its output is:
        refined_pred = base_pred + residual_pred
    """

    set_random_seed(seed)

    if (
        train_data is None
        or train_splits is None
        or int(train_splits.train.sum()) == 0
        or int(train_splits.val.sum()) == 0
    ):
        base_pred = inference_data.base_pred.detach().cpu().numpy()
        return GNNResult(
            residual_pred=np.zeros_like(base_pred, dtype=np.float32),
            refined_pred=base_pred.astype(np.float32),
            best_epoch=0,
            best_val_loss=float("nan"),
            confidence_weights=np.array(DEFAULT_CONFIDENCE_WEIGHTS, dtype=np.float32),
        )

    model = MultiScaleGraphSAGE(
        in_dim=int(train_data.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    train_data_device = train_data.to(device)
    train_mask = train_splits.train.to(device)
    val_mask = train_splits.val.to(device)

    logits = nn.Parameter(
        torch.log(
            torch.tensor(DEFAULT_CONFIDENCE_WEIGHTS, dtype=torch.float32)
        ).to(device)
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [logits],
        lr=cfg.learning_rate,
    )
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
        confidence_weights = torch.softmax(logits, dim=0)
        sample_weights = confidence_from_component_weights(
            train_data_device,
            confidence_weights,
        )[train_mask]
        sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)
        loss = (loss_per_node * sample_weights).mean()

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
            best_logits = logits.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_logits is not None:
        logits.data.copy_(best_logits)

    learned_weights = torch.softmax(logits.detach(), dim=0).cpu().numpy()

    model.eval()
    inference_data_device = inference_data.to(device)
    with torch.no_grad():
        residual_pred, refined_pred = model(inference_data_device)

    return GNNResult(
        residual_pred=residual_pred.detach().cpu().numpy(),
        refined_pred=refined_pred.detach().cpu().numpy(),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        confidence_weights=learned_weights,
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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


def metrics_record(
    target_hour: pd.Timestamp,
    model_id: str,
    metrics: Dict[str, float],
    splits: SplitMasks,
    node_count: int,
    window_id: Optional[int] = None,
    window_start: Optional[pd.Timestamp] = None,
    window_step: Optional[int] = None,
    train_splits: Optional[SplitMasks] = None,
    best_epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    confidence_weights: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    weights = (
        confidence_weights
        if confidence_weights is not None
        else np.full(3, np.nan, dtype=np.float32)
    )
    return {
        "window_id": window_id,
        "window_start": window_start,
        "window_step": window_step,
        "target_hour": target_hour,
        "model_id": model_id,
        "Model": MODEL_SPECS[model_id]["model"],
        "Description": MODEL_SPECS[model_id]["description"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MSE": metrics["MSE"],
        "n_nodes": node_count,
        "n_train": int(
            (train_splits.train if train_splits is not None else splits.train).sum().item()
        ),
        "n_val": int(
            (train_splits.val if train_splits is not None else splits.val).sum().item()
        ),
        "n_test": int(splits.test.sum().item()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "w_prior": float(weights[0]),
        "w_stability": float(weights[1]),
        "w_history_consistency": float(weights[2]),
    }


def collect_errors(
    error_store: Dict[str, List[np.ndarray]],
    model_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    error_store[model_id].append(y_pred[mask] - y_true[mask])


def summarize_results(error_store: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    temporal_mae = np.nan
    temporal_rmse = np.nan
    temporal_mse = np.nan

    metric_by_model: Dict[str, Dict[str, float]] = {}
    for model_id in MODEL_ORDER:
        errors = np.concatenate(error_store[model_id]) if error_store[model_id] else np.array([])
        if errors.size == 0:
            metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
        else:
            mse = float(np.mean(errors**2))
            metrics = {
                "MAE": float(np.mean(np.abs(errors))),
                "RMSE": float(np.sqrt(mse)),
                "MSE": mse,
            }
        metric_by_model[model_id] = metrics

    temporal_mae = metric_by_model["T4"]["MAE"]
    temporal_rmse = metric_by_model["T4"]["RMSE"]
    temporal_mse = metric_by_model["T4"]["MSE"]

    for model_id in MODEL_ORDER:
        metrics = metric_by_model[model_id]
        if model_id == "T4" or not np.isfinite(temporal_mae) or temporal_mae == 0.0:
            improvement = "-"
            improvement_mae = np.nan
            improvement_rmse = np.nan
            improvement_mse = np.nan
        else:
            improvement_mae = (temporal_mae - metrics["MAE"]) / temporal_mae * 100.0
            improvement_rmse = (
                (temporal_rmse - metrics["RMSE"]) / temporal_rmse * 100.0
                if np.isfinite(temporal_rmse) and temporal_rmse != 0.0
                else np.nan
            )
            improvement_mse = (
                (temporal_mse - metrics["MSE"]) / temporal_mse * 100.0
                if np.isfinite(temporal_mse) and temporal_mse != 0.0
                else np.nan
            )
            improvement = f"{improvement_mae:.2f}%"

        rows.append(
            {
                "Model ID": model_id,
                "Model": MODEL_SPECS[model_id]["model"],
                "Description": MODEL_SPECS[model_id]["description"],
                "MAE ↓": metrics["MAE"],
                "RMSE ↓": metrics["RMSE"],
                "MSE ↓": metrics["MSE"],
                "Improvement over Temporal": improvement,
                "MAE Improvement (%)": improvement_mae,
                "RMSE Improvement (%)": improvement_rmse,
                "MSE Improvement (%)": improvement_mse,
            }
        )
    return pd.DataFrame(rows)


def save_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    results_dir: Path,
    make_plot: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "experiment_2_graph_refinement_summary.csv"
    detailed_path = results_dir / "experiment_2_graph_refinement_detailed.csv"
    plot_path = results_dir / "experiment_2_graph_refinement_metrics.png"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)

    plot_written = False
    if make_plot:
        try:
            plot_metrics(summary_df, plot_path)
            plot_written = True
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed rolling metrics to {detailed_path}")
    if plot_written:
        print(f"Saved plot to {plot_path}")


def plot_metrics(summary_df: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = summary_df["Model"].tolist()
    mae_values = summary_df["MAE ↓"].to_numpy(dtype=float)
    rmse_values = summary_df["RMSE ↓"].to_numpy(dtype=float)
    mse_values = summary_df["MSE ↓"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(11, 5))
    plt.bar(x - width, mae_values, width, label="MAE")
    plt.bar(x, rmse_values, width, label="RMSE")
    plt.bar(x + width, mse_values, width, label="MSE")
    plt.xticks(x, labels, rotation=10, ha="right")
    plt.ylabel("Error")
    plt.title("Experiment 2: Effect of Graph Refinement")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def print_summary_table(summary_df: pd.DataFrame) -> None:
    display_df = summary_df[
        ["Model", "MAE ↓", "RMSE ↓", "MSE ↓", "Improvement over Temporal"]
    ].copy()
    for column in ["MAE ↓", "RMSE ↓", "MSE ↓"]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "nan"
        )
    print("\nFinal Summary Table")
    print(display_df.to_string(index=False))


def clean_checkpoint_dir(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"[debug] deleted checkpoint directory: {checkpoint_dir}")


def select_zones(df: pd.DataFrame, max_zones: Optional[int]) -> List[int]:
    zones = sorted(int(zone_id) for zone_id in df["PULocationID"].unique())
    if max_zones is not None:
        zones = zones[: max(1, max_zones)]
        print(f"Using first {len(zones)} zones for smoke-test run.")
    return zones


def validate_matching_nodes(reference: Data, candidates: Dict[str, Data]) -> None:
    reference_ids = list(reference.location_ids)
    for model_id, data in candidates.items():
        if list(data.location_ids) != reference_ids:
            raise ValueError(f"{model_id} graph snapshot does not match node order.")


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_random_seed(args.seed)
    if args.clean_checkpoints:
        clean_checkpoint_dir(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)

    df, lookup_df, graph_template = load_required_data(
        data_path=args.data,
        lookup_path=args.lookup,
        edge_csv=args.edge_csv,
        excluded_zones=args.excluded_zones,
    )
    window_starts = resolve_window_starts(args)
    validate_windows(df, window_starts, args.rolling_steps)
    print(
        "Window starts:",
        ", ".join(str(start) for start in window_starts),
    )
    zones = select_zones(df, args.max_zones)
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

    detailed_records: List[Dict[str, object]] = []
    error_store: Dict[str, List[np.ndarray]] = {model_id: [] for model_id in MODEL_ORDER}
    residual_prediction_cache: Dict[Tuple[int, pd.Timestamp], float] = {}
    residual_summary_records: List[Dict[str, object]] = []

    for window_idx, window_start in enumerate(window_starts, start=1):
        # Keep GNN residual-training snapshots local to this evaluation window.
        historical_step_frames: List[pd.DataFrame] = []
        print(
            f"\n===== Experiment 2 window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            print(
                f"\n///// Experiment 2 target hour: {target_hour} "
                f"window {window_idx} step {step} /////"
            )

            set_random_seed(args.seed + global_step)
            prior_scores = compute_prior_scores_from_zone_hourly_counts(
                zone_hourly_counts=zone_hourly_counts,
                target_hour=target_hour,
            )
            step_df = run_multiscale_temporal_baseline(
                df=df,
                manager=manager,
                target_hour=target_hour,
                zones=zones,
                zone_hourly_counts=zone_hourly_counts,
                prior_scores=prior_scores,
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
            step_df["window_id"] = window_idx
            step_df["window_start"] = window_start
            step_df["window_step"] = step

            inference_data_by_model = {
                model_id: build_gnn_features_for_model(model_id, step_df, graph)
                for model_id in GNN_MODEL_ORDER
            }
            reference_data = inference_data_by_model[GNN_MODEL_ORDER[0]]
            validate_matching_nodes(reference_data, inference_data_by_model)

            split_sets = make_location_split_sets(
                location_ids=[int(loc_id) for loc_id in reference_data.location_ids],
                seed=args.seed + global_step,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            splits = masks_from_location_splits(
                data=reference_data,
                split_sets=split_sets,
                require_train_val=False,
                require_test=True,
            )
            history_frames = filter_history_frames_before(historical_step_frames, target_hour)
            train_data_by_model: Dict[str, Optional[Data]] = {}
            train_splits_by_model: Dict[str, Optional[SplitMasks]] = {}
            for model_id in GNN_MODEL_ORDER:
                train_data = build_historical_gnn_training_data(
                    history_frames=history_frames,
                    graph=graph,
                    feature_columns=FEATURE_COLUMNS_BY_MODEL[model_id],
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
                        print(f"[{target_hour}] {model_id} historical GNN training skipped: {exc}")
                        train_data = None
                train_data_by_model[model_id] = train_data
                train_splits_by_model[model_id] = train_splits

            test_mask_np = splits.test.cpu().numpy().astype(bool)
            y_true = reference_data.true_value.cpu().numpy()[test_mask_np]
            base_pred = reference_data.base_pred.cpu().numpy()[test_mask_np]

            metrics_t4 = evaluate_predictions(y_true=y_true, y_pred=base_pred)
            collect_errors(error_store, "T4", y_true, base_pred)
            detailed_records.append(
                metrics_record(
                    target_hour=target_hour,
                    model_id="T4",
                    metrics=metrics_t4,
                    splits=splits,
                    node_count=reference_data.num_nodes,
                    window_id=window_idx,
                    window_start=window_start,
                    window_step=step,
                )
            )

            hour_messages = [f"T4 MAE={metrics_t4['MAE']:.4f}"]
            for model_id in GNN_MODEL_ORDER:
                result = train_residual_gnn(
                    train_data=train_data_by_model[model_id],
                    train_splits=train_splits_by_model[model_id],
                    inference_data=inference_data_by_model[model_id],
                    device=device,
                    cfg=gnn_cfg,
                    seed=args.seed + global_step,
                )
                refined = result.refined_pred[test_mask_np]
                metrics = evaluate_predictions(y_true=y_true, y_pred=refined)
                collect_errors(error_store, model_id, y_true, refined)
                detailed_records.append(
                    metrics_record(
                        target_hour=target_hour,
                        model_id=model_id,
                        metrics=metrics,
                        splits=splits,
                        node_count=inference_data_by_model[model_id].num_nodes,
                        window_id=window_idx,
                        window_start=window_start,
                        window_step=step,
                        train_splits=train_splits_by_model[model_id],
                        best_epoch=result.best_epoch,
                        best_val_loss=result.best_val_loss,
                        confidence_weights=result.confidence_weights,
                    )
                )

                weights = result.confidence_weights
                hour_messages.append(
                    f"{model_id} MAE={metrics['MAE']:.4f} "
                    f"(w={weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f})"
                )

            print(f"[{target_hour}] " + ", ".join(hour_messages))
            historical_step_frames.append(step_df.copy())

    detailed_df = pd.DataFrame(detailed_records)
    summary_df = summarize_results(error_store)
    return summary_df, detailed_df


def main() -> None:
    args = parse_args()
    summary_df, detailed_df = run_experiment(args)
    save_results(
        summary_df=summary_df,
        detailed_df=detailed_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    print_summary_table(summary_df)


if __name__ == "__main__":
    main()
