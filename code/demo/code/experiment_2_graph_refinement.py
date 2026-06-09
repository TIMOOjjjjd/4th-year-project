"""Experiment 2: effect of GraphSAGE residual graph refinement.

This script keeps the temporal baseline from persistent_multiscale_confi.py and
adds two residual GNN variants on top:

T4: Multi-scale temporal only
G1: Multi-scale + GraphSAGE residual correction
G2: Multi-scale + GraphSAGE residual correction + historical mean features
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

from gnn_model import MultiScaleGraphSAGE
from persistent_multiscale_confi import ManagerConfig, MultiScaleModelManager


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
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_experiment_2_multiscale"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

START_TARGET = pd.Timestamp("2021-07-05 00:00")
ROLLING_STEPS = 24
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
        "model": "Multi-scale Temporal",
        "description": "Only base prediction",
    },
    "G1": {
        "model": "Multi-scale + GNN",
        "description": "GNN learns residual correction",
    },
    "G2": {
        "model": "Multi-scale + GNN + History Features",
        "description": "GNN uses base prediction + historical mean features",
    },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: effect of GraphSAGE residual refinement."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_WEIGHT_MATRIX)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-target", type=pd.Timestamp, default=START_TARGET)
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
    return parser.parse_args()


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


def combine_confidence_components(
    prior: float,
    stability: float,
    history_consistency: float,
    weights: Dict[str, float],
) -> float:
    prior_c = np.clip(prior, 0.05, 1.0)
    stability_c = np.clip(stability, 0.05, 1.0)
    history_c = np.clip(history_consistency, 0.05, 1.0)
    log_score = (
        weights["prior"] * np.log(prior_c)
        + weights["stability"] * np.log(stability_c)
        + weights["history_consistency"] * np.log(history_c)
    )
    return float(np.clip(np.exp(log_score), 0.05, 1.0))


def assign_confidence_scores(
    step_df: pd.DataFrame,
    prior_scores: Dict[int, float],
) -> Dict[int, float]:
    stability_scores = compute_stability_scores(step_df)
    history_scores = compute_history_consistency_scores(step_df)
    weights = {"prior": 0.3, "stability": 0.4, "history_consistency": 0.3}

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
            weights=weights,
        )
        zone_confidence[zone_id] = combined

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
    """Generate base_pred using persistent_multiscale_confi.py only."""

    y_true_dict = get_true_counts(df, target_hour)
    records: List[Dict[str, object]] = []

    for zone_id in zones:
        try:
            manager.train_and_predict_if_needed(df, int(zone_id), target_hour, auto_train=True)
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


def build_gnn_features_g1(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(step_df, graph, feature_columns=["base_pred", "confidence"])


def build_gnn_features_g2(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    return build_gnn_data(
        step_df,
        graph,
        feature_columns=["base_pred", "confidence", *HISTORY_FEATURES],
    )


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
    )
    data.zone_names = tuple(zone_names)
    data.location_ids = tuple(location_ids)
    return data


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


def train_residual_gnn(
    data: Data,
    splits: SplitMasks,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> GNNResult:
    """Train GraphSAGE to predict residual_target, then refine base_pred.

    The model never predicts the true count directly. Its output is:
        refined_pred = base_pred + residual_pred
    """

    set_random_seed(seed)
    model = MultiScaleGraphSAGE(
        in_dim=int(data.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    data = data.to(device)
    train_mask = splits.train.to(device)
    val_mask = splits.val.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_func = nn.SmoothL1Loss(reduction="none")

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        residual_pred, _ = model(data)

        loss_per_node = loss_func(residual_pred[train_mask], data.y[train_mask])
        sample_weights = data.confidence[train_mask].clamp(min=0.05)
        sample_weights = sample_weights / sample_weights.mean().clamp(min=1e-6)
        loss = (loss_per_node * sample_weights).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_residual_pred, _ = model(data)
            val_loss = loss_func(val_residual_pred[val_mask], data.y[val_mask]).mean()
            val_loss_value = float(val_loss.item())

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        residual_pred, refined_pred = model(data)

    return GNNResult(
        residual_pred=residual_pred.detach().cpu().numpy(),
        refined_pred=refined_pred.detach().cpu().numpy(),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
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
    best_epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
) -> Dict[str, object]:
    return {
        "target_hour": target_hour,
        "model_id": model_id,
        "Model": MODEL_SPECS[model_id]["model"],
        "Description": MODEL_SPECS[model_id]["description"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MSE": metrics["MSE"],
        "n_nodes": node_count,
        "n_train": int(splits.train.sum().item()),
        "n_val": int(splits.val.sum().item()),
        "n_test": int(splits.test.sum().item()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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
    for model_id in ("T4", "G1", "G2"):
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

    for model_id in ("T4", "G1", "G2"):
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


def validate_matching_nodes(data_a: Data, data_b: Data) -> None:
    if list(data_a.location_ids) != list(data_b.location_ids):
        raise ValueError("G1 and G2 graph snapshots do not have matching node order.")


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_random_seed(args.seed)
    if args.clean_checkpoints:
        clean_checkpoint_dir(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)

    df, _lookup_df, graph = load_required_data(
        data_path=args.data,
        lookup_path=args.lookup,
        edge_csv=args.edge_csv,
        excluded_zones=args.excluded_zones,
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
    error_store: Dict[str, List[np.ndarray]] = {"T4": [], "G1": [], "G2": []}

    for step in range(args.rolling_steps):
        target_hour = args.start_target + pd.Timedelta(hours=step)
        print(f"\n///// Experiment 2 target hour: {target_hour} step {step} /////")

        history_df = df[df["datetime"] < target_hour]
        prior_scores = compute_prior_scores(history_df)
        step_df = run_multiscale_temporal_baseline(
            df=df,
            manager=manager,
            target_hour=target_hour,
            zones=zones,
            zone_hourly_counts=zone_hourly_counts,
            prior_scores=prior_scores,
        )

        data_g1 = build_gnn_features_g1(step_df, graph)
        data_g2 = build_gnn_features_g2(step_df, graph)
        validate_matching_nodes(data_g1, data_g2)

        splits = make_node_splits(
            node_count=data_g1.num_nodes,
            seed=args.seed + step,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        test_mask_np = splits.test.cpu().numpy().astype(bool)
        y_true = data_g1.true_value.cpu().numpy()[test_mask_np]
        base_pred = data_g1.base_pred.cpu().numpy()[test_mask_np]

        metrics_t4 = evaluate_predictions(y_true=y_true, y_pred=base_pred)
        collect_errors(error_store, "T4", y_true, base_pred)
        detailed_records.append(
            metrics_record(
                target_hour=target_hour,
                model_id="T4",
                metrics=metrics_t4,
                splits=splits,
                node_count=data_g1.num_nodes,
            )
        )

        result_g1 = train_residual_gnn(
            data=data_g1,
            splits=splits,
            device=device,
            cfg=gnn_cfg,
            seed=args.seed + step * 10 + 1,
        )
        refined_g1 = result_g1.refined_pred[test_mask_np]
        metrics_g1 = evaluate_predictions(y_true=y_true, y_pred=refined_g1)
        collect_errors(error_store, "G1", y_true, refined_g1)
        detailed_records.append(
            metrics_record(
                target_hour=target_hour,
                model_id="G1",
                metrics=metrics_g1,
                splits=splits,
                node_count=data_g1.num_nodes,
                best_epoch=result_g1.best_epoch,
                best_val_loss=result_g1.best_val_loss,
            )
        )

        result_g2 = train_residual_gnn(
            data=data_g2,
            splits=splits,
            device=device,
            cfg=gnn_cfg,
            seed=args.seed + step * 10 + 2,
        )
        refined_g2 = result_g2.refined_pred[test_mask_np]
        metrics_g2 = evaluate_predictions(y_true=y_true, y_pred=refined_g2)
        collect_errors(error_store, "G2", y_true, refined_g2)
        detailed_records.append(
            metrics_record(
                target_hour=target_hour,
                model_id="G2",
                metrics=metrics_g2,
                splits=splits,
                node_count=data_g2.num_nodes,
                best_epoch=result_g2.best_epoch,
                best_val_loss=result_g2.best_val_loss,
            )
        )

        print(
            f"[{target_hour}] T4 MAE={metrics_t4['MAE']:.4f}, "
            f"G1 MAE={metrics_g1['MAE']:.4f}, G2 MAE={metrics_g2['MAE']:.4f}"
        )

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
