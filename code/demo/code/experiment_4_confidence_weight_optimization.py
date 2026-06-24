"""Experiment 4: confidence component weight optimization.

This script compares three confidence-weight optimization schemes against the
fixed 0.3/0.4/0.3 confidence used in Experiment 2.

All modes keep the same residual refinement formulation:

    residual_target = true_value - base_pred
    refined_pred = base_pred + residual_pred

The compared confidence weight modes are:

    fixed: fixed prior/stability/history-consistency weights, 0.3/0.4/0.3
    grid_search: validation-selected simplex grid weights
    random_search: validation-selected Dirichlet random weights
    learned_softmax: trainable softmax component weights

The confidence score is used as a normalized residual-loss sample weight. It is
not used as a training label.

All GNN training and confidence-weight validation use residual snapshots from
hours strictly before each rolling target hour. Target-hour labels are used only
after inference for test metrics.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

from experiment_3_confidence_weighted_gnn_ablation import (
    BASE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_WINDOW_STARTS,
    EXCLUDED_ZONES,
    ROLLING_STEPS,
    START_TARGET,
    GNNResult,
    GNNTrainingConfig,
    GraphContext,
    HISTORY_FEATURES,
    SplitMasks,
    build_graph_context,
    build_zone_hourly_counts,
    clamp_score,
    clean_checkpoint_dir,
    compute_prior_scores,
    compute_stability_scores,
    evaluate_refined_predictions,
    filter_history_frames_before,
    load_required_data,
    make_location_split_sets,
    masks_from_location_splits,
    remap_edges_to_valid_nodes,
    run_multiscale_temporal_baseline,
    resolve_window_starts,
    select_zones,
    set_random_seed,
    validate_windows,
)
from gnn_model import MultiScaleGraphSAGE
from persistent_tcn import ManagerConfig, MultiScaleModelManager
from residual_graph_utils import (
    build_dynamic_residual_graph_context,
    residual_graph_summary_path,
)


COMPONENT_NAMES = ("prior", "stability", "history_consistency")
FIXED_WEIGHTS = np.array([0.3, 0.4, 0.3], dtype=np.float32)
DEFAULT_OPT_EDGE_WEIGHT_MATRIX = BASE_DIR / "edge_weight_matrix_od.csv"
DEFAULT_GRAPH_TYPE = "residual"
DEFAULT_RESIDUAL_WINDOW_HOURS = 24 * 30
HISTORY_CONSISTENCY_TAU = 1.0

MODE_SPECS = {
    "fixed": {
        "model": "Fixed confidence weights",
        "description": "fixed log-space weights 0.3/0.4/0.3",
    },
    "grid_search": {
        "model": "Grid-searched confidence weights",
        "description": "validation-selected simplex grid weights",
    },
    "random_search": {
        "model": "Random-searched confidence weights",
        "description": "validation-selected Dirichlet random weights",
    },
    "learned_softmax": {
        "model": "Learned softmax confidence weights",
        "description": "component weights optimized jointly with GraphSAGE",
    },
}
MODE_ORDER = ["fixed", "grid_search", "random_search", "learned_softmax"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: confidence weight optimization."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_OPT_EDGE_WEIGHT_MATRIX)
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=BASE_DIR / "checkpoints_tcn_shared_v1",
    )
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
        "--modes",
        nargs="*",
        choices=MODE_ORDER,
        default=MODE_ORDER,
        help="Subset of confidence optimization modes to run.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.25,
        help="Simplex grid step for grid_search. 0.25 gives 15 candidates.",
    )
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=24,
        help="Number of Dirichlet candidates for random_search.",
    )
    parser.add_argument(
        "--learned-entropy",
        type=float,
        default=0.0,
        help="Optional entropy regularizer for learned_softmax weights.",
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
        help="Optional smoke-test limit. Default evaluates all graph-compatible zones.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing experiment_4_confidence_weight_optimization_metrics.png.",
    )
    args = parser.parse_args()
    if not args.modes:
        parser.error("--modes requires at least one mode when provided.")
    if args.residual_window_hours <= 0:
        parser.error("--residual-window-hours must be > 0.")
    if args.residual_top_k < 0:
        parser.error("--residual-top-k must be >= 0.")
    if args.residual_min_corr < 0:
        parser.error("--residual-min-corr must be >= 0.")
    return args


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
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the three confidence components used by Experiment 4."""
    step_df = step_df.copy()
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
        lambda zone_id: clamp_score(history_scores.get(int(zone_id), 0.6), default=0.6)
    )
    step_df["full_confidence"] = (
        np.exp(
            FIXED_WEIGHTS[0] * np.log(step_df["prior_score"].clip(0.05, 1.0))
            + FIXED_WEIGHTS[1] * np.log(step_df["stability_score"].clip(0.05, 1.0))
            + FIXED_WEIGHTS[2]
            * np.log(step_df["history_consistency_score"].clip(0.05, 1.0))
        )
    ).map(lambda value: clamp_score(float(value), default=0.5))
    return step_df


def build_gnn_features(step_df: pd.DataFrame, graph: GraphContext) -> Data:
    """Build node features and residual targets with confidence in x."""
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

    edge_index = remap_edges_to_valid_nodes(graph.edge_index, valid_indices)
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

    return Data(
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


def build_historical_gnn_training_data(
    history_frames: Sequence[pd.DataFrame],
    graph: GraphContext,
) -> Optional[Data]:
    """Concatenate historical residual snapshots into one train graph."""

    snapshots: List[Data] = []
    for frame in history_frames:
        if frame.empty:
            continue
        try:
            snapshots.append(build_gnn_features(step_df=frame, graph=graph))
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
    history_parts: List[torch.Tensor] = []
    confidence_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []

    offset = 0
    for snapshot in snapshots:
        x_parts.append(snapshot.x)
        y_parts.append(snapshot.y)
        base_parts.append(snapshot.base_pred)
        true_parts.append(snapshot.true_value)
        location_parts.append(snapshot.location_id)
        prior_parts.append(snapshot.prior_score)
        stability_parts.append(snapshot.stability_score)
        history_parts.append(snapshot.history_consistency_score)
        confidence_parts.append(snapshot.full_confidence)

        if snapshot.edge_index.numel() > 0:
            edge_parts.append(snapshot.edge_index + offset)
        offset += int(snapshot.num_nodes)

    edge_index = (
        torch.cat(edge_parts, dim=1)
        if edge_parts
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=edge_index,
        y=torch.cat(y_parts, dim=0),
        base_pred=torch.cat(base_parts, dim=0),
        true_value=torch.cat(true_parts, dim=0),
        location_id=torch.cat(location_parts, dim=0),
        prior_score=torch.cat(prior_parts, dim=0),
        stability_score=torch.cat(stability_parts, dim=0),
        history_consistency_score=torch.cat(history_parts, dim=0),
        full_confidence=torch.cat(confidence_parts, dim=0),
    )


def component_tensor(data: Data) -> torch.Tensor:
    return torch.stack(
        [
            data.prior_score.float(),
            data.stability_score.float(),
            data.history_consistency_score.float(),
        ],
        dim=1,
    ).clamp(min=0.05, max=1.0)


def confidence_from_weights(data: Data, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    weights = weights / weights.sum().clamp(min=1e-6)
    components = component_tensor(data).to(weights.device)
    log_confidence = (torch.log(components) * weights.unsqueeze(0)).sum(dim=1)
    return torch.exp(log_confidence).clamp(min=0.05, max=1.0)


def train_residual_gnn_with_sample_weight(
    train_data: Data,
    train_splits: SplitMasks,
    train_sample_weight: torch.Tensor,
    inference_data: Data,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> Tuple[GNNResult, GNNResult]:
    set_random_seed(seed)
    model = MultiScaleGraphSAGE(
        in_dim=int(train_data.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    train_data_device = copy.deepcopy(train_data).to(device)
    sample_weight = train_sample_weight.to(device).float().clamp(min=0.05, max=1.0)
    train_mask = train_splits.train.to(device)
    val_mask = train_splits.val.to(device)

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
        residual_pred, _ = model(train_data_device)

        loss_per_node = loss_func(
            residual_pred[train_mask],
            train_data_device.y[train_mask],
        )
        train_weight = sample_weight[train_mask]
        train_weight = train_weight / train_weight.mean().clamp(min=1e-6)
        loss = (loss_per_node * train_weight).mean()
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    inference_data_device = copy.deepcopy(inference_data).to(device)
    with torch.no_grad():
        train_residual_pred, train_refined_pred = model(train_data_device)
        residual_pred, refined_pred = model(inference_data_device)

    return (
        GNNResult(
            residual_pred=residual_pred.detach().cpu().numpy(),
            refined_pred=refined_pred.detach().cpu().numpy(),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        ),
        GNNResult(
            residual_pred=train_residual_pred.detach().cpu().numpy(),
            refined_pred=train_refined_pred.detach().cpu().numpy(),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        ),
    )


def train_residual_gnn_with_learned_weights(
    train_data: Data,
    train_splits: SplitMasks,
    inference_data: Data,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
    entropy_weight: float,
) -> Tuple[GNNResult, GNNResult, np.ndarray]:
    set_random_seed(seed)
    model = MultiScaleGraphSAGE(
        in_dim=int(train_data.x.shape[1]),
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    logits = nn.Parameter(
        torch.log(torch.tensor(FIXED_WEIGHTS, dtype=torch.float32)).to(device)
    )

    train_data_device = copy.deepcopy(train_data).to(device)
    train_mask = train_splits.train.to(device)
    val_mask = train_splits.val.to(device)
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

        weights = torch.softmax(logits, dim=0)
        sample_weight = confidence_from_weights(train_data_device, weights)
        residual_pred, _ = model(train_data_device)
        loss_per_node = loss_func(
            residual_pred[train_mask],
            train_data_device.y[train_mask],
        )
        train_weight = sample_weight[train_mask]
        train_weight = train_weight / train_weight.mean().clamp(min=1e-6)
        loss = (loss_per_node * train_weight).mean()
        if entropy_weight > 0.0:
            entropy = -(weights * torch.log(weights.clamp(min=1e-8))).sum()
            loss = loss - entropy_weight * entropy

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

    learned_weights = torch.softmax(logits.detach(), dim=0)
    model.eval()
    inference_data_device = copy.deepcopy(inference_data).to(device)
    with torch.no_grad():
        train_residual_pred, train_refined_pred = model(train_data_device)
        residual_pred, refined_pred = model(inference_data_device)

    return (
        GNNResult(
            residual_pred=residual_pred.detach().cpu().numpy(),
            refined_pred=refined_pred.detach().cpu().numpy(),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        ),
        GNNResult(
            residual_pred=train_residual_pred.detach().cpu().numpy(),
            refined_pred=train_refined_pred.detach().cpu().numpy(),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        ),
        learned_weights.detach().cpu().numpy(),
    )


def simplex_grid(step: float) -> List[np.ndarray]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("--grid-step must be in (0, 1].")

    values = np.arange(0.0, 1.0 + step / 2.0, step)
    candidates: List[np.ndarray] = []
    for w_prior in values:
        for w_stability in values:
            w_history = 1.0 - w_prior - w_stability
            if w_history < -1e-8:
                continue
            weights = np.array(
                [w_prior, w_stability, max(0.0, w_history)],
                dtype=np.float32,
            )
            weights = weights / weights.sum()
            candidates.append(weights)
    return candidates


def random_simplex_candidates(count: int, seed: int) -> List[np.ndarray]:
    if count <= 0:
        raise ValueError("--random-candidates must be positive.")
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(np.ones(len(COMPONENT_NAMES)), size=count)
    return [sample.astype(np.float32) for sample in samples]


def evaluate_split(
    data: Data,
    result: GNNResult,
    mask: torch.Tensor,
) -> Dict[str, float]:
    mask_np = mask.cpu().numpy().astype(bool)
    return evaluate_refined_predictions(
        y_true=data.true_value.cpu().numpy()[mask_np],
        y_pred=result.refined_pred[mask_np],
    )


def split_labels(splits: SplitMasks) -> np.ndarray:
    labels = np.full(splits.train.numel(), "test", dtype=object)
    labels[splits.train.cpu().numpy()] = "train"
    labels[splits.val.cpu().numpy()] = "val"
    labels[splits.test.cpu().numpy()] = "test"
    return labels


def collect_predictions(
    target_hour: pd.Timestamp,
    mode: str,
    data: Data,
    splits: SplitMasks,
    result: GNNResult,
    sample_weight: torch.Tensor,
    weights: np.ndarray,
) -> pd.DataFrame:
    labels = split_labels(splits)
    return pd.DataFrame(
        {
            "target_hour": target_hour,
            "mode": mode,
            "Model": MODE_SPECS[mode]["model"],
            "PULocationID": data.location_id.cpu().numpy().astype(int),
            "split": labels,
            "is_test": labels == "test",
            "base_pred": data.base_pred.cpu().numpy(),
            "true_value": data.true_value.cpu().numpy(),
            "residual_target": data.y.cpu().numpy(),
            "residual_pred": result.residual_pred,
            "refined_pred": result.refined_pred,
            "sample_weight": sample_weight.cpu().numpy(),
            "prior_score": data.prior_score.cpu().numpy(),
            "stability_score": data.stability_score.cpu().numpy(),
            "history_consistency_score": data.history_consistency_score.cpu().numpy(),
            "w_prior": float(weights[0]),
            "w_stability": float(weights[1]),
            "w_history_consistency": float(weights[2]),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        }
    )


def train_candidate(
    train_data: Data,
    train_splits: SplitMasks,
    inference_data: Data,
    weights: np.ndarray,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> Tuple[GNNResult, torch.Tensor, Dict[str, float]]:
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    train_sample_weight = confidence_from_weights(train_data, weight_tensor)
    result, train_result = train_residual_gnn_with_sample_weight(
        train_data=train_data,
        train_splits=train_splits,
        train_sample_weight=train_sample_weight,
        inference_data=inference_data,
        device=device,
        cfg=cfg,
        seed=seed,
    )
    val_metrics = evaluate_split(data=train_data, result=train_result, mask=train_splits.val)
    inference_sample_weight = confidence_from_weights(inference_data, weight_tensor)
    return result, inference_sample_weight, val_metrics


def run_weight_search_mode(
    mode: str,
    candidates: Sequence[np.ndarray],
    train_data: Data,
    train_splits: SplitMasks,
    inference_data: Data,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
) -> Tuple[GNNResult, torch.Tensor, np.ndarray, Dict[str, float]]:
    best_result: Optional[GNNResult] = None
    best_sample_weight: Optional[torch.Tensor] = None
    best_weights: Optional[np.ndarray] = None
    best_val_metrics: Optional[Dict[str, float]] = None
    best_val_mae = float("inf")

    for candidate_idx, weights in enumerate(candidates):
        result, sample_weight, val_metrics = train_candidate(
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            weights=weights,
            device=device,
            cfg=cfg,
            seed=seed,
        )
        val_mae = val_metrics["MAE"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_result = result
            best_sample_weight = sample_weight
            best_weights = weights
            best_val_metrics = val_metrics
        print(
            f"    {mode} candidate {candidate_idx + 1}/{len(candidates)} "
            f"val_MAE={val_mae:.4f} weights={format_weights(weights)}"
        )

    if best_result is None or best_sample_weight is None or best_weights is None:
        raise RuntimeError(f"No valid candidates for mode {mode}.")
    if best_val_metrics is None:
        best_val_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
    return best_result, best_sample_weight, best_weights, best_val_metrics


def run_single_mode(
    target_hour: pd.Timestamp,
    mode: str,
    train_data: Optional[Data],
    train_splits: Optional[SplitMasks],
    inference_data: Data,
    inference_splits: SplitMasks,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
    grid_step: float,
    random_candidates: int,
    learned_entropy: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if train_data is None or train_splits is None:
        selected_weights = FIXED_WEIGHTS
        result = GNNResult(
            residual_pred=np.zeros_like(
                inference_data.base_pred.detach().cpu().numpy(),
                dtype=np.float32,
            ),
            refined_pred=inference_data.base_pred.detach().cpu().numpy().astype(np.float32),
            best_epoch=0,
            best_val_loss=float("nan"),
        )
        sample_weight = confidence_from_weights(
            inference_data,
            torch.tensor(selected_weights, dtype=torch.float32),
        )
        val_metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MSE": float("nan")}
    elif mode == "fixed":
        result, sample_weight, val_metrics = train_candidate(
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            weights=FIXED_WEIGHTS,
            device=device,
            cfg=cfg,
            seed=seed,
        )
        selected_weights = FIXED_WEIGHTS
    elif mode == "grid_search":
        candidates = simplex_grid(grid_step)
        result, sample_weight, selected_weights, val_metrics = run_weight_search_mode(
            mode=mode,
            candidates=candidates,
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            device=device,
            cfg=cfg,
            seed=seed,
        )
    elif mode == "random_search":
        candidates = random_simplex_candidates(count=random_candidates, seed=seed)
        result, sample_weight, selected_weights, val_metrics = run_weight_search_mode(
            mode=mode,
            candidates=candidates,
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            device=device,
            cfg=cfg,
            seed=seed,
        )
    elif mode == "learned_softmax":
        result, train_result, selected_weights = train_residual_gnn_with_learned_weights(
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            device=device,
            cfg=cfg,
            seed=seed,
            entropy_weight=learned_entropy,
        )
        sample_weight = confidence_from_weights(
            inference_data,
            torch.tensor(selected_weights, dtype=torch.float32),
        )
        val_metrics = evaluate_split(data=train_data, result=train_result, mask=train_splits.val)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    test_metrics = evaluate_split(
        data=inference_data,
        result=result,
        mask=inference_splits.test,
    )
    prediction_df = collect_predictions(
        target_hour=target_hour,
        mode=mode,
        data=inference_data,
        splits=inference_splits,
        result=result,
        sample_weight=sample_weight,
        weights=selected_weights,
    )
    hourly_record = {
        "target_hour": target_hour,
        "mode": mode,
        "Model": MODE_SPECS[mode]["model"],
        "Description": MODE_SPECS[mode]["description"],
        "val_mae": val_metrics["MAE"],
        "val_rmse": val_metrics["RMSE"],
        "val_mse": val_metrics["MSE"],
        "test_mae": test_metrics["MAE"],
        "test_rmse": test_metrics["RMSE"],
        "test_mse": test_metrics["MSE"],
        "w_prior": float(selected_weights[0]),
        "w_stability": float(selected_weights[1]),
        "w_history_consistency": float(selected_weights[2]),
        "n_train": int(train_splits.train.sum().item()) if train_splits is not None else 0,
        "n_val": int(train_splits.val.sum().item()) if train_splits is not None else 0,
        "n_test": int(inference_splits.test.sum().item()),
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
    }
    return prediction_df, hourly_record


def format_weights(weights: np.ndarray) -> str:
    return (
        f"prior={weights[0]:.3f}, "
        f"stability={weights[1]:.3f}, "
        f"history_consistency={weights[2]:.3f}"
    )


def summarize_results(detailed_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    fixed_mae = float("nan")

    for mode in MODE_ORDER:
        if mode not in set(hourly_df["mode"]):
            continue

        mode_df = detailed_df[(detailed_df["mode"] == mode) & detailed_df["is_test"]]
        metrics = evaluate_refined_predictions(
            y_true=mode_df["true_value"].to_numpy(dtype=float),
            y_pred=mode_df["refined_pred"].to_numpy(dtype=float),
        )
        if mode == "fixed":
            fixed_mae = metrics["MAE"]

        mode_hourly = hourly_df[hourly_df["mode"] == mode]
        rows.append(
            {
                "mode": mode,
                "Model": MODE_SPECS[mode]["model"],
                "Description": MODE_SPECS[mode]["description"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MSE": metrics["MSE"],
                "Std hourly MAE": float(mode_hourly["test_mae"].std(ddof=0)),
                "Mean w_prior": float(mode_hourly["w_prior"].mean()),
                "Mean w_stability": float(mode_hourly["w_stability"].mean()),
                "Mean w_history_consistency": float(
                    mode_hourly["w_history_consistency"].mean()
                ),
            }
        )

    summary_df = pd.DataFrame(rows)
    if np.isfinite(fixed_mae) and fixed_mae != 0.0:
        summary_df["MAE improvement over fixed (%)"] = (
            (fixed_mae - summary_df["MAE"]) / fixed_mae * 100.0
        )
    else:
        summary_df["MAE improvement over fixed (%)"] = np.nan
    return summary_df


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    historical_step_frames: List[pd.DataFrame] = []
    residual_prediction_cache: Dict[Tuple[int, pd.Timestamp], float] = {}
    residual_summary_records: List[Dict[str, object]] = []

    for window_idx, window_start in enumerate(window_starts, start=1):
        print(
            f"\n===== Experiment 4 window {window_idx}/{len(window_starts)} "
            f"start={window_start} ====="
        )
        for step in range(args.rolling_steps):
            global_step = (window_idx - 1) * args.rolling_steps + step
            target_hour = window_start + pd.Timedelta(hours=step)
            print(
                f"\n///// Experiment 4 target hour: {target_hour} "
                f"window {window_idx} step {step} /////"
            )

            set_random_seed(args.seed + global_step)
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
            history_df = df[df["datetime"] < target_hour]
            step_df = compute_confidence_components(
                step_df=baseline_df,
                history_df=history_df,
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
            for mode in args.modes:
                prediction_df, hourly_record = run_single_mode(
                    target_hour=target_hour,
                    mode=mode,
                    train_data=train_data,
                    train_splits=train_splits,
                    inference_data=inference_data,
                    inference_splits=inference_splits,
                    device=device,
                    cfg=gnn_cfg,
                    seed=args.seed + global_step,
                    grid_step=args.grid_step,
                    random_candidates=args.random_candidates,
                    learned_entropy=args.learned_entropy,
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
                weights = np.array(
                    [
                        hourly_record["w_prior"],
                        hourly_record["w_stability"],
                        hourly_record["w_history_consistency"],
                    ],
                    dtype=np.float32,
                )
                hour_messages.append(
                    f"{mode} test_MAE={hourly_record['test_mae']:.4f} "
                    f"({format_weights(weights)})"
                )

            print(f"[{target_hour}] " + ", ".join(hour_messages))
            historical_step_frames.append(step_df.copy())

    detailed_df = pd.concat(detailed_frames, ignore_index=True)
    hourly_df = pd.DataFrame(hourly_records)
    summary_df = summarize_results(detailed_df=detailed_df, hourly_df=hourly_df)
    return summary_df, detailed_df, hourly_df

def save_results(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    results_dir: Path,
    make_plot: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "experiment_4_confidence_weight_optimization_summary.csv"
    detailed_path = results_dir / "experiment_4_confidence_weight_optimization_detailed.csv"
    hourly_path = results_dir / "experiment_4_confidence_weight_optimization_hourly.csv"
    plot_path = results_dir / "experiment_4_confidence_weight_optimization_metrics.png"

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    hourly_df.to_csv(hourly_path, index=False)

    plot_written = False
    if make_plot:
        try:
            plot_metrics(summary_df=summary_df, hourly_df=hourly_df, plot_path=plot_path)
            plot_written = True
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    print(f"\nSaved summary to {summary_path}")
    print(f"Saved detailed predictions to {detailed_path}")
    print(f"Saved hourly records to {hourly_path}")
    if plot_written:
        print(f"Saved plot to {plot_path}")


def plot_metrics(summary_df: pd.DataFrame, hourly_df: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    models = summary_df["Model"].tolist()
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(x, summary_df["MAE"], label="MAE")
    axes[0].set_title("Overall Test MAE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - 0.25, summary_df["Mean w_prior"], 0.25, label="prior")
    axes[1].bar(x, summary_df["Mean w_stability"], 0.25, label="stability")
    axes[1].bar(
        x + 0.25,
        summary_df["Mean w_history_consistency"],
        0.25,
        label="history consistency",
    )
    axes[1].set_title("Mean Selected Weights")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=18, ha="right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()

    for mode in MODE_ORDER:
        mode_df = hourly_df[hourly_df["mode"] == mode].sort_values("target_hour")
        if mode_df.empty:
            continue
        axes[2].plot(
            pd.to_datetime(mode_df["target_hour"]),
            mode_df["test_mae"],
            marker="o",
            linewidth=1.5,
            label=MODE_SPECS[mode]["model"],
        )
    axes[2].set_title("Hourly Test MAE")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].legend(fontsize=8)

    fig.suptitle("Experiment 4: Confidence Weight Optimization")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def print_summary_table(summary_df: pd.DataFrame) -> None:
    display_columns = [
        "Model",
        "MAE",
        "RMSE",
        "MSE",
        "Mean w_prior",
        "Mean w_stability",
        "Mean w_history_consistency",
        "MAE improvement over fixed (%)",
    ]
    display_df = summary_df[display_columns].copy()
    for column in display_columns[1:]:
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
