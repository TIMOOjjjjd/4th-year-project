"""Experiment 5: compare OD, geographic, random, and no-graph baselines.

This experiment reuses Experiment 3's confidence-weighted residual GraphSAGE
pipeline. The only variable is the graph structure:

    no_graph: temporal baseline only, no residual GNN
    od:       residual GraphSAGE on the OD-flow graph
    geo:      residual GraphSAGE on the geographic graph
    random:   residual GraphSAGE on random graph baselines across fixed seeds

All graph variants use Experiment 3's learned-softmax confidence-weighted
residual loss.
For fairness, each rolling hour shares the same temporal predictions and the
same location-based train/validation/test split across all graph structures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from build_od_graph import _build_random_matrix, _load_zone_names
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
    build_gnn_features,
    build_graph_context,
    build_zone_hourly_counts,
    clean_checkpoint_dir,
    compute_confidence_components,
    evaluate_refined_predictions,
    run_multiscale_temporal_baseline,
    run_single_ablation,
    set_random_seed,
)


GRAPH_CONFIDENCE_MODE = "learned_softmax"
DEFAULT_OD_EDGE_MATRIX = BASE_DIR / "edge_weight_matrix_od.csv"
DEFAULT_GEO_EDGE_MATRIX = BASE_DIR / "edge_weight_matrix_with_flow.csv"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_experiment_5_multiscale"
DEFAULT_RANDOM_SEEDS = [1, 2, 3, 4, 5]


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
        description="Run Experiment 5: OD vs geographic vs random vs no graph."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP_PATH)
    parser.add_argument("--od-edge-csv", type=Path, default=DEFAULT_OD_EDGE_MATRIX)
    parser.add_argument("--geo-edge-csv", type=Path, default=DEFAULT_GEO_EDGE_MATRIX)
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
        "--graph-types",
        nargs="*",
        choices=["no_graph", "od", "geo", "random"],
        default=["no_graph", "od", "geo", "random"],
        help="Subset and order of graph variants to evaluate.",
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
    base_specs = {
        "no_graph": GraphSpec(
            key="no_graph",
            model_name="No Graph Temporal",
            description="temporal baseline only; no residual GraphSAGE",
            edge_csv=None,
        ),
        "od": GraphSpec(
            key="od",
            model_name="OD Graph",
            description="Experiment 3 learned-softmax GraphSAGE on OD-flow graph",
            edge_csv=args.od_edge_csv,
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


def load_taxi_data(data_path: Path, excluded_zones: Sequence[int]) -> pd.DataFrame:
    df = pd.read_parquet(data_path, columns=["pickup_datetime", "PULocationID"])
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
) -> Dict[str, GraphContext]:
    graphs: Dict[str, GraphContext] = {}
    for graph_key in requested_graphs:
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


def masks_from_location_splits(data, split_sets: LocationSplitSets) -> SplitMasks:
    location_ids = data.location_id.cpu().numpy().astype(int)
    train = torch.tensor([loc_id in split_sets.train for loc_id in location_ids])
    val = torch.tensor([loc_id in split_sets.val for loc_id in location_ids])
    test = torch.tensor([loc_id in split_sets.test for loc_id in location_ids])
    if int(train.sum()) == 0 or int(val.sum()) == 0 or int(test.sum()) == 0:
        raise ValueError("Graph data does not contain a non-empty train/val/test split.")
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
    split_sets: LocationSplitSets,
    device: torch.device,
    cfg: GNNTrainingConfig,
    seed: int,
    spec: GraphSpec,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data = build_gnn_features(step_df=step_df, graph=graph)
    splits = masks_from_location_splits(data=data, split_sets=split_sets)
    prediction_df, hourly_record = run_single_ablation(
        target_hour=target_hour,
        mode=GRAPH_CONFIDENCE_MODE,
        data=data,
        splits=splits,
        device=device,
        cfg=cfg,
        seed=seed,
    )

    prediction_df["graph_type"] = graph_key
    prediction_df["mode"] = graph_key
    prediction_df["experiment_3_mode"] = GRAPH_CONFIDENCE_MODE
    prediction_df["Model"] = spec.model_name
    prediction_df["random_seed"] = spec.random_seed

    hourly_record["graph_type"] = graph_key
    hourly_record["mode"] = graph_key
    hourly_record["experiment_3_mode"] = GRAPH_CONFIDENCE_MODE
    hourly_record["Model"] = spec.model_name
    hourly_record["random_seed"] = spec.random_seed
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
    generate_random_graphs(args=args, specs=specs, graph_order=requested_graphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("Graph variants:", ", ".join(requested_graphs))

    df = load_taxi_data(data_path=args.data, excluded_zones=args.excluded_zones)
    lookup_df = pd.read_csv(args.lookup).drop_duplicates(subset="LocationID")
    graphs = load_graph_contexts(
        specs=specs,
        requested_graphs=requested_graphs,
        lookup_df=lookup_df,
    )
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

    for step in range(args.rolling_steps):
        target_hour = args.start_target + pd.Timedelta(hours=step)
        print(f"\n///// Experiment 5 target hour: {target_hour} step {step} /////")

        baseline_df = run_multiscale_temporal_baseline(
            df=df,
            manager=manager,
            target_hour=target_hour,
            zones=zones,
            zone_hourly_counts=zone_hourly_counts,
        )
        history_df = df[df["datetime"] < target_hour]
        step_df = compute_confidence_components(
            step_df=baseline_df,
            history_df=history_df,
        )
        valid_locations = finite_step_zone_ids(step_df=step_df, candidate_zones=zones)
        split_sets = make_location_split_sets(
            location_ids=valid_locations,
            seed=args.seed + step,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

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
                    split_sets=split_sets,
                    device=device,
                    cfg=gnn_cfg,
                    seed=args.seed + step,
                    spec=spec,
                )

            detailed_frames.append(prediction_df)
            hourly_records.append(hourly_record)
            hour_messages.append(f"{graph_key} MAE={hourly_record['hourly_mae']:.4f}")

        print(f"[{target_hour}] " + ", ".join(hour_messages))

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
