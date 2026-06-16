"""Experiment 6: confidence reliability validation.

This experiment checks whether the learned confidence score is aligned with
prediction reliability on held-out test nodes.

For each rolling-hour test node it records:

    learned_confidence = learned-softmax confidence sample weight
    absolute_error = abs(true_value - refined_pred)

The test predictions are sorted by learned confidence and split into:

    Top 25% confidence
    Middle 50% confidence
    Bottom 25% confidence

The experiment reports MAE, RMSE, and mean absolute residual for each group,
then computes Pearson and Spearman correlations between learned confidence and
absolute error. A reliable confidence score should show lower errors in the
higher-confidence groups and a negative confidence-vs-error correlation.

The learned-softmax GNN is trained only on residual snapshots from hours
strictly before each target hour. Target-hour labels are used only after
inference to compute reliability metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from experiment_3_confidence_weighted_gnn_ablation import (
    BASE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_EDGE_WEIGHT_MATRIX,
    DEFAULT_LOOKUP_PATH,
    DEFAULT_RESULTS_DIR,
    EXCLUDED_ZONES,
    ROLLING_STEPS,
    START_TARGET,
    GNNTrainingConfig,
    ManagerConfig,
    MultiScaleModelManager,
    SplitMasks,
    build_historical_gnn_training_data,
    build_gnn_features,
    build_zone_hourly_counts,
    clean_checkpoint_dir,
    compute_confidence_components,
    filter_history_frames_before,
    load_required_data,
    make_location_split_sets,
    masks_from_location_splits,
    run_multiscale_temporal_baseline,
    run_single_ablation,
    select_zones,
    set_random_seed,
)


DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_experiment_6_multiscale"
EXPERIMENT_MODE = "learned_softmax"
OUTPUT_STEM = "experiment_6_confidence_reliability_validation"
CONFIDENCE_GROUP_ORDER = [
    "Top 25% confidence",
    "Middle 50% confidence",
    "Bottom 25% confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiment 6: confidence reliability validation."
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
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=None,
        help=(
            "Optional existing detailed prediction CSV to analyse instead of "
            "rerunning temporal and GNN inference."
        ),
    )
    parser.add_argument(
        "--mode",
        default=EXPERIMENT_MODE,
        help=(
            "Prediction mode to analyse when --detailed-csv contains multiple modes. "
            "Use 'all' to disable mode filtering."
        ),
    )
    parser.add_argument("--confidence-column", default="sample_weight")
    parser.add_argument("--prediction-column", default="refined_pred")
    parser.add_argument("--truth-column", default="true_value")
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
        help=f"Skip writing {OUTPUT_STEM}_metrics.png.",
    )
    args = parser.parse_args()
    if args.quantile_bins < 2:
        parser.error("--quantile-bins must be at least 2.")
    return args


def finite_metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = finite_metric_values(y_true=y_true, y_pred=y_pred)
    if y_true.size == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "Mean Absolute Residual": float("nan"),
        }

    residual = y_true - y_pred
    absolute_residual = np.abs(residual)
    return {
        "MAE": float(absolute_residual.mean()),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "Mean Absolute Residual": float(absolute_residual.mean()),
    }


def test_mask_from_predictions(predictions_df: pd.DataFrame) -> pd.Series:
    if "is_test" in predictions_df.columns:
        values = predictions_df["is_test"]
        if values.dtype == bool:
            return values
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "1", "yes", "y"})

    if "split" in predictions_df.columns:
        return predictions_df["split"].astype(str).str.lower().eq("test")

    raise ValueError("Prediction dataframe must contain either 'is_test' or 'split'.")


def filter_prediction_mode(predictions_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return predictions_df.copy()

    filter_columns = []
    if "mode" in predictions_df.columns:
        filter_columns.append("mode")
        mode_mask = predictions_df["mode"].astype(str).eq(mode)
        if bool(mode_mask.any()):
            return predictions_df[mode_mask].copy()

    if "experiment_3_mode" in predictions_df.columns:
        filter_columns.append("experiment_3_mode")
        mode_mask = predictions_df["experiment_3_mode"].astype(str).eq(mode)
        if bool(mode_mask.any()):
            return predictions_df[mode_mask].copy()

    if filter_columns:
        raise ValueError(
            f"No predictions found for mode '{mode}' in columns {filter_columns}. "
            "Use --mode all to analyse every row."
        )
    return predictions_df.copy()


def prepare_test_predictions(
    predictions_df: pd.DataFrame,
    mode: str,
    confidence_column: str,
    prediction_column: str,
    truth_column: str,
) -> pd.DataFrame:
    predictions_df = filter_prediction_mode(predictions_df=predictions_df, mode=mode)
    required = {confidence_column, prediction_column, truth_column}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing columns: {sorted(missing)}")

    test_df = predictions_df[test_mask_from_predictions(predictions_df)].copy()
    if test_df.empty:
        raise ValueError("No test predictions found for confidence reliability validation.")

    test_df["learned_confidence"] = pd.to_numeric(
        test_df[confidence_column],
        errors="coerce",
    )
    test_df["y_pred"] = pd.to_numeric(test_df[prediction_column], errors="coerce")
    test_df["y_true"] = pd.to_numeric(test_df[truth_column], errors="coerce")
    test_df = test_df[
        np.isfinite(test_df["learned_confidence"].to_numpy(dtype=float))
        & np.isfinite(test_df["y_pred"].to_numpy(dtype=float))
        & np.isfinite(test_df["y_true"].to_numpy(dtype=float))
    ].copy()
    if test_df.empty:
        raise ValueError("No finite test predictions with learned confidence were found.")

    test_df["residual"] = test_df["y_true"] - test_df["y_pred"]
    test_df["absolute_error"] = test_df["residual"].abs()
    if "target_hour" in test_df.columns:
        test_df["target_hour"] = pd.to_datetime(test_df["target_hour"])
    return assign_confidence_groups(test_df)


def assign_confidence_groups(test_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = test_df.sort_values(
        ["learned_confidence", "absolute_error"],
        ascending=[False, True],
    ).reset_index(drop=True)
    n_rows = len(sorted_df)
    top_count = max(1, int(np.ceil(n_rows * 0.25)))
    bottom_start = max(top_count, int(np.floor(n_rows * 0.75)))

    labels = np.full(n_rows, "Middle 50% confidence", dtype=object)
    labels[:top_count] = "Top 25% confidence"
    labels[bottom_start:] = "Bottom 25% confidence"

    sorted_df["confidence_rank"] = np.arange(1, n_rows + 1)
    sorted_df["confidence_percentile"] = 1.0 - (
        (sorted_df["confidence_rank"] - 1) / max(n_rows - 1, 1)
    )
    sorted_df["confidence_group"] = pd.Categorical(
        labels,
        categories=CONFIDENCE_GROUP_ORDER,
        ordered=True,
    )
    return sorted_df


def summarize_confidence_groups(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_name in CONFIDENCE_GROUP_ORDER:
        group_df = test_df[test_df["confidence_group"] == group_name]
        metrics = error_metrics(
            y_true=group_df["y_true"].to_numpy(dtype=float),
            y_pred=group_df["y_pred"].to_numpy(dtype=float),
        )
        confidence_values = group_df["learned_confidence"].to_numpy(dtype=float)
        rows.append(
            {
                "Confidence Group": group_name,
                "n_predictions": int(len(group_df)),
                "mean_confidence": (
                    float(np.mean(confidence_values)) if confidence_values.size else np.nan
                ),
                "min_confidence": (
                    float(np.min(confidence_values)) if confidence_values.size else np.nan
                ),
                "max_confidence": (
                    float(np.max(confidence_values)) if confidence_values.size else np.nan
                ),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "Mean Absolute Residual": metrics["Mean Absolute Residual"],
            }
        )
    return pd.DataFrame(rows)


def compute_correlations(test_df: pd.DataFrame) -> pd.DataFrame:
    values = test_df[["learned_confidence", "absolute_error"]].dropna()
    rows = []
    for method in ["pearson", "spearman"]:
        if values["learned_confidence"].nunique() <= 1 or values["absolute_error"].nunique() <= 1:
            correlation = np.nan
        else:
            correlation = float(
                values["learned_confidence"].corr(values["absolute_error"], method=method)
            )
        rows.append(
            {
                "metric": f"{method}_confidence_vs_absolute_error",
                "correlation": correlation,
                "n_predictions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def compute_quantile_errors(test_df: pd.DataFrame, quantile_bins: int) -> pd.DataFrame:
    quantile_count = min(int(quantile_bins), len(test_df))
    sorted_df = test_df.sort_values("learned_confidence", ascending=True).reset_index(drop=True)
    quantile_ids = np.floor(np.arange(len(sorted_df)) * quantile_count / len(sorted_df)).astype(int)
    sorted_df["confidence_quantile"] = quantile_ids + 1

    rows: List[Dict[str, object]] = []
    for quantile_id in range(1, quantile_count + 1):
        quantile_df = sorted_df[sorted_df["confidence_quantile"] == quantile_id]
        metrics = error_metrics(
            y_true=quantile_df["y_true"].to_numpy(dtype=float),
            y_pred=quantile_df["y_pred"].to_numpy(dtype=float),
        )
        confidence_values = quantile_df["learned_confidence"].to_numpy(dtype=float)
        rows.append(
            {
                "confidence_quantile": quantile_id,
                "quantile_label": f"Q{quantile_id} low->high",
                "n_predictions": int(len(quantile_df)),
                "mean_confidence": float(np.mean(confidence_values)),
                "min_confidence": float(np.min(confidence_values)),
                "max_confidence": float(np.max(confidence_values)),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "Mean Absolute Residual": metrics["Mean Absolute Residual"],
            }
        )
    return pd.DataFrame(rows)


def run_prediction_generation(args: argparse.Namespace) -> pd.DataFrame:
    set_random_seed(args.seed)
    if args.clean_checkpoints:
        clean_checkpoint_dir(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("Confidence mode:", EXPERIMENT_MODE)

    df, _lookup_df, graph = load_required_data(
        data_path=args.data,
        lookup_path=args.lookup,
        edge_csv=args.edge_csv,
        excluded_zones=args.excluded_zones,
    )
    zones = select_zones(df, graph, args.max_zones)
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
    historical_step_frames: List[pd.DataFrame] = []
    for step in range(args.rolling_steps):
        target_hour = args.start_target + pd.Timedelta(hours=step)
        print(f"\n///// Experiment 6 target hour: {target_hour} step {step} /////")

        baseline_df = run_multiscale_temporal_baseline(
            df=df,
            manager=manager,
            target_hour=target_hour,
            zones=zones,
            zone_hourly_counts=zone_hourly_counts,
        )
        history_df = df[df["datetime"] < target_hour]
        step_df = compute_confidence_components(step_df=baseline_df, history_df=history_df)
        inference_data = build_gnn_features(step_df=step_df, graph=graph)
        split_sets = make_location_split_sets(
            location_ids=inference_data.location_id.cpu().numpy().astype(int),
            seed=args.seed + step,
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

        prediction_df, hourly_record = run_single_ablation(
            target_hour=target_hour,
            mode=EXPERIMENT_MODE,
            train_data=train_data,
            train_splits=train_splits,
            inference_data=inference_data,
            inference_splits=inference_splits,
            device=device,
            cfg=gnn_cfg,
            seed=args.seed + step,
        )
        detailed_frames.append(prediction_df)
        test_mean_confidence = prediction_df.loc[
            prediction_df["is_test"],
            "sample_weight",
        ].mean()
        print(
            f"[{target_hour}] test_MAE={hourly_record['hourly_mae']:.4f} "
            f"mean_test_confidence={test_mean_confidence:.4f}"
        )
        historical_step_frames.append(step_df.copy())

    return pd.concat(detailed_frames, ignore_index=True)


def run_experiment(
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.detailed_csv is not None:
        detailed_df = pd.read_csv(args.detailed_csv)
        print(f"Loaded detailed predictions from {args.detailed_csv}")
    else:
        detailed_df = run_prediction_generation(args)

    test_predictions_df = prepare_test_predictions(
        predictions_df=detailed_df,
        mode=args.mode,
        confidence_column=args.confidence_column,
        prediction_column=args.prediction_column,
        truth_column=args.truth_column,
    )
    summary_df = summarize_confidence_groups(test_df=test_predictions_df)
    correlations_df = compute_correlations(test_df=test_predictions_df)
    quantile_df = compute_quantile_errors(
        test_df=test_predictions_df,
        quantile_bins=args.quantile_bins,
    )
    return summary_df, correlations_df, quantile_df, detailed_df, test_predictions_df


def save_results(
    summary_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    results_dir: Path,
    make_plot: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{OUTPUT_STEM}_summary.csv"
    correlations_path = results_dir / f"{OUTPUT_STEM}_correlations.csv"
    quantile_path = results_dir / f"{OUTPUT_STEM}_quantile_errors.csv"
    detailed_path = results_dir / f"{OUTPUT_STEM}_detailed.csv"
    test_predictions_path = results_dir / f"{OUTPUT_STEM}_test_predictions.csv"
    plot_path = results_dir / f"{OUTPUT_STEM}_metrics.png"

    summary_df.to_csv(summary_path, index=False)
    correlations_df.to_csv(correlations_path, index=False)
    quantile_df.to_csv(quantile_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    test_predictions_df.to_csv(test_predictions_path, index=False)

    plot_written = False
    if make_plot:
        try:
            plot_metrics(
                test_predictions_df=test_predictions_df,
                quantile_df=quantile_df,
                correlations_df=correlations_df,
                plot_path=plot_path,
            )
            plot_written = True
        except ModuleNotFoundError as exc:
            print(f"Plot skipped because optional dependency is missing: {exc}")

    print(f"\nSaved confidence group table to {summary_path}")
    print(f"Saved correlations to {correlations_path}")
    print(f"Saved quantile errors to {quantile_path}")
    print(f"Saved detailed predictions to {detailed_path}")
    print(f"Saved sorted test predictions to {test_predictions_path}")
    if plot_written:
        print(f"Saved plot to {plot_path}")


def plot_metrics(
    test_predictions_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    pearson = correlations_df.loc[
        correlations_df["metric"] == "pearson_confidence_vs_absolute_error",
        "correlation",
    ].iloc[0]
    spearman = correlations_df.loc[
        correlations_df["metric"] == "spearman_confidence_vs_absolute_error",
        "correlation",
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    group_colors = {
        "Top 25% confidence": "#1b9e77",
        "Middle 50% confidence": "#7570b3",
        "Bottom 25% confidence": "#d95f02",
    }
    for group_name in CONFIDENCE_GROUP_ORDER:
        group_df = test_predictions_df[test_predictions_df["confidence_group"] == group_name]
        axes[0].scatter(
            group_df["learned_confidence"],
            group_df["absolute_error"],
            s=18,
            alpha=0.65,
            color=group_colors[group_name],
            label=group_name,
        )
    axes[0].set_title(
        f"Confidence vs Absolute Error\nPearson={pearson:.3f}, Spearman={spearman:.3f}"
    )
    axes[0].set_xlabel("Learned Confidence")
    axes[0].set_ylabel("Absolute Error")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    x = quantile_df["confidence_quantile"].to_numpy(dtype=int)
    axes[1].plot(
        x,
        quantile_df["MAE"],
        marker="o",
        linewidth=1.8,
        label="MAE / Mean Absolute Residual",
    )
    axes[1].plot(
        x,
        quantile_df["RMSE"],
        marker="s",
        linewidth=1.5,
        label="RMSE",
    )
    axes[1].set_title("Error by Confidence Quantile")
    axes[1].set_xlabel("Confidence Quantile (Low to High)")
    axes[1].set_ylabel("Error")
    axes[1].set_xticks(x)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.suptitle("Experiment 6: Confidence Reliability Validation")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def print_summary_table(summary_df: pd.DataFrame, correlations_df: pd.DataFrame) -> None:
    display_columns = [
        "Confidence Group",
        "n_predictions",
        "mean_confidence",
        "MAE",
        "RMSE",
        "Mean Absolute Residual",
    ]
    display_df = summary_df[display_columns].copy()
    for column in display_columns[2:]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "nan"
        )
    print("\nConfidence Reliability Table")
    print(display_df.to_string(index=False))

    corr_df = correlations_df.copy()
    corr_df["correlation"] = corr_df["correlation"].map(
        lambda value: f"{value:.4f}" if np.isfinite(value) else "nan"
    )
    print("\nConfidence vs Absolute Error Correlations")
    print(corr_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    summary_df, correlations_df, quantile_df, detailed_df, test_predictions_df = (
        run_experiment(args)
    )
    save_results(
        summary_df=summary_df,
        correlations_df=correlations_df,
        quantile_df=quantile_df,
        detailed_df=detailed_df,
        test_predictions_df=test_predictions_df,
        results_dir=args.results_dir,
        make_plot=not args.no_plot,
    )
    print_summary_table(summary_df=summary_df, correlations_df=correlations_df)


if __name__ == "__main__":
    main()
