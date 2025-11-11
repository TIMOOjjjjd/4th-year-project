import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_model import run_gnn_pipeline
from persistent_multiscale_incre_confi import ManagerConfig, MultiScaleModelManager
# from persistent_multiscale_incremental import MultiScaleModelManager, ManagerConfig  # ⚠️ 改成你自己的模块路径

warnings.simplefilter(action="ignore", category=FutureWarning)
# # === 临时调试：启动时自动清理旧 checkpoint 目录 ===
# import shutil
# import os
# for f in os.listdir('.'):
#     if f.startswith("checkpoints_") and os.path.isdir(f):
#         try:
#             shutil.rmtree(f)
#             print(f"[debug] deleted old checkpoint directory: {f}")
#         except Exception as e:
#             print(f"[debug] failed to delete {f}: {e}")
# # === 结束 ===

# =====================================================
# ✅ 用户配置区（直接改这里即可）
# =====================================================
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
EDGE_WEIGHT_MATRIX = "edge_weight_matrix_with_flow.csv"
CHECKPOINT_DIR = "checkpoints_multiscale"

START_TARGET = pd.Timestamp("2021-03-10 00:00")
ROLLING_STEPS = 3
HIDDEN_SIZE = 64
# EXCLUDED_ZONES = [1,2,3,4,5,6,100]
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
RETRAIN_EACH_HOUR = False
MC_DROPOUT_SAMPLES = 10
# =====================================================

HISTORY_WINDOWS = {
    "mean_24h": 24,
    "mean_168h": 24 * 7,
    "mean_720h": 24 * 30,
}
HISTORY_FEATURES = list(HISTORY_WINDOWS.keys())


def _cleanup_old_checkpoints(root: Path) -> None:
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("checkpoints_"):
            try:
                shutil.rmtree(item)
                print(f"[debug] deleted old checkpoint directory: {item}")
            except Exception as exc:  # noqa: BLE001
                print(f"[debug] failed to delete {item}: {exc}")


def prepare_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("H")
    df = df[~df["PULocationID"].isin(EXCLUDED_ZONES)]
    return df


def get_true_counts(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def _build_zone_hourly_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["PULocationID", "datetime"]).size().rename("count")


def _compute_history_means(
    zone_hourly_counts: pd.Series, zone_id: int, target_hour: pd.Timestamp
) -> Dict[str, float]:
    means = {name: 0.0 for name in HISTORY_FEATURES}
    try:
        zone_series = zone_hourly_counts.loc[zone_id]
    except KeyError:
        return means

    for feat_name, hours in HISTORY_WINDOWS.items():
        start = target_hour - pd.Timedelta(hours=hours)
        window = zone_series[(zone_series.index >= start) & (zone_series.index < target_hour)]
        total = float(window.sum()) if not window.empty else 0.0
        means[feat_name] = total / float(hours)
    return means


def _load_zone_lookup(path: str) -> pd.DataFrame:
    df_lookup = pd.read_csv(path)
    return df_lookup.drop_duplicates(subset="LocationID")


def _build_zone_adjacency(edge_csv: str, lookup_df: pd.DataFrame) -> Dict[int, List[int]]:
    """Create adjacency lists keyed by LocationID based on the OD flow graph."""
    df_adj = pd.read_csv(edge_csv, index_col=0)
    df_adj.index = [str(idx).lstrip("\ufeff") for idx in df_adj.index]
    df_adj.columns = [str(col).lstrip("\ufeff") for col in df_adj.columns]

    zone_lookup = lookup_df.drop_duplicates(subset="Zone")
    zone_to_location = dict(zip(zone_lookup["Zone"], zone_lookup["LocationID"]))

    adjacency: Dict[int, List[int]] = {}
    for zone_name in df_adj.index:
        loc_id = zone_to_location.get(zone_name)
        if loc_id is None or pd.isna(loc_id):
            continue

        weights = df_adj.loc[zone_name]
        neighbors: List[int] = []
        for neighbor_zone, weight in weights.items():
            if weight <= 0:
                continue
            neighbor_loc = zone_to_location.get(neighbor_zone)
            if neighbor_loc is None or pd.isna(neighbor_loc):
                continue
            neighbors.append(int(neighbor_loc))

        adjacency[int(loc_id)] = neighbors
    return adjacency


def _compute_prior_scores(df: pd.DataFrame) -> Dict[int, float]:
    """Higher scores for zones with richer history (proxy for low sparsity)."""
    counts = df.groupby("PULocationID").size().astype(float)
    if counts.empty:
        return {}

    log_counts = np.log1p(counts)
    vmin, vmax = log_counts.min(), log_counts.max()
    if np.isclose(vmin, vmax):
        normalized = pd.Series(1.0, index=log_counts.index)
    else:
        normalized = (log_counts - vmin) / (vmax - vmin)

    # Keep scores away from exact 0 to avoid killing gradients later.
    scaled = 0.2 + 0.8 * normalized
    return {int(idx): float(val) for idx, val in scaled.items()}


def _compute_stability_scores(step_df: pd.DataFrame) -> Dict[int, float]:
    """Use MC variance to measure intra-model stability."""
    finite_variance = step_df["variance"].dropna()
    if finite_variance.empty:
        return {int(row.PULocationID): 1.0 for row in step_df.itertuples()}

    scale = float(np.median(finite_variance))
    scale = max(scale, 1e-3)
    scores: Dict[int, float] = {}
    for row in step_df.itertuples():
        zid = int(row.PULocationID)
        variance = float(row.variance)
        if np.isfinite(variance):
            raw = float(np.exp(-variance / (3.0 * scale)))
            scores[zid] = float(np.clip(raw, 0.05, 1.0))
        else:
            scores[zid] = 0.2
    return scores


def _compute_neighborhood_scores(step_df: pd.DataFrame, adjacency: Dict[int, List[int]]) -> Dict[int, float]:
    """Compare each zone prediction to its graph neighbors to measure trend agreement."""
    pred_map = {
        int(row.PULocationID): float(row.y_pred)
        for row in step_df.itertuples()
        if np.isfinite(row.y_pred)
    }
    if not pred_map:
        return {}

    all_preds = np.array(list(pred_map.values()))
    global_scale = float(np.percentile(np.abs(all_preds), 75)) + 1.0

    neighborhood_scores: Dict[int, float] = {}
    for zid, pred in pred_map.items():
        neighbors = adjacency.get(zid, [])
        neighbor_preds = [pred_map[n] for n in neighbors if n in pred_map]
        if not neighbor_preds:
            neighborhood_scores[zid] = 0.6
            continue

        neighbor_array = np.array(neighbor_preds, dtype=np.float32)
        mu = float(neighbor_array.mean())
        sigma = float(neighbor_array.std())
        denom = sigma + 0.1 * abs(mu) + global_scale
        diff = abs(pred - mu)
        raw = float(np.exp(-diff / max(denom, 1e-3)))
        neighborhood_scores[zid] = float(np.clip(raw, 0.05, 1.0))
    return neighborhood_scores


def _combine_confidence_components(
    prior: float, stability: float, neighborhood: float, weights: Dict[str, float]
) -> float:
    prior_c = np.clip(prior, 0.05, 1.0)
    stability_c = np.clip(stability, 0.05, 1.0)
    neighborhood_c = np.clip(neighborhood, 0.05, 1.0)

    log_score = (
        weights["prior"] * np.log(prior_c)
        + weights["stability"] * np.log(stability_c)
        + weights["neighborhood"] * np.log(neighborhood_c)
    )
    return float(np.clip(np.exp(log_score), 0.05, 1.0))


def _assign_confidence_scores(
    step_df: pd.DataFrame,
    prior_scores: Dict[int, float],
    adjacency: Dict[int, List[int]],
) -> Dict[int, float]:
    stability_scores = _compute_stability_scores(step_df)
    neighborhood_scores = _compute_neighborhood_scores(step_df, adjacency)
    weights = {"prior": 0.4, "stability": 0.35, "neighborhood": 0.25}

    zone_confidence: Dict[int, float] = {}
    for row in step_df.itertuples():
        zid = int(row.PULocationID)
        if not np.isfinite(row.y_pred):
            zone_confidence[zid] = 0.0
            continue

        prior = prior_scores.get(zid, 0.4)
        stability = stability_scores.get(zid, 0.5)
        neighborhood = neighborhood_scores.get(zid, 0.6)
        combined = _combine_confidence_components(prior, stability, neighborhood, weights)
        zone_confidence[zid] = combined

    step_df["confidence"] = step_df["PULocationID"].map(
        lambda zid: zone_confidence.get(int(zid), 0.0)
    )
    return zone_confidence


def run_rolling_with_gnn(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    device: torch.device,
    prior_scores: Dict[int, float],
    adjacency: Dict[int, List[int]],
    zone_hourly_counts: pd.Series,
) -> None:
    zones = sorted(df["PULocationID"].unique())

    baseline_records: List[pd.DataFrame] = []
    gnn_records: List[pd.DataFrame] = []
    baseline_metrics: List[Dict[str, float]] = []
    gnn_metrics: List[Dict[str, float]] = []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n/////Predicting target hour: {target_ts} in step {step}/////")

        if RETRAIN_EACH_HOUR:
            hour_dir = Path(CHECKPOINT_DIR) / target_ts.strftime("%Y%m%d_%H%M")
            hour_dir.mkdir(parents=True, exist_ok=True)
            cfg = ManagerConfig(hidden_size=HIDDEN_SIZE, M_mc_test=MC_DROPOUT_SAMPLES)
            mgr = MultiScaleModelManager(checkpoint_dir=str(hour_dir), cfg=cfg)
        else:
            mgr = manager

        y_true_dict = get_true_counts(df, target_ts)
        step_records: List[Dict[str, float]] = []

        for zid in zones:
            try:
                print(f"-----current zone is {zid}-----")
                mgr.train_and_predict_if_needed(df, zid, target_ts, auto_train=True)
                point, std, _ = mgr.predict_with_uncertainty(df, zid, target_ts)
                variance = float(std ** 2)
                true_val = float(y_true_dict.get(zid, 0.0))
                history_means = _compute_history_means(zone_hourly_counts, zid, target_ts)
                step_records.append(
                    {
                        "PULocationID": zid,
                        "y_pred": float(point),
                        "y_true": true_val,
                        "mc_std": float(std),
                        "variance": variance,
                        "error": "",
                        **history_means,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                step_records.append(
                    {
                        "PULocationID": zid,
                        "y_pred": np.nan,
                        "y_true": np.nan,
                        "mc_std": np.nan,
                        "variance": np.nan,
                        "error": str(exc),
                        **{feat: np.nan for feat in HISTORY_FEATURES},
                    }
                )

        step_df = pd.DataFrame(step_records)
        step_df["target_hour"] = target_ts
        zone_confidence = _assign_confidence_scores(step_df, prior_scores, adjacency)
        baseline_records.append(step_df)

        mask = step_df["y_pred"].notna() & step_df["y_true"].notna()
        if mask.any():
            preds = step_df.loc[mask, "y_pred"].values
            trues = step_df.loc[mask, "y_true"].values
            mae = float(np.mean(np.abs(preds - trues)))
            rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
        else:
            mae = float("nan")
            rmse = float("nan")

        baseline_metrics.append(
            {
                "target_hour": target_ts,
                "MAE": mae,
                "RMSE": rmse,
                "mean_true": float(step_df.loc[mask, "y_true"].mean()) if mask.any() else float("nan"),
            }
        )

        rename_map = {"y_pred": "Prediction", "y_true": "True Value"}
        gnn_input_cols = ["PULocationID", "Prediction", "True Value", *HISTORY_FEATURES]
        gnn_input = step_df.rename(columns=rename_map)[gnn_input_cols]

        gnn_output_path = f"final_pred_ms_gnn\final_predictions_multiscale_{target_ts.strftime('%Y%m%d_%H%M')}.csv"
        gnn_output_df, gnn_metric = run_gnn_pipeline(
            df_temp=df,
            target_date=target_ts,
            excluded_zones=EXCLUDED_ZONES,
            device=device,
            merged_csv_path=None,
            zone_total_number=len(zones),
            final_output_csv=gnn_output_path,
            predictions_df=gnn_input,
            zone_confidence=zone_confidence,
            show_plots=False,
        )
        gnn_output_df["target_hour"] = target_ts
        gnn_records.append(gnn_output_df)

        gnn_metrics.append(
            {
                "target_hour": target_ts,
                "MAE_GRU": gnn_metric["mae_gru"],
                "MSE_GRU": gnn_metric["mse_gru"],
                "MAE_GNN": gnn_metric["mae_refined"],
                "MSE_GNN": gnn_metric["mse_refined"],
            }
        )

    baseline_df = pd.concat(baseline_records, ignore_index=True)
    gnn_df = pd.concat(gnn_records, ignore_index=True) if gnn_records else pd.DataFrame()
    baseline_metrics_df = pd.DataFrame(baseline_metrics)
    gnn_metrics_df = pd.DataFrame(gnn_metrics)

    baseline_df.to_csv("predictions_rolling_mc.csv", index=False)
    baseline_metrics_df.to_csv("hourly_metrics_gru.csv", index=False)
    if not gnn_df.empty:
        gnn_df.to_csv("gnn_refined_predictions.csv", index=False)
    gnn_metrics_df.to_csv("hourly_metrics_gnn.csv", index=False)

    valid_baseline = baseline_df.dropna(subset=["y_pred", "y_true"])
    if not valid_baseline.empty:
        overall_mae = float(np.mean(np.abs(valid_baseline["y_pred"] - valid_baseline["y_true"])))
        overall_rmse = float(np.sqrt(np.mean((valid_baseline["y_pred"] - valid_baseline["y_true"]) ** 2)))
    else:
        overall_mae = float("nan")
        overall_rmse = float("nan")

    valid_gnn = gnn_df.dropna(subset=["Refined_Pred", "True_Value"]) if not gnn_df.empty else pd.DataFrame()
    if not valid_gnn.empty:
        overall_mae_gnn = float(
            np.mean(np.abs(valid_gnn["Refined_Pred"] - valid_gnn["True_Value"]))
        )
        overall_rmse_gnn = float(
            np.sqrt(np.mean((valid_gnn["Refined_Pred"] - valid_gnn["True_Value"]) ** 2))
        )
    else:
        overall_mae_gnn = float("nan")
        overall_rmse_gnn = float("nan")

    with open("overall_metrics_gnn.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Baseline MAE: {overall_mae}\n")
        fh.write(f"Baseline RMSE: {overall_rmse}\n")
        fh.write(f"GNN MAE: {overall_mae_gnn}\n")
        fh.write(f"GNN RMSE: {overall_rmse_gnn}\n")

    print(f"\n🎯 Baseline MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    print(f"🎯 GNN MAE={overall_mae_gnn:.4f}, RMSE={overall_rmse_gnn:.4f}")


def main() -> None:
    _cleanup_old_checkpoints(Path("."))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    df = prepare_df()
    prior_scores = _compute_prior_scores(df)
    lookup_df = _load_zone_lookup(LOOKUP_PATH)
    adjacency = _build_zone_adjacency(EDGE_WEIGHT_MATRIX, lookup_df)
    zone_hourly_counts = _build_zone_hourly_counts(df)
    cfg = ManagerConfig(hidden_size=HIDDEN_SIZE, M_mc_test=MC_DROPOUT_SAMPLES)
    manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)

    run_rolling_with_gnn(df, manager, device, prior_scores, adjacency, zone_hourly_counts)


if __name__ == "__main__":
    main()
