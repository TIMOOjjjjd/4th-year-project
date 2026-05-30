import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_model import run_gnn_pipeline

MODEL_BACKEND = "multiscale"  # "lstm", "gru", "transformer", or "multiscale"

if MODEL_BACKEND == "lstm":
    from persistent_lstm import ManagerConfig, PureLSTMModelManager as MultiScaleModelManager
elif MODEL_BACKEND == "gru":
    from persistent_gru import ManagerConfig, PureGRUModelManager as MultiScaleModelManager
elif MODEL_BACKEND == "transformer":
    from persistent_transformer import ManagerConfig, PureTransformerModelManager as MultiScaleModelManager
elif MODEL_BACKEND == "multiscale":
    from persistent_multiscale_incre_confi import ManagerConfig, MultiScaleModelManager
else:
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")



warnings.simplefilter(action="ignore", category=FutureWarning)
# # === 临时调试：启动时自动清理旧 checkpoint 目录 ===
import shutil
import os
for f in os.listdir('.'):
    if f.startswith("checkpoints_") and os.path.isdir(f):
        try:
            shutil.rmtree(f)
            print(f"[debug] deleted old checkpoint directory: {f}")
        except Exception as e:
            print(f"[debug] failed to delete {f}: {e}")
# # === 结束 ===

# =====================================================
# ✅ 用户配置区（直接改这里即可）
# =====================================================
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
EDGE_WEIGHT_MATRIX = "edge_weight_matrix_od.csv"
CHECKPOINT_DIR = f"checkpoints_{MODEL_BACKEND}"

START_TARGET = pd.Timestamp("2021-03-05 00:00")
ROLLING_STEPS = 24
# EXCLUDED_ZONES = [1,2,3,4,5,6,100]
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
RETRAIN_EACH_HOUR = False
CLEAN_CHECKPOINTS_ON_START = False
MC_DROPOUT_SAMPLES = 10
# =====================================================

HISTORY_WINDOWS = {
    "mean_24h": 24,
    "mean_168h": 24 * 7,
    "mean_720h": 24 * 30,
}
HISTORY_FEATURES = list(HISTORY_WINDOWS.keys())


def _cleanup_old_checkpoints(root: Path, checkpoint_dir: str) -> None:
    target_name = Path(checkpoint_dir).name
    for item in root.iterdir():
        if item.is_dir() and item.name == target_name:
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
    print("Earliest timestamp:", df["datetime"].min())
    print("Latest timestamp:", df["datetime"].max())
    print("Total hours:", df["datetime"].nunique())
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
    weights = {"prior": 0.3, "stability": 0.4, "neighborhood": 0.3}

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


def _build_gnn_input_frame(step_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "y_pred": "Prediction",
        "y_true": "True Value",
        "confidence": "Confidence",
    }
    gnn_frame = step_df.rename(columns=rename_map)
    cols = [
        "target_hour",
        "PULocationID",
        "Prediction",
        "True Value",
        "Confidence",
        *HISTORY_FEATURES,
    ]
    return gnn_frame[[col for col in cols if col in gnn_frame.columns]]


def run_rolling_with_gnn(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    device: torch.device,
    adjacency: Dict[int, List[int]],
    zone_hourly_counts: pd.Series,
) -> None:
    zones = sorted(df["PULocationID"].unique())

    baseline_records: List[pd.DataFrame] = []
    gnn_records: List[pd.DataFrame] = []
    baseline_metrics: List[Dict[str, float]] = [] #{ "target_hour": <Timestamp>, "MAE": <float>, "RMSE": <float>, "mean_true": <float> },
    gnn_metrics: List[Dict[str, float]] = []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n/////Predicting target hour: {target_ts} in step {step}/////")

        if RETRAIN_EACH_HOUR:
            hour_dir = Path(CHECKPOINT_DIR) / target_ts.strftime("%Y%m%d_%H%M")
            hour_dir.mkdir(parents=True, exist_ok=True)
            cfg = ManagerConfig(M_mc_test=MC_DROPOUT_SAMPLES)
            mgr = MultiScaleModelManager(checkpoint_dir=str(hour_dir), cfg=cfg)
        else:
            mgr = manager

        y_true_dict = get_true_counts(df, target_ts) #{10: 3, 11: 0, 12: 5, ...}
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
                # Diagnostics: print exception type and message for this zone/target
                try:
                    print(f"[diag] zone={zid} target={target_ts} error={type(exc).__name__}: {exc}")
                except Exception:
                    pass
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
        history_df = df[df["datetime"] < target_ts]
        hour_prior_scores = _compute_prior_scores(history_df)
        zone_confidence = _assign_confidence_scores(step_df, hour_prior_scores, adjacency)

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

        gnn_input = _build_gnn_input_frame(step_df)
        if baseline_records:
            gnn_train_input = pd.concat(
                [_build_gnn_input_frame(prev_df) for prev_df in baseline_records],
                ignore_index=True,
            )
        else:
            gnn_train_input = pd.DataFrame(columns=gnn_input.columns)

        gnn_output_path = (
            f"final_predictions_{MODEL_BACKEND}_{target_ts.strftime('%Y%m%d_%H%M')}.csv"
        )
        gnn_output_df, gnn_metric = run_gnn_pipeline(
            df_temp=df,
            target_date=target_ts,
            excluded_zones=EXCLUDED_ZONES,
            device=device,
            merged_csv_path=None,
            zone_total_number=len(zones),
            final_output_csv=gnn_output_path,
            predictions_df=gnn_input,
            train_predictions_df=gnn_train_input,
            zone_confidence=zone_confidence,
            show_plots=False,
        )
        gnn_output_df["target_hour"] = target_ts
        gnn_records.append(gnn_output_df)
        baseline_records.append(step_df)

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

    baseline_df.to_csv(f"predictions_rolling_{MODEL_BACKEND}.csv", index=False)
    baseline_metrics_df.to_csv(f"hourly_metrics_{MODEL_BACKEND}.csv", index=False)
    if not gnn_df.empty:
        gnn_df.to_csv(f"gnn_refined_predictions_{MODEL_BACKEND}.csv", index=False)
    gnn_metrics_df.to_csv(f"hourly_metrics_gnn_{MODEL_BACKEND}.csv", index=False)

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

    with open(f"overall_metrics_{MODEL_BACKEND}_gnn.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Baseline MAE: {overall_mae}\n")
        fh.write(f"Baseline RMSE: {overall_rmse}\n")
        fh.write(f"GNN MAE: {overall_mae_gnn}\n")
        fh.write(f"GNN RMSE: {overall_rmse_gnn}\n")

    print(f"\n🎯 {MODEL_BACKEND.upper()} baseline MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    print(f"🎯 GNN MAE={overall_mae_gnn:.4f}, RMSE={overall_rmse_gnn:.4f}")


def main() -> None:
    if CLEAN_CHECKPOINTS_ON_START:
        _cleanup_old_checkpoints(Path("."), CHECKPOINT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    df = prepare_df()
    lookup_df = _load_zone_lookup(LOOKUP_PATH)
    adjacency = _build_zone_adjacency(EDGE_WEIGHT_MATRIX, lookup_df)
    zone_hourly_counts = _build_zone_hourly_counts(df)
    cfg = ManagerConfig(M_mc_test=MC_DROPOUT_SAMPLES)
    manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)

    run_rolling_with_gnn(df, manager, device, adjacency, zone_hourly_counts)


if __name__ == "__main__":
    main()




# parpare_df() 负责导入dataframe，并进行预处理，包括提取pickup location 和 pickuptime，去除没有数据的taxizone
# _compute_prior_scores(df)负责统计每一个taxizone的历史数据量并进行归一化，作为GraphSAGE confidence的其中一个参考
# _load_zone_lookup()负责对应zonename和zoneid
# _build_zone_adjacency 构建一个 以 LocationID 为 key 的邻接表（adjacency list），用于GraphSAGE GNN。 使用edgeweight matrix（如果 OD 流量为 0 → 不视为邻居
# 如果 >0 → 视为有边）
# _build_zone_hourly_counts 按 Taxi Zone（PULocationID） + 小时（datetime） 分组统计订单数量，（每个区域每小时有多少订单？）返回 pandas series
# cfg定义RNN参数
# 配置 cfg和 MultiScaleModelManager（chekpointpath 和 cfg）
#启动run_rolling_with_gnn（df, manager, device, prior_scores, adjacency, zone_hourly_counts）
