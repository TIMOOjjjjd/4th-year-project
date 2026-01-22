import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List
from scipy.special import erf

import numpy as np
import pandas as pd
import torch

from gnn_model import run_gnn_pipeline
# from persistent_multiscale_incre_confi import ManagerConfig, MultiScaleModelManager
from persistent_multiscale_confi import MultiScaleModelManager, ManagerConfig

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

START_TARGET = pd.Timestamp("2021-03-05 00:00")
ROLLING_STEPS = 1
# EXCLUDED_ZONES = [1,2,3,4,5,6,100]
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
RETRAIN_EACH_HOUR = False
MC_DROPOUT_SAMPLES = 10
PICP_ALPHA = 0.1
PICP_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4]
# =====================================================

HISTORY_WINDOWS = {
    "mean_24h": 24,
    "mean_168h": 24 * 7,
    "mean_720h": 24 * 15,
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


def _compute_confidence_calibration(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    required_cols = ["confidence", "y_pred", "y_true"]
    valid = df.dropna(subset=required_cols).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["confidence"] = valid["confidence"].clip(0.0, 1.0)
    valid["abs_error"] = (valid["y_true"] - valid["y_pred"]).abs()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    valid["conf_bin"] = pd.cut(valid["confidence"], bins=bins, include_lowest=True, right=True)

    rows = []
    for interval, group in valid.groupby("conf_bin", observed=True):
        if group.empty:
            continue
        rmse = float(np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2)))
        coverage = float("nan")
        if {"pi_lower", "pi_upper"}.issubset(group.columns):
            cov_mask = group[["pi_lower", "pi_upper", "y_true"]].notna().all(axis=1)
            if cov_mask.any():
                lower = group.loc[cov_mask, "pi_lower"].to_numpy()
                upper = group.loc[cov_mask, "pi_upper"].to_numpy()
                true_vals = group.loc[cov_mask, "y_true"].to_numpy()
                coverage = float(np.mean((true_vals >= lower) & (true_vals <= upper)))
        rows.append(
            {
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "count": int(len(group)),
                "mean_confidence": float(group["confidence"].mean()),
                "mae": float(group["abs_error"].mean()),
                "rmse": rmse,
                "coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def _confidence_error_correlation(df: pd.DataFrame) -> float:
    required_cols = ["confidence", "y_pred", "y_true"]
    valid = df.dropna(subset=required_cols).copy()
    if len(valid) < 2:
        return float("nan")

    valid["confidence"] = valid["confidence"].clip(0.0, 1.0)
    abs_error = (valid["y_true"] - valid["y_pred"]).abs().to_numpy()
    conf = valid["confidence"].to_numpy()
    if np.std(conf) == 0.0 or np.std(abs_error) == 0.0:
        return float("nan")
    return float(np.corrcoef(conf, abs_error)[0, 1])


def _gaussian_nll_crps(
    y_true: np.ndarray, y_pred: np.ndarray, std: np.ndarray, eps: float = 1e-6
) -> Dict[str, float]:
    sigma = np.maximum(std, eps)
    z = (y_true - y_pred) / sigma
    nll = 0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * z**2

    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))
    return {"nll": float(np.mean(nll)), "crps": float(np.mean(crps))}


def _compute_picp_curve(
    y_true: np.ndarray, y_pred: np.ndarray, alphas: List[float]
) -> pd.DataFrame:
    if y_true.size == 0 or y_pred.size == 0:
        return pd.DataFrame()

    residuals = y_true - y_pred
    rows = []
    for alpha in alphas:
        if not 0.0 < alpha < 1.0:
            continue
        q_low, q_high = np.quantile(residuals, [alpha / 2.0, 1.0 - alpha / 2.0])
        lower = y_pred + q_low
        upper = y_pred + q_high
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        rows.append(
            {
                "alpha": float(alpha),
                "expected_coverage": float(1.0 - alpha),
                "observed_coverage": coverage,
            }
        )
    return pd.DataFrame(rows)


def _ece_ace_from_curve(curve_df: pd.DataFrame) -> Dict[str, float]:
    if curve_df.empty:
        return {"ece": float("nan"), "ace": float("nan")}
    gaps = np.abs(curve_df["observed_coverage"] - curve_df["expected_coverage"])
    ace = float(np.mean(gaps))
    ece = float(np.mean(gaps))
    return {"ece": ece, "ace": ace}


def _sigma_from_quantiles(q_low: np.ndarray, q_high: np.ndarray, alpha: float) -> np.ndarray:
    """Assume central (1-alpha) Normal interval to approximate sigma.
    sigma ≈ (q_high - q_low) / (2 * z_{1-alpha/2}).
    """
    try:
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        z = 1.6448536269514722  # fallback for alpha=0.1 (~90% interval)
    width = np.maximum(q_high - q_low, 0.0)
    denom = max(2.0 * z, 1e-6)
    sigma = width / denom
    return np.maximum(sigma, 1e-6)


def _is_monotonic_nonincreasing(values: np.ndarray) -> bool:
    if values.size < 2:
        return True
    diffs = np.diff(values)
    return bool(np.all(diffs <= 1e-12))


def _confidence_bin_monotonicity(
    calibration_df: pd.DataFrame,
) -> Dict[str, object]:
    if calibration_df.empty:
        return {"mae_nonincreasing": float("nan"), "coverage_nonincreasing": float("nan")}

    ordered = calibration_df.sort_values("bin_left").reset_index(drop=True)
    mae_vals = ordered["mae"].dropna().to_numpy()
    coverage_vals = ordered["coverage"].dropna().to_numpy()

    mae_monotonic = _is_monotonic_nonincreasing(mae_vals)
    if coverage_vals.size:
        coverage_monotonic = _is_monotonic_nonincreasing(coverage_vals)
    else:
        coverage_monotonic = float("nan")
    return {
        "mae_nonincreasing": mae_monotonic,
        "coverage_nonincreasing": coverage_monotonic,
    }


def run_rolling_with_gnn(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    device: torch.device,
    prior_scores: Dict[int, float],
    adjacency: Dict[int, List[int]],
    zone_hourly_counts: pd.Series,
) -> None:
    zones = sorted(df["PULocationID"].unique())

    zones = zones[:20]

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
        step_df["pi_lower"] = np.nan
        step_df["pi_upper"] = np.nan
        zone_confidence = _assign_confidence_scores(step_df, prior_scores, adjacency)
        baseline_records.append(step_df)

        mask = step_df["y_pred"].notna() & step_df["y_true"].notna()
        if mask.any():
            preds = step_df.loc[mask, "y_pred"].values
            trues = step_df.loc[mask, "y_true"].values
            stds = step_df.loc[mask, "mc_std"].values
            residuals = trues - preds
            q_low, q_high = np.quantile(
                residuals, [PICP_ALPHA / 2.0, 1.0 - PICP_ALPHA / 2.0]
            )
            step_df.loc[mask, "pi_lower"] = step_df.loc[mask, "y_pred"] + q_low
            step_df.loc[mask, "pi_upper"] = step_df.loc[mask, "y_pred"] + q_high
            mae = float(np.mean(np.abs(preds - trues)))
            rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
            lower = step_df.loc[mask, "pi_lower"].values
            upper = step_df.loc[mask, "pi_upper"].values
            picp = float(np.mean((trues >= lower) & (trues <= upper)))
            valid_std = np.isfinite(stds) & (stds > 0.0)
            if np.any(valid_std):
                nll_crps = _gaussian_nll_crps(trues[valid_std], preds[valid_std], stds[valid_std])
                nll = nll_crps["nll"]
                crps = nll_crps["crps"]
            else:
                nll = float("nan")
                crps = float("nan")
        else:
            mae = float("nan")
            rmse = float("nan")
            picp = float("nan")
            nll = float("nan")
            crps = float("nan")

        baseline_metrics.append(
            {
                "target_hour": target_ts,
                "MAE": mae,
                "RMSE": rmse,
                "mean_true": float(step_df.loc[mask, "y_true"].mean()) if mask.any() else float("nan"),
                "PICP": picp,
                "NLL": nll,
                "CRPS": crps,
            }
        )

        rename_map = {"y_pred": "Prediction", "y_true": "True Value"}
        gnn_input_cols = ["PULocationID", "Prediction", "True Value", *HISTORY_FEATURES]
        gnn_input = step_df.rename(columns=rename_map)[gnn_input_cols]

        gnn_output_path = f"final_predictions_multiscale_{target_ts.strftime('%Y%m%d_%H%M')}.csv"
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
        gnn_output_df["gru_pi_lower"] = np.nan
        gnn_output_df["gru_pi_upper"] = np.nan
        gnn_output_df["gnn_pi_lower"] = np.nan
        gnn_output_df["gnn_pi_upper"] = np.nan
        gnn_records.append(gnn_output_df)

        gnn_mask = gnn_output_df[["True_Value", "GRU_Pred", "Refined_Pred"]].notna().all(axis=1)
        if gnn_mask.any():
            true_vals = gnn_output_df.loc[gnn_mask, "True_Value"].values
            gru_residuals = true_vals - gnn_output_df.loc[gnn_mask, "GRU_Pred"].values
            gnn_residuals = true_vals - gnn_output_df.loc[gnn_mask, "Refined_Pred"].values
            gru_q_low, gru_q_high = np.quantile(
                gru_residuals, [PICP_ALPHA / 2.0, 1.0 - PICP_ALPHA / 2.0]
            )
            gnn_q_low, gnn_q_high = np.quantile(
                gnn_residuals, [PICP_ALPHA / 2.0, 1.0 - PICP_ALPHA / 2.0]
            )
            gnn_output_df.loc[gnn_mask, "gru_pi_lower"] = (
                gnn_output_df.loc[gnn_mask, "GRU_Pred"] + gru_q_low
            )
            gnn_output_df.loc[gnn_mask, "gru_pi_upper"] = (
                gnn_output_df.loc[gnn_mask, "GRU_Pred"] + gru_q_high
            )
            gnn_output_df.loc[gnn_mask, "gnn_pi_lower"] = (
                gnn_output_df.loc[gnn_mask, "Refined_Pred"] + gnn_q_low
            )
            gnn_output_df.loc[gnn_mask, "gnn_pi_upper"] = (
                gnn_output_df.loc[gnn_mask, "Refined_Pred"] + gnn_q_high
            )
            gru_lower = gnn_output_df.loc[gnn_mask, "gru_pi_lower"].values
            gru_upper = gnn_output_df.loc[gnn_mask, "gru_pi_upper"].values
            gnn_lower = gnn_output_df.loc[gnn_mask, "gnn_pi_lower"].values
            gnn_upper = gnn_output_df.loc[gnn_mask, "gnn_pi_upper"].values
            picp_gru = float(np.mean((true_vals >= gru_lower) & (true_vals <= gru_upper)))
            picp_gnn = float(np.mean((true_vals >= gnn_lower) & (true_vals <= gnn_upper)))
        else:
            picp_gru = float("nan")
            picp_gnn = float("nan")

        gnn_metrics.append(
            {
                "target_hour": target_ts,
                "MAE_GRU": gnn_metric["mae_gru"],
                "MSE_GRU": gnn_metric["mse_gru"],
                "MAE_GNN": gnn_metric["mae_refined"],
                "MSE_GNN": gnn_metric["mse_refined"],
                "PICP_GRU": picp_gru,
                "PICP_GNN": picp_gnn,
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

    calibration_df = _compute_confidence_calibration(baseline_df, n_bins=5)
    confidence_corr = _confidence_error_correlation(baseline_df)
    monotonicity = _confidence_bin_monotonicity(calibration_df)
    if not calibration_df.empty:
        calibration_df.to_csv("confidence_calibration.csv", index=False)
        with open("confidence_calibration.txt", "w", encoding="utf-8") as fh:
            fh.write(f"Confidence vs abs error correlation: {confidence_corr}\n")
            fh.write(
                "Binned monotonicity (non-increasing with confidence): "
                f"mae={monotonicity['mae_nonincreasing']}, "
                f"coverage={monotonicity['coverage_nonincreasing']}\n"
            )
    print(f"📊 Confidence vs abs error correlation: {confidence_corr}")
    if not calibration_df.empty:
        print(
            "📊 Binned monotonicity (non-increasing with confidence): "
            f"mae={monotonicity['mae_nonincreasing']}, "
            f"coverage={monotonicity['coverage_nonincreasing']}"
        )

    valid_baseline = baseline_df.dropna(subset=["y_pred", "y_true"])
    if not valid_baseline.empty:
        overall_mae = float(np.mean(np.abs(valid_baseline["y_pred"] - valid_baseline["y_true"])))
        overall_rmse = float(np.sqrt(np.mean((valid_baseline["y_pred"] - valid_baseline["y_true"]) ** 2)))
        overall_picp = float(
            np.mean(
                (valid_baseline["y_true"] >= valid_baseline["pi_lower"])
                & (valid_baseline["y_true"] <= valid_baseline["pi_upper"])
            )
        )
        valid_std = valid_baseline["mc_std"].to_numpy()
        valid_std_mask = np.isfinite(valid_std) & (valid_std > 0.0)
        if np.any(valid_std_mask):
            nll_crps = _gaussian_nll_crps(
                valid_baseline.loc[valid_std_mask, "y_true"].to_numpy(),
                valid_baseline.loc[valid_std_mask, "y_pred"].to_numpy(),
                valid_std[valid_std_mask],
            )
            overall_nll = nll_crps["nll"]
            overall_crps = nll_crps["crps"]
        else:
            overall_nll = float("nan")
            overall_crps = float("nan")
    else:
        overall_mae = float("nan")
        overall_rmse = float("nan")
        overall_picp = float("nan")
        overall_nll = float("nan")
        overall_crps = float("nan")

    baseline_curve = _compute_picp_curve(
        valid_baseline["y_true"].to_numpy(),
        valid_baseline["y_pred"].to_numpy(),
        PICP_ALPHAS,
    )
    baseline_cal = _ece_ace_from_curve(baseline_curve)
    if not baseline_curve.empty:
        baseline_curve.to_csv("picp_curve_baseline.csv", index=False)

    valid_gnn = gnn_df.dropna(subset=["Refined_Pred", "True_Value"]) if not gnn_df.empty else pd.DataFrame()
    if not valid_gnn.empty:
        overall_mae_gnn = float(
            np.mean(np.abs(valid_gnn["Refined_Pred"] - valid_gnn["True_Value"]))
        )
        overall_rmse_gnn = float(
            np.sqrt(np.mean((valid_gnn["Refined_Pred"] - valid_gnn["True_Value"]) ** 2))
        )
        overall_picp_gru = float(
            np.mean(
                (valid_gnn["True_Value"] >= valid_gnn["gru_pi_lower"])
                & (valid_gnn["True_Value"] <= valid_gnn["gru_pi_upper"])
            )
        )
        overall_picp_gnn = float(
            np.mean(
                (valid_gnn["True_Value"] >= valid_gnn["gnn_pi_lower"])
                & (valid_gnn["True_Value"] <= valid_gnn["gnn_pi_upper"])
            )
        )
        # GNN(MC-approx) 概率指标：用GNN区间反推sigma并计算NLL/CRPS（不覆盖原有结果）
        gmask = valid_gnn[["Refined_Pred", "True_Value", "gnn_pi_lower", "gnn_pi_upper"]].notna().all(axis=1)
        if gmask.any():
            mu_g = valid_gnn.loc[gmask, "Refined_Pred"].to_numpy(dtype=float)
            y_g = valid_gnn.loc[gmask, "True_Value"].to_numpy(dtype=float)
            ql_g = valid_gnn.loc[gmask, "gnn_pi_lower"].to_numpy(dtype=float)
            qh_g = valid_gnn.loc[gmask, "gnn_pi_upper"].to_numpy(dtype=float)
            sigma_g = _sigma_from_quantiles(ql_g, qh_g, PICP_ALPHA)
            gnn_mc_res = _gaussian_nll_crps(y_g, mu_g, sigma_g)
            overall_nll_gnn_mc = float(gnn_mc_res["nll"])
            overall_crps_gnn_mc = float(gnn_mc_res["crps"])
        else:
            overall_nll_gnn_mc = float("nan")
            overall_crps_gnn_mc = float("nan")
    else:
        overall_mae_gnn = float("nan")
        overall_rmse_gnn = float("nan")
        overall_picp_gru = float("nan")
        overall_picp_gnn = float("nan")
        overall_nll_gnn_mc = float("nan")
        overall_crps_gnn_mc = float("nan")

    gru_curve = pd.DataFrame()
    gnn_curve = pd.DataFrame()
    gru_cal = {"ece": float("nan"), "ace": float("nan")}
    gnn_cal = {"ece": float("nan"), "ace": float("nan")}
    if not valid_gnn.empty:
        gmask_gru = valid_gnn[["GRU_Pred", "True_Value"]].notna().all(axis=1)
        if gmask_gru.any():
            gru_curve = _compute_picp_curve(
                valid_gnn.loc[gmask_gru, "True_Value"].to_numpy(),
                valid_gnn.loc[gmask_gru, "GRU_Pred"].to_numpy(),
                PICP_ALPHAS,
            )
            gru_cal = _ece_ace_from_curve(gru_curve)
            if not gru_curve.empty:
                gru_curve.to_csv("picp_curve_gru.csv", index=False)
        gmask_gnn = valid_gnn[["Refined_Pred", "True_Value"]].notna().all(axis=1)
        if gmask_gnn.any():
            gnn_curve = _compute_picp_curve(
                valid_gnn.loc[gmask_gnn, "True_Value"].to_numpy(),
                valid_gnn.loc[gmask_gnn, "Refined_Pred"].to_numpy(),
                PICP_ALPHAS,
            )
            gnn_cal = _ece_ace_from_curve(gnn_curve)
            if not gnn_curve.empty:
                gnn_curve.to_csv("picp_curve_gnn.csv", index=False)

    with open("overall_metrics_gnn.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Baseline MAE: {overall_mae}\n")
        fh.write(f"Baseline RMSE: {overall_rmse}\n")
        fh.write(f"Baseline PICP@{1.0 - PICP_ALPHA:.2f}: {overall_picp}\n")
        fh.write(f"Baseline NLL: {overall_nll}\n")
        fh.write(f"Baseline CRPS: {overall_crps}\n")
        fh.write(f"Baseline curve ECE: {baseline_cal['ece']}\n")
        fh.write(f"Baseline curve ACE: {baseline_cal['ace']}\n")
        fh.write(f"GNN MAE: {overall_mae_gnn}\n")
        fh.write(f"GNN RMSE: {overall_rmse_gnn}\n")
        fh.write(f"GRU PICP@{1.0 - PICP_ALPHA:.2f}: {overall_picp_gru}\n")
        fh.write(f"GNN PICP@{1.0 - PICP_ALPHA:.2f}: {overall_picp_gnn}\n")
        fh.write(f"GNN(MC-approx) NLL: {overall_nll_gnn_mc}\n")
        fh.write(f"GNN(MC-approx) CRPS: {overall_crps_gnn_mc}\n")
        fh.write(f"GRU curve ECE: {gru_cal['ece']}\n")
        fh.write(f"GRU curve ACE: {gru_cal['ace']}\n")
        fh.write(f"GNN curve ECE: {gnn_cal['ece']}\n")
        fh.write(f"GNN curve ACE: {gnn_cal['ace']}\n")

    print(f"\n🎯 Baseline MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")
    print(f"🎯 Baseline PICP@{1.0 - PICP_ALPHA:.2f}={overall_picp:.4f}")
    print(f"🎯 Baseline NLL={overall_nll:.4f}, CRPS={overall_crps:.4f}")
    print(f"🎯 Baseline curve ECE={baseline_cal['ece']:.4f}, ACE={baseline_cal['ace']:.4f}")
    print(f"🎯 GNN MAE={overall_mae_gnn:.4f}, RMSE={overall_rmse_gnn:.4f}")
    print(f"🎯 GRU PICP@{1.0 - PICP_ALPHA:.2f}={overall_picp_gru:.4f}")
    print(f"🎯 GNN PICP@{1.0 - PICP_ALPHA:.2f}={overall_picp_gnn:.4f}")
    print(f"🎯 GNN(MC-approx) NLL={overall_nll_gnn_mc:.4f}, CRPS={overall_crps_gnn_mc:.4f}")
    print(f"🎯 GRU curve ECE={gru_cal['ece']:.4f}, ACE={gru_cal['ace']:.4f}")
    print(f"🎯 GNN curve ECE={gnn_cal['ece']:.4f}, ACE={gnn_cal['ace']:.4f}")


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
    cfg = ManagerConfig(M_mc_test=MC_DROPOUT_SAMPLES)
    manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)

    run_rolling_with_gnn(df, manager, device, prior_scores, adjacency, zone_hourly_counts)


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
