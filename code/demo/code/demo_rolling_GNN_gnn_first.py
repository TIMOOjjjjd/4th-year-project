import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from gnn_model_gnn_first import train_spatial_gnn_embeddings
from persistent_multiscale_confi_gnn_first import MultiScaleModelManager, ManagerConfig

warnings.simplefilter(action="ignore", category=FutureWarning)

# =====================================================
# ✅ 用户配置区（直接改这里即可）
# =====================================================
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
EDGE_WEIGHT_MATRIX = "edge_weight_matrix_with_flow.csv"
CHECKPOINT_DIR = "checkpoints_multiscale_gnn_first"

START_TARGET = pd.Timestamp("2021-03-05 00:00")
ROLLING_STEPS = 1
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
RETRAIN_EACH_HOUR = False
MC_DROPOUT_SAMPLES = 10
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


def run_rolling_with_gnn_first(
    df: pd.DataFrame,
    manager: MultiScaleModelManager,
    device: torch.device,
    spatial_embeddings: Dict[int, np.ndarray],
    zone_hourly_counts: pd.Series,
) -> None:
    zones = sorted(df["PULocationID"].unique())

    baseline_records: List[pd.DataFrame] = []
    baseline_metrics: List[Dict[str, float]] = []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n/////Predicting target hour: {target_ts} in step {step}/////")

        if RETRAIN_EACH_HOUR:
            hour_dir = Path(CHECKPOINT_DIR) / target_ts.strftime("%Y%m%d_%H%M")
            hour_dir.mkdir(parents=True, exist_ok=True)
            cfg = ManagerConfig(M_mc_test=MC_DROPOUT_SAMPLES, spatial_dim=manager.cfg.spatial_dim)
            mgr = MultiScaleModelManager(checkpoint_dir=str(hour_dir), cfg=cfg)
        else:
            mgr = manager

        y_true_dict = get_true_counts(df, target_ts)
        step_records: List[Dict[str, float]] = []

        for zid in zones:
            try:
                print(f"-----current zone is {zid}-----")
                spatial_feat = spatial_embeddings.get(int(zid))
                if spatial_feat is None:
                    spatial_feat = np.zeros(manager.cfg.spatial_dim, dtype=np.float32)
                mgr.train_and_predict_if_needed(
                    df, zid, target_ts, auto_train=True, spatial_feat=spatial_feat
                )
                point, std, _ = mgr.predict_with_uncertainty(
                    df, zid, target_ts, spatial_feat=spatial_feat
                )
                variance = float(std**2)
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

    baseline_df = pd.concat(baseline_records, ignore_index=True)
    baseline_metrics_df = pd.DataFrame(baseline_metrics)

    baseline_df.to_csv("predictions_rolling_mc_gnn_first.csv", index=False)
    baseline_metrics_df.to_csv("hourly_metrics_gru_gnn_first.csv", index=False)

    valid_baseline = baseline_df.dropna(subset=["y_pred", "y_true"])
    if not valid_baseline.empty:
        overall_mae = float(np.mean(np.abs(valid_baseline["y_pred"] - valid_baseline["y_true"])))
        overall_rmse = float(
            np.sqrt(np.mean((valid_baseline["y_pred"] - valid_baseline["y_true"]) ** 2))
        )
    else:
        overall_mae = float("nan")
        overall_rmse = float("nan")

    with open("overall_metrics_gnn_first.txt", "w", encoding="utf-8") as fh:
        fh.write(f"GNN-first MAE: {overall_mae}\n")
        fh.write(f"GNN-first RMSE: {overall_rmse}\n")

    print(f"\n🎯 GNN-first MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")


def main() -> None:
    _cleanup_old_checkpoints(Path("."))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    df = prepare_df()
    zone_hourly_counts = _build_zone_hourly_counts(df)

    gnn_ref_ts = START_TARGET - pd.Timedelta(hours=1)
    spatial_embeddings = train_spatial_gnn_embeddings(
        df=df,
        target_date=gnn_ref_ts,
        excluded_zones=EXCLUDED_ZONES,
        device=device,
        edge_weight_csv=EDGE_WEIGHT_MATRIX,
        taxi_zone_lookup=LOOKUP_PATH,
    )
    if not spatial_embeddings:
        raise ValueError("No spatial embeddings were generated; check input data and graph files.")

    spatial_dim = len(next(iter(spatial_embeddings.values())))
    cfg = ManagerConfig(M_mc_test=MC_DROPOUT_SAMPLES, spatial_dim=spatial_dim)
    manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)

    run_rolling_with_gnn_first(df, manager, device, spatial_embeddings, zone_hourly_counts)


if __name__ == "__main__":
    main()
