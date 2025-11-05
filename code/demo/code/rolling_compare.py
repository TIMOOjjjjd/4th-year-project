import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# === Overall Summary ===
#   strategy       MAE      RMSE  count
#   baseline 14.271946 27.978876   3885
# confidence 18.289778 32.927296   3885



# === 临时调试：启动时自动清理旧 checkpoint 目录 ===
import shutil
import os
for f in os.listdir('.'):
    if f.startswith("checkpoints_") and os.path.isdir(f):
        try:
            shutil.rmtree(f)
            print(f"[debug] deleted old checkpoint directory: {f}")
        except Exception as e:
            print(f"[debug] failed to delete {f}: {e}")
# === 结束 ===

warnings.simplefilter(action="ignore", category=FutureWarning)

# Managers
from persistent_multiscale_incremental import (
    MultiScaleModelManager as BaseManager,
    ManagerConfig as BaseConfig,
)
from persistent_multiscale_incre_confi import (
    MultiScaleModelManager as ConfiManager,
    ManagerConfig as ConfiConfig,
)


# =====================================================
# User Config
# =====================================================
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
START_TARGET = pd.Timestamp("2021-03-06 12:00")
ROLLING_STEPS = 15
HIDDEN_SIZE = 64
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
OUTPUT_DIR = Path("rolling_compare_results")
CHECKPOINT_ROOT = Path("checkpoints_compare")
CLEAN_CHECKPOINTS = True
RETRAIN_EACH_HOUR = False


def prepare_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("H")
    if EXCLUDED_ZONES:
        df = df[~df["PULocationID"].isin(EXCLUDED_ZONES)]
    return df


def get_true_counts(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def prepare_checkpoint_dir(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_rolling(
    df: pd.DataFrame,
    manager_factory,
    strategy_label: str,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zones = sorted(df["PULocationID"].unique())
    records, metrics = [], []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n[{strategy_label}] ---- Predicting target hour {target_ts} (step {step}) ----")

        if RETRAIN_EACH_HOUR:
            hour_dir = checkpoint_dir / target_ts.strftime("%Y%m%d_%H%M")
            prepare_checkpoint_dir(hour_dir, clean=True)
            manager = manager_factory(str(hour_dir))
        else:
            manager = manager_factory(str(checkpoint_dir))

        y_true_dict = get_true_counts(df, target_ts)
        preds, trues = [], []

        for zid in zones:
            try:
                print(f"-----current zone is {zid}-----")
                pred = manager.train_and_predict_if_needed(df, zid, target_ts, auto_train=True)
                true_val = float(y_true_dict.get(zid, 0.0))
                records.append(
                    {
                        "strategy": strategy_label,
                        "target_hour": target_ts,
                        "PULocationID": zid,
                        "y_pred": pred,
                        "y_true": true_val,
                    }
                )
                preds.append(pred)
                trues.append(true_val)
            except Exception as exc:
                records.append(
                    {
                        "strategy": strategy_label,
                        "target_hour": target_ts,
                        "PULocationID": zid,
                        "y_pred": np.nan,
                        "y_true": np.nan,
                        "error": str(exc),
                    }
                )

        if preds:
            preds_arr, trues_arr = np.array(preds), np.array(trues)
            mask = ~np.isnan(preds_arr) & ~np.isnan(trues_arr)
            mean_true = np.mean(trues_arr[mask]) if mask.any() else np.nan
            if mask.any():
                mae = np.mean(np.abs(preds_arr[mask] - trues_arr[mask]))
                rmse = np.sqrt(np.mean((preds_arr[mask] - trues_arr[mask]) ** 2))
                metrics.append(
                    {
                        "strategy": strategy_label,
                        "target_hour": target_ts,
                        "MAE": mae,
                        "RMSE": rmse,
                        "NMAE": mae / (mean_true + 1e-8),
                        "NRMSE": rmse / (mean_true + 1e-8),
                        "mean_true": mean_true,
                    }
                )
                print(f"[{strategy_label}] MAE={mae:.3f} RMSE={rmse:.3f} mean_true={mean_true:.3f}")
            else:
                metrics.append(
                    {
                        "strategy": strategy_label,
                        "target_hour": target_ts,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "NMAE": np.nan,
                        "NRMSE": np.nan,
                        "mean_true": mean_true,
                    }
                )
        else:
            metrics.append(
                {
                    "strategy": strategy_label,
                    "target_hour": target_ts,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "NMAE": np.nan,
                    "NRMSE": np.nan,
                    "mean_true": np.nan,
                }
            )

    return pd.DataFrame(records), pd.DataFrame(metrics)


def summarize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for strategy, group in pred_df.groupby("strategy"):
        valid = group.dropna(subset=["y_pred", "y_true"])
        if valid.empty:
            summaries.append({"strategy": strategy, "MAE": np.nan, "RMSE": np.nan, "count": 0})
            continue
        y_pred = valid["y_pred"].to_numpy()
        y_true = valid["y_true"].to_numpy()
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        summaries.append({"strategy": strategy, "MAE": mae, "RMSE": rmse, "count": len(valid)})
    return pd.DataFrame(summaries)


def build_manager_factories():
    def base_factory(checkpoint_dir: str):
        cfg = BaseConfig(hidden_size=HIDDEN_SIZE)
        return BaseManager(checkpoint_dir=checkpoint_dir, cfg=cfg)

    def confi_factory(checkpoint_dir: str):
        cfg = ConfiConfig(hidden_size=HIDDEN_SIZE)
        return ConfiManager(checkpoint_dir=checkpoint_dir, cfg=cfg)

    return {
        "baseline": base_factory,
        "confidence": confi_factory,
    }


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    df = prepare_df()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies = build_manager_factories()
    combined_preds, combined_metrics = [], []

    for label, factory in strategies.items():
        strategy_ckpt = CHECKPOINT_ROOT / label
        prepare_checkpoint_dir(strategy_ckpt, clean=CLEAN_CHECKPOINTS)

        # Persistent manager (unless RETRAIN_EACH_HOUR splits)
        persistent_manager = factory(str(strategy_ckpt))

        def manager_factory(parent_dir: str):
            return persistent_manager if not RETRAIN_EACH_HOUR else factory(parent_dir)

        preds_df, metrics_df = run_rolling(df, manager_factory, label, strategy_ckpt)
        preds_df.to_csv(OUTPUT_DIR / f"predictions_{label}.csv", index=False)
        metrics_df.to_csv(OUTPUT_DIR / f"hourly_metrics_{label}.csv", index=False)

        combined_preds.append(preds_df)
        combined_metrics.append(metrics_df)

    all_preds = pd.concat(combined_preds, ignore_index=True)
    all_metrics = pd.concat(combined_metrics, ignore_index=True)
    all_preds.to_csv(OUTPUT_DIR / "predictions_all_strategies.csv", index=False)
    all_metrics.to_csv(OUTPUT_DIR / "hourly_metrics_all_strategies.csv", index=False)

    summary_df = summarize_predictions(all_preds)
    summary_df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    print("\n=== Overall Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
