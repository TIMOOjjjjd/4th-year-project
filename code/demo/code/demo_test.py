"""Rolling 24-hour baseline test without uncertainty or GNN refinement."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


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


MODEL_BACKEND = "multiscale"  # "lstm", "gru", "transformer", or "multiscale"

if MODEL_BACKEND == "lstm":
    from persistent_lstm import ManagerConfig, PureLSTMModelManager as ModelManager
elif MODEL_BACKEND == "gru":
    from persistent_gru import ManagerConfig, PureGRUModelManager as ModelManager
elif MODEL_BACKEND == "transformer":
    from persistent_transformer import ManagerConfig, PureTransformerModelManager as ModelManager
elif MODEL_BACKEND == "multiscale":
    from persistent_multiscale_incre_confi import ManagerConfig, MultiScaleModelManager as ModelManager
else:
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")


warnings.simplefilter(action="ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.parquet"
CHECKPOINT_DIR = BASE_DIR / f"checkpoints_{MODEL_BACKEND}_test"
OUTPUT_DIR = BASE_DIR / "outputs"

START_TARGET = pd.Timestamp("2021-07-05 00:00")
ROLLING_STEPS = 24
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]
CLEAN_CHECKPOINTS_ON_START = False


def cleanup_checkpoints(checkpoint_dir: Path) -> None:
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"[debug] deleted checkpoint directory: {checkpoint_dir}")


def prepare_df(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(data_path, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(EXCLUDED_ZONES)]
    print("Earliest timestamp:", df["datetime"].min())
    print("Latest timestamp:", df["datetime"].max())
    print("Total hours:", df["datetime"].nunique())
    return df


def get_true_counts(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def compute_metrics(step_df: pd.DataFrame) -> Dict[str, float]:
    valid = step_df.dropna(subset=["y_pred", "y_true"])
    if valid.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "mean_true": float("nan")}

    errors = valid["y_pred"].to_numpy() - valid["y_true"].to_numpy()
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "mean_true": float(valid["y_true"].mean()),
    }


def run_rolling_test(df: pd.DataFrame, manager: ModelManager) -> None:
    zones = sorted(df["PULocationID"].unique())
    prediction_records: List[pd.DataFrame] = []
    metric_records: List[Dict[str, float]] = []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n///// Predicting target hour: {target_ts} in step {step} /////")
        y_true_dict = get_true_counts(df, target_ts)
        step_records: List[Dict[str, float]] = []

        for zone_id in zones:
            try:
                print(f"----- current zone is {zone_id} -----")
                pred = manager.train_and_predict_if_needed(
                    df,
                    int(zone_id),
                    target_ts,
                    auto_train=True,
                )
                true_val = float(y_true_dict.get(zone_id, 0.0))
                step_records.append(
                    {
                        "target_hour": target_ts,
                        "PULocationID": int(zone_id),
                        "y_pred": float(pred),
                        "y_true": true_val,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[diag] zone={zone_id} target={target_ts} "
                    f"error={type(exc).__name__}: {exc}"
                )
                step_records.append(
                    {
                        "target_hour": target_ts,
                        "PULocationID": int(zone_id),
                        "y_pred": np.nan,
                        "y_true": np.nan,
                        "error": str(exc),
                    }
                )

        step_df = pd.DataFrame(step_records)
        prediction_records.append(step_df)
        step_metrics = compute_metrics(step_df)
        step_metrics["target_hour"] = target_ts
        metric_records.append(step_metrics)
        print(
            f"[{MODEL_BACKEND}] {target_ts} "
            f"MAE={step_metrics['MAE']:.4f}, RMSE={step_metrics['RMSE']:.4f}"
        )

    predictions_df = pd.concat(prediction_records, ignore_index=True)
    metrics_df = pd.DataFrame(metric_records)
    overall_metrics = compute_metrics(predictions_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / f"rolling_24h_predictions_{MODEL_BACKEND}.csv"
    metrics_path = OUTPUT_DIR / f"rolling_24h_metrics_{MODEL_BACKEND}.csv"
    overall_path = OUTPUT_DIR / f"rolling_24h_overall_{MODEL_BACKEND}.txt"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    with open(overall_path, "w", encoding="utf-8") as fh:
        fh.write(f"Backend: {MODEL_BACKEND}\n")
        fh.write(f"Start target: {START_TARGET}\n")
        fh.write(f"Rolling steps: {ROLLING_STEPS}\n")
        fh.write(f"MAE: {overall_metrics['MAE']}\n")
        fh.write(f"RMSE: {overall_metrics['RMSE']}\n")
        fh.write(f"Mean true: {overall_metrics['mean_true']}\n")

    print(f"\nSaved predictions to {predictions_path}")
    print(f"Saved hourly metrics to {metrics_path}")
    print(
        f"Overall {MODEL_BACKEND.upper()} "
        f"MAE={overall_metrics['MAE']:.4f}, RMSE={overall_metrics['RMSE']:.4f}"
    )


def main() -> None:
    if CLEAN_CHECKPOINTS_ON_START:
        cleanup_checkpoints(CHECKPOINT_DIR)

    print("CUDA available:", torch.cuda.is_available())
    print("Using backend:", MODEL_BACKEND)
    print("Checkpoint dir:", CHECKPOINT_DIR)

    df = prepare_df()
    cfg = ManagerConfig()
    manager = ModelManager(checkpoint_dir=str(CHECKPOINT_DIR), cfg=cfg)
    run_rolling_test(df, manager)


if __name__ == "__main__":
    main()
