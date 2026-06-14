"""Experiment 1 runner for temporal forecasting backends.

Default run:
    python code/demo_test.py

This evaluates every backend over twelve 24-hour rolling windows between
March and December 2021 and writes all outputs to code/ex_1_result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


warnings.simplefilter(action="ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data.parquet"
DEFAULT_RESULTS_DIR = BASE_DIR / "ex_1_result"

ALL_BACKENDS = (
    "lstm",
    "gru",
    "transformer",
    "multiscale",
    "sarima",
    "tcn",
    "dlinear",
    "patchtst",
)
EXCLUDED_ZONES = (103, 104, 105, 46, 264, 265)
ROLLING_STEPS = 24

DEFAULT_WINDOW_STARTS = (
    "2021-03-01 00:00",
    "2021-03-29 00:00",
    "2021-04-26 00:00",
    "2021-05-24 00:00",
    "2021-06-21 00:00",
    "2021-07-05 00:00",
    "2021-07-19 00:00",
    "2021-08-16 00:00",
    "2021-09-13 00:00",
    "2021-10-11 00:00",
    "2021-11-08 00:00",
    "2021-12-06 00:00",
)

BACKEND_IMPORTS = {
    "lstm": ("persistent_lstm", "ManagerConfig", "PureLSTMModelManager"),
    "gru": ("persistent_gru", "ManagerConfig", "PureGRUModelManager"),
    "transformer": (
        "persistent_transformer",
        "ManagerConfig",
        "PureTransformerModelManager",
    ),
    # Keep the incremental confidence-weighted multiscale version for Experiment 1.
    "multiscale": (
        "persistent_multiscale_incre_confi",
        "ManagerConfig",
        "MultiScaleModelManager",
    ),
    "sarima": ("persistent_sarima", "ManagerConfig", "SARIMAModelManager"),
    "tcn": ("persistent_tcn", "ManagerConfig", "PureTCNModelManager"),
    "dlinear": ("persistent_dlinear", "ManagerConfig", "DLinearModelManager"),
    "patchtst": ("persistent_patchtst", "ManagerConfig", "PatchTSTModelManager"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 1 over multiple 24-hour windows and temporal "
            "forecasting backends."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=ALL_BACKENDS,
        default=list(ALL_BACKENDS),
        help="Backends to run sequentially. Default runs all backends.",
    )
    parser.add_argument(
        "--window-starts",
        nargs="*",
        default=None,
        help=(
            "Optional explicit start timestamps. Default uses twelve 24-hour "
            "windows from March to December 2021."
        ),
    )
    parser.add_argument("--rolling-steps", type=int, default=ROLLING_STEPS)
    parser.add_argument(
        "--excluded-zones",
        type=int,
        nargs="*",
        default=list(EXCLUDED_ZONES),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help=(
            "Reuse existing checkpoints under ex_1_result/checkpoints. By "
            "default, selected backend checkpoints are cleaned to avoid leakage."
        ),
    )
    parser.add_argument(
        "--max-zones",
        type=int,
        default=None,
        help="Optional smoke-test limit. Default evaluates every non-excluded zone.",
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


def load_backend(backend: str) -> Tuple[Any, Any]:
    module_name, config_name, manager_name = BACKEND_IMPORTS[backend]
    module = importlib.import_module(module_name)
    return getattr(module, config_name), getattr(module, manager_name)


def normalize_window_starts(values: Optional[Sequence[str]]) -> List[pd.Timestamp]:
    raw_values = list(values) if values else list(DEFAULT_WINDOW_STARTS)
    starts = sorted({pd.Timestamp(value).floor("h") for value in raw_values})
    if not starts:
        raise ValueError("At least one window start is required.")
    return starts


def prepare_df(data_path: Path, excluded_zones: Sequence[int]) -> pd.DataFrame:
    df = pd.read_parquet(data_path, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(excluded_zones)].copy()

    print("Earliest timestamp:", df["datetime"].min(), flush=True)
    print("Latest timestamp:", df["datetime"].max(), flush=True)
    print("Total hours:", df["datetime"].nunique(), flush=True)
    print("Total non-excluded zones:", df["PULocationID"].nunique(), flush=True)
    return df


def validate_windows(
    df: pd.DataFrame,
    window_starts: Sequence[pd.Timestamp],
    rolling_steps: int,
) -> None:
    if rolling_steps <= 0:
        raise ValueError("--rolling-steps must be positive.")

    min_ts = df["datetime"].min()
    max_ts = df["datetime"].max()
    for start in window_starts:
        end = start + pd.Timedelta(hours=rolling_steps - 1)
        if start < min_ts or end > max_ts:
            raise ValueError(
                f"Window {start} to {end} is outside data range {min_ts} to {max_ts}."
            )


def get_true_counts(df: pd.DataFrame, target_hour: pd.Timestamp) -> pd.Series:
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def compute_metrics(records_df: pd.DataFrame) -> Dict[str, float]:
    if records_df.empty:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MSE": float("nan"),
            "mean_true": float("nan"),
            "valid_samples": 0.0,
            "failed_samples": 0.0,
        }

    valid = records_df.dropna(subset=["y_pred", "y_true"])
    failed = int(records_df["error"].astype(str).ne("").sum())
    if valid.empty:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MSE": float("nan"),
            "mean_true": float("nan"),
            "valid_samples": 0.0,
            "failed_samples": float(failed),
        }

    errors = valid["y_pred"].to_numpy(dtype=float) - valid["y_true"].to_numpy(dtype=float)
    mse = float(np.mean(errors**2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(mse)),
        "MSE": mse,
        "mean_true": float(valid["y_true"].mean()),
        "valid_samples": float(len(valid)),
        "failed_samples": float(failed),
    }


def clean_backend_artifacts(results_dir: Path, backend: str) -> None:
    for path in (results_dir / backend, results_dir / "checkpoints" / backend):
        if path.exists():
            shutil.rmtree(path)
            print(f"[clean] deleted {path}", flush=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(payload, default=str))
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def write_backend_outputs(
    backend_dir: Path,
    backend: str,
    prediction_frames: Sequence[pd.DataFrame],
    hourly_metrics: Sequence[Dict[str, Any]],
    window_metrics: Sequence[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    backend_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    hourly_df = pd.DataFrame(hourly_metrics)
    window_df = pd.DataFrame(window_metrics)

    predictions_df.to_csv(backend_dir / f"predictions_{backend}.csv", index=False)
    hourly_df.to_csv(backend_dir / f"hourly_metrics_{backend}.csv", index=False)
    window_df.to_csv(backend_dir / f"window_metrics_{backend}.csv", index=False)

    if "error" in predictions_df.columns:
        failure_log = predictions_df[predictions_df["error"].astype(str).ne("")]
    else:
        failure_log = pd.DataFrame()
    failure_log.to_csv(backend_dir / f"failure_log_{backend}.csv", index=False)
    return predictions_df, hourly_df, window_df


def write_overall_text(path: Path, backend: str, metrics: Dict[str, Any]) -> None:
    lines = [f"Backend: {backend}"]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backend(
    df: pd.DataFrame,
    backend: str,
    zones: Sequence[int],
    window_starts: Sequence[pd.Timestamp],
    rolling_steps: int,
    checkpoint_dir: Path,
    backend_dir: Path,
) -> Dict[str, Any]:
    ManagerConfig, ModelManager = load_backend(backend)
    manager = ModelManager(checkpoint_dir=str(checkpoint_dir), cfg=ManagerConfig())

    prediction_frames: List[pd.DataFrame] = []
    hourly_metrics: List[Dict[str, Any]] = []
    window_metrics: List[Dict[str, Any]] = []

    for window_idx, window_start in enumerate(window_starts, start=1):
        print(
            f"\n[{backend}] window {window_idx}/{len(window_starts)} "
            f"start={window_start}",
            flush=True,
        )
        window_frames: List[pd.DataFrame] = []

        for step in range(rolling_steps):
            target_ts = window_start + pd.Timedelta(hours=step)
            print(
                f"[{backend}] target={target_ts} "
                f"step={step + 1}/{rolling_steps}",
                flush=True,
            )
            y_true_dict = get_true_counts(df, target_ts)
            step_records: List[Dict[str, Any]] = []

            for zone_pos, zone_id in enumerate(zones, start=1):
                true_val = float(y_true_dict.get(zone_id, 0.0))
                try:
                    pred = manager.train_and_predict_if_needed(
                        df,
                        int(zone_id),
                        target_ts,
                        auto_train=True,
                    )
                    step_records.append(
                        {
                            "backend": backend,
                            "window_id": window_idx,
                            "window_start": window_start,
                            "target_hour": target_ts,
                            "PULocationID": int(zone_id),
                            "y_pred": float(pred),
                            "y_true": true_val,
                            "error": "",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[diag] backend={backend} zone={zone_id} "
                        f"target={target_ts} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    step_records.append(
                        {
                            "backend": backend,
                            "window_id": window_idx,
                            "window_start": window_start,
                            "target_hour": target_ts,
                            "PULocationID": int(zone_id),
                            "y_pred": np.nan,
                            "y_true": true_val,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

                if zone_pos % 50 == 0:
                    print(
                        f"[{backend}] target={target_ts} "
                        f"processed {zone_pos}/{len(zones)} zones",
                        flush=True,
                    )

            step_df = pd.DataFrame(step_records)
            window_frames.append(step_df)
            prediction_frames.append(step_df)

            step_metrics = compute_metrics(step_df)
            step_metrics.update(
                {
                    "backend": backend,
                    "window_id": window_idx,
                    "window_start": window_start,
                    "target_hour": target_ts,
                }
            )
            hourly_metrics.append(step_metrics)
            print(
                f"[{backend}] {target_ts} "
                f"MAE={step_metrics['MAE']:.4f}, "
                f"RMSE={step_metrics['RMSE']:.4f}, "
                f"failures={int(step_metrics['failed_samples'])}",
                flush=True,
            )

        window_df = pd.concat(window_frames, ignore_index=True)
        summary = compute_metrics(window_df)
        summary.update(
            {
                "backend": backend,
                "window_id": window_idx,
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(hours=rolling_steps - 1),
                "rolling_steps": rolling_steps,
            }
        )
        window_metrics.append(summary)

        window_file = backend_dir / f"predictions_{backend}_window_{window_idx:02d}.csv"
        window_df.to_csv(window_file, index=False)
        write_backend_outputs(
            backend_dir,
            backend,
            prediction_frames,
            hourly_metrics,
            window_metrics,
        )
        print(
            f"[{backend}] window {window_idx} summary "
            f"MAE={summary['MAE']:.4f}, RMSE={summary['RMSE']:.4f}",
            flush=True,
        )

    predictions_df, _, _ = write_backend_outputs(
        backend_dir,
        backend,
        prediction_frames,
        hourly_metrics,
        window_metrics,
    )
    overall = compute_metrics(predictions_df)
    overall.update(
        {
            "backend": backend,
            "windows": len(window_starts),
            "rolling_steps_per_window": rolling_steps,
            "zones": len(zones),
        }
    )
    write_json(backend_dir / f"overall_{backend}.json", overall)
    write_overall_text(backend_dir / f"overall_{backend}.txt", backend, overall)
    return overall


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    window_starts = normalize_window_starts(args.window_starts)
    results_dir = args.results_dir.resolve()
    checkpoint_root = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Results dir:", results_dir, flush=True)
    print("Backends:", ", ".join(args.backends), flush=True)
    print(
        "Window starts:",
        ", ".join(str(start) for start in window_starts),
        flush=True,
    )

    df = prepare_df(args.data, args.excluded_zones)
    validate_windows(df, window_starts, args.rolling_steps)

    zones = sorted(int(zone_id) for zone_id in df["PULocationID"].unique())
    if args.max_zones is not None:
        zones = zones[: max(1, args.max_zones)]
        print(f"Using first {len(zones)} zones for smoke-test run.", flush=True)

    run_config = {
        "data": args.data,
        "results_dir": results_dir,
        "backends": args.backends,
        "window_starts": window_starts,
        "rolling_steps": args.rolling_steps,
        "excluded_zones": args.excluded_zones,
        "zones": len(zones),
        "seed": args.seed,
        "reuse_checkpoints": args.reuse_checkpoints,
        "multiscale_backend": "persistent_multiscale_incre_confi",
        "dlinear_backend": "persistent_dlinear",
        "patchtst_backend": "persistent_patchtst",
    }
    write_json(results_dir / "run_config.json", run_config)

    overall_records: List[Dict[str, Any]] = []
    for backend_idx, backend in enumerate(args.backends, start=1):
        set_random_seed(args.seed + backend_idx)
        backend_dir = results_dir / backend
        checkpoint_dir = checkpoint_root / backend
        if not args.reuse_checkpoints:
            clean_backend_artifacts(results_dir, backend)

        backend_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n===== Running backend {backend_idx}/{len(args.backends)}: {backend} =====",
            flush=True,
        )
        print("Checkpoint dir:", checkpoint_dir, flush=True)

        try:
            overall = run_backend(
                df=df,
                backend=backend,
                zones=zones,
                window_starts=window_starts,
                rolling_steps=args.rolling_steps,
                checkpoint_dir=checkpoint_dir,
                backend_dir=backend_dir,
            )
        except Exception as exc:  # noqa: BLE001
            overall = {
                "backend": backend,
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "MSE": float("nan"),
                "mean_true": float("nan"),
                "valid_samples": 0.0,
                "failed_samples": float(len(zones) * len(window_starts) * args.rolling_steps),
                "windows": len(window_starts),
                "rolling_steps_per_window": args.rolling_steps,
                "zones": len(zones),
                "fatal_error": f"{type(exc).__name__}: {exc}",
            }
            write_json(backend_dir / f"overall_{backend}.json", overall)
            write_overall_text(backend_dir / f"overall_{backend}.txt", backend, overall)
            print(
                f"[fatal] backend={backend} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

        overall_records.append(overall)
        print(
            f"===== Completed {backend}: MAE={overall['MAE']}, "
            f"RMSE={overall['RMSE']} =====",
            flush=True,
        )

    overall_df = pd.DataFrame(overall_records)
    overall_df.to_csv(results_dir / "overall_summary.csv", index=False)

    window_frames = []
    hourly_frames = []
    for backend in args.backends:
        path = results_dir / backend / f"window_metrics_{backend}.csv"
        if path.exists():
            window_frames.append(pd.read_csv(path))
        path = results_dir / backend / f"hourly_metrics_{backend}.csv"
        if path.exists():
            hourly_frames.append(pd.read_csv(path))
    if window_frames:
        pd.concat(window_frames, ignore_index=True).to_csv(
            results_dir / "window_summary_all_backends.csv",
            index=False,
        )
    if hourly_frames:
        pd.concat(hourly_frames, ignore_index=True).to_csv(
            results_dir / "hourly_metrics_all_backends.csv",
            index=False,
        )

    print("\nExperiment 1 finished.", flush=True)
    print("Overall summary:", results_dir / "overall_summary.csv", flush=True)


if __name__ == "__main__":
    main()
