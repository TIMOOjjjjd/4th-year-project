"""Train shared temporal TCN checkpoints for Ex2-Ex6.

This script trains only the per-zone temporal backbone checkpoints. The graph
experiments can then reuse the same checkpoint directory so their differences
come from residual refinement and confidence settings, not separate TCN fits.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch

from persistent_tcn import ManagerConfig, MultiScaleModelManager


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data.parquet"
DEFAULT_CHECKPOINT_DIR = BASE_DIR / "checkpoints_tcn_shared_v1"
DEFAULT_FIRST_WINDOW_START = pd.Timestamp("2021-03-01 00:00:00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train shared temporal TCN checkpoints for graph experiments."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--first-window-start",
        type=pd.Timestamp,
        default=DEFAULT_FIRST_WINDOW_START,
        help="Earliest evaluation target hour; checkpoints train through the previous hour.",
    )
    parser.add_argument("--zones", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument(
        "--clean-checkpoints",
        action="store_true",
        help="Delete the shared checkpoint directory before training.",
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


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("h")
    return df


def infer_zones(df: pd.DataFrame, requested_zones: Iterable[int] | None) -> List[int]:
    if requested_zones:
        return sorted({int(zone_id) for zone_id in requested_zones})
    return sorted(int(zone_id) for zone_id in df["PULocationID"].dropna().unique())


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    if args.clean_checkpoints and args.checkpoint_dir.exists():
        shutil.rmtree(args.checkpoint_dir)
        print(f"[shared-tcn] deleted checkpoint directory: {args.checkpoint_dir}")

    df = load_data(args.data)
    zones = infer_zones(df, args.zones)
    context_end = args.first_window_start - pd.Timedelta(hours=1)

    print("CUDA available:", torch.cuda.is_available())
    print("Data:", args.data)
    print("Checkpoint dir:", args.checkpoint_dir)
    print("First window start:", args.first_window_start)
    print("Training context end:", context_end)
    print("Zones:", len(zones))

    manager_cfg = ManagerConfig(M_mc_test=args.mc_samples)
    manager = MultiScaleModelManager(checkpoint_dir=str(args.checkpoint_dir), cfg=manager_cfg)

    for index, zone_id in enumerate(zones, start=1):
        if manager.has_checkpoint(zone_id):
            print(f"[shared-tcn] {index}/{len(zones)} zone {zone_id}: exists")
            continue
        print(f"[shared-tcn] {index}/{len(zones)} zone {zone_id}: training")
        manager.train_once(df, zone_id, context_end)

    print("[shared-tcn] complete")


if __name__ == "__main__":
    main()
