import pandas as pd
import numpy as np
import torch
from pathlib import Path

from persistent_multiscale import MultiScaleModelManager


# --------------------------
# Smoke-test configuration
# --------------------------
K_ZONES = 10          # 只跑最活跃的前 K 个区域
N_LABELS = 168        # 总共评估的标签时刻（小时数）
N_VAL = 24            # 最后 N_VAL 个小时做验证
HISTORY_NEEDED = 720 # 管理器构建样本所需历史窗口（小时）
assert N_LABELS > N_VAL > 0


def build_label_times(target_date: pd.Timestamp, n_labels: int) -> list[pd.Timestamp]:
    """生成最近 n_labels 个小时的标签时刻：[..., target-3h, target-2h, target-1h]"""
    start = target_date - pd.Timedelta(hours=n_labels)
    end = target_date - pd.Timedelta(hours=1)
    return list(pd.date_range(start, end, freq='H'))


def main():
    device = torch.device("cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    # ---- Base Config ----
    target_date = pd.Timestamp('2021-04-05 16:00')
    data_parquet = 'data.parquet'
    excluded_zones = [103, 104, 105, 46, 264, 265]
    checkpoints_dir = 'checkpoints_multiscale'
    hidden_size = 64

    # ---- Build label times (small set) ----
    label_times = build_label_times(target_date, N_LABELS)
    train_label_times = label_times[:-N_VAL]
    val_label_times = label_times[-N_VAL:]
    assert len(train_label_times) > 0 and len(val_label_times) > 0

    # ---- Compute minimal time window to read ----
    time_min = label_times[0] - pd.Timedelta(hours=HISTORY_NEEDED)
    time_max = label_times[-1]  # == target_date - 1h

    # ---- Load & prune dataset to minimal window ----
    columns_to_load = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(data_parquet, columns=columns_to_load)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # 只保留必要时间窗
    df = df[(df['pickup_datetime'] >= time_min) & (df['pickup_datetime'] <= time_max)]

    # 区域过滤 & 小时对齐
    df = df[~df['PULocationID'].isin(excluded_zones)]
    df['datetime'] = df['pickup_datetime'].dt.floor('H')

    # ---- Select top-K active zones to speed up ----
    # 若想随机挑选，把 sort_values 替换为 sample(n=K_ZONES, random_state=0)
    top_zones = (
        df.groupby('PULocationID').size()
          .sort_values(ascending=False)
          .head(K_ZONES).index.tolist()
    )
    zone_ids = sorted(top_zones)
    print(f"Zones selected ({len(zone_ids)}): {zone_ids}")

    manager = MultiScaleModelManager(checkpoint_dir=checkpoints_dir, hidden_size=hidden_size)

    # ---- Main loop ----
    results = []

    for zid in zone_ids:
        zone_records = []
        for t in label_times:
            # manager 内部会用 [t-720h, ..., t-1h] 的历史
            try:
                pred = manager.train_and_predict_if_needed(df, int(zid), t, auto_train=True)
            except Exception as e:
                print(f"[warn] zid={zid}, t={t} pred failed: {e}")
                pred = np.nan

            # 该小时真实值
            true_val = (
                df[(df['datetime'] == t) & (df['PULocationID'] == zid)]
                .shape[0]
            )
            zone_records.append({
                'PULocationID': int(zid),
                't': t,
                'Prediction': float(pred) if pred == pred else np.nan,
                'True': float(true_val)
            })

        zone_df = pd.DataFrame(zone_records)
        zone_df['split'] = np.where(zone_df['t'].isin(val_label_times), 'val', 'train')

        def mae(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan

        train_mae = mae(zone_df.loc[zone_df['split'] == 'train', 'Prediction'].values,
                        zone_df.loc[zone_df['split'] == 'train', 'True'].values)
        val_mae = mae(zone_df.loc[zone_df['split'] == 'val', 'Prediction'].values,
                      zone_df.loc[zone_df['split'] == 'val', 'True'].values)

        results.append({
            'PULocationID': int(zid),
            'train_mae': train_mae,
            'val_mae': val_mae
        })

    # ---- Save ----
    out_csv = f"smoke_results_K{K_ZONES}_L{N_LABELS}_V{N_VAL}_{target_date.strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved per-zone train/val MAE to {out_csv}")


if __name__ == '__main__':
    main()
