import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os

from multiscale_confidence import MultiScaleModelManager
# from persistent_multiscale import MultiScaleModelManager
# from persistent_lstm import PureLSTMModelManager
from persistent_transformer import PureTransformerModelManager as PureLSTMModelManager

# === 临时调试：启动时自动清理旧 checkpoint 目录 ===
import shutil
for f in os.listdir('.'):
    if f.startswith("checkpoints_") and os.path.isdir(f):
        try:
            shutil.rmtree(f)
            print(f"[debug] deleted old checkpoint directory: {f}")
        except Exception as e:
            print(f"[debug] failed to delete {f}: {e}")
# === 结束 ===



# --------------------------
# 配置
# --------------------------
K_ZONES = 5
N_LABELS = 168
N_VAL = 24
HISTORY_NEEDED = 720
assert N_LABELS > N_VAL > 0

def build_label_times(target_date: pd.Timestamp, n_labels: int) -> list[pd.Timestamp]:
    start = target_date - pd.Timedelta(hours=n_labels)
    end = target_date - pd.Timedelta(hours=1)
    return list(pd.date_range(start, end, freq='H'))

def mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    # ---- Base Config ----
    target_date = pd.Timestamp('2021-04-05 16:00')
    data_parquet = 'data.parquet'
    excluded_zones = [103, 104, 105, 46, 264, 265]
    hidden_size = 64

    # ---- Build label times ----
    label_times = build_label_times(target_date, N_LABELS)
    train_label_times = label_times[:-N_VAL]
    val_label_times = label_times[-N_VAL:]

    # ---- Compute minimal time window ----
    time_min = label_times[0] - pd.Timedelta(hours=HISTORY_NEEDED)
    time_max = label_times[-1]  # == target_date - 1h

    # ---- Load & prune dataset ----
    columns_to_load = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(data_parquet, columns=columns_to_load)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df = df[(df['pickup_datetime'] >= time_min) & (df['pickup_datetime'] <= time_max)]
    df = df[~df['PULocationID'].isin(excluded_zones)]
    df['datetime'] = df['pickup_datetime'].dt.floor('H')

    # ---- Select top-K active zones ----
    top_zones = (
        df.groupby('PULocationID').size()
          .sort_values(ascending=False)
          .head(K_ZONES).index.tolist()
    )
    zone_ids = sorted(top_zones)
    print(f"Zones selected ({len(zone_ids)}): {zone_ids}")

    # ---- Two managers (独立 checkpoint 目录，避免相互覆盖) ----
    mscale = MultiScaleModelManager(checkpoint_dir='checkpoints_multiscale', hidden_size=hidden_size)
    plstm  = PureLSTMModelManager(checkpoint_dir='checkpoints_lstm', hidden_size=hidden_size,
                                  history_needed=HISTORY_NEEDED, epochs_initial=3, epochs_incremental=1)

    # ---- Main loop: 同一批标签分别预测 ----
    rows = []
    for zid in zone_ids:
        # 收集逐时预测
        recs_m = []
        recs_l = []
        for t in label_times:
            # MultiScale
            try:
                pm = mscale.train_and_predict_if_needed(df, int(zid), t, auto_train=True)
            except Exception as e:
                print(f"[warn][MultiScale] zid={zid}, t={t} pred failed: {e}")
                pm = np.nan

            # Pure LSTM
            try:
                pl = plstm.train_and_predict_if_needed(df, int(zid), t, auto_train=True)
            except Exception as e:
                print(f"[warn][PureLSTM] zid={zid}, t={t} pred failed: {e}")
                pl = np.nan

            # 真值
            true_val = (
                df[(df['datetime'] == t) & (df['PULocationID'] == zid)]
                .shape[0]
            )

            recs_m.append({'t': t, 'pred': float(pm) if pm == pm else np.nan, 'true': float(true_val)})
            recs_l.append({'t': t, 'pred': float(pl) if pl == pl else np.nan, 'true': float(true_val)})

        # 计算 MAE（train/val）
        zm = pd.DataFrame(recs_m); zm['split'] = np.where(zm['t'].isin(val_label_times), 'val', 'train')
        zl = pd.DataFrame(recs_l); zl['split'] = np.where(zl['t'].isin(val_label_times), 'val', 'train')

        m_train = mae(zm.loc[zm['split']=='train','pred'].values, zm.loc[zm['split']=='train','true'].values)
        m_val   = mae(zm.loc[zm['split']=='val','pred'].values,   zm.loc[zm['split']=='val','true'].values)

        l_train = mae(zl.loc[zl['split']=='train','pred'].values, zl.loc[zl['split']=='train','true'].values)
        l_val   = mae(zl.loc[zl['split']=='val','pred'].values,   zl.loc[zl['split']=='val','true'].values)

        rows.append({
            'PULocationID': int(zid),
            'MultiScale_train_MAE': m_train,
            'MultiScale_val_MAE':   m_val,
            'PureLSTM_train_MAE':   l_train,
            'PureLSTM_val_MAE':     l_val,
            'Val_Diff(LSTM - MS)':  l_val - m_val
        })

    out = pd.DataFrame(rows).sort_values('PULocationID')
    # 汇总行
    summary = {
        'PULocationID': 'AVG',
        'MultiScale_train_MAE': float(out['MultiScale_train_MAE'].mean()),
        'MultiScale_val_MAE':   float(out['MultiScale_val_MAE'].mean()),
        'PureLSTM_train_MAE':   float(out['PureLSTM_train_MAE'].mean()),
        'PureLSTM_val_MAE':     float(out['PureLSTM_val_MAE'].mean()),
        'Val_Diff(LSTM - MS)':  float(out['Val_Diff(LSTM - MS)'].mean()),
    }
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)

    out_csv = f"compare_LSTM_vs_MultiScale_K{K_ZONES}_L{N_LABELS}_V{N_VAL}_{pd.Timestamp('2021-04-05 16:00').strftime('%Y%m%d_%H%M')}.csv"
    out.to_csv(out_csv, index=False)
    print(out)
    print(f"Saved comparison to {out_csv}")

if __name__ == '__main__':
    main()
