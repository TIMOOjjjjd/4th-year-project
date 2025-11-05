import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

# from persistent_multiscale_incremental import MultiScaleModelManager, ManagerConfig  # ⚠️ 改成你自己的模块路径
# from persistent_multiscale import MultiScaleModelManager
from persistent_multiscale_incre_confi import MultiScaleModelManager, ManagerConfig

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


# =====================================================
# ✅ 用户配置区（直接改这里即可）
# =====================================================
DATA_PATH = "data.parquet"
LOOKUP_PATH = "taxi-zone-lookup.csv"
CHECKPOINT_DIR = "checkpoints_multiscale"

START_TARGET = pd.Timestamp("2021-03-06 12:00")  # 第721小时
ROLLING_STEPS = 15                               # 连续预测24小时
HIDDEN_SIZE = 64                                 # 模型隐藏层大小
# EXCLUDED_ZONES = [1,2,3,   103, 104, 105, 46, 264, 265]   # 排除的区域
EXCLUDED_ZONES = [103, 104, 105, 46, 264, 265]   # 排除的区域
RETRAIN_EACH_HOUR = False                        # 是否每小时重新训练
# =====================================================


def prepare_df():
    df = pd.read_parquet(DATA_PATH, columns=["pickup_datetime", "PULocationID"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["datetime"] = df["pickup_datetime"].dt.floor("h")
    df = df[~df["PULocationID"].isin(EXCLUDED_ZONES)]
    # df = df[df["PULocationID"].isin(EXCLUDED_ZONES)]

    return df


def get_true_counts(df, target_hour):
    mask = df["datetime"] == target_hour
    return df.loc[mask].groupby("PULocationID").size()


def run_rolling(df, manager):
    seq_len = 24 * 30
    zones = sorted(df["PULocationID"].unique())
    records, metrics = [], []

    for step in range(ROLLING_STEPS):
        target_ts = START_TARGET + pd.Timedelta(hours=step)
        print(f"\n/////Predicting target hour: {target_ts} in step {step}/////")

        # 每小时重训：创建独立checkpoint目录
        if RETRAIN_EACH_HOUR:
            hour_dir = Path(CHECKPOINT_DIR) / target_ts.strftime("%Y%m%d_%H%M")
            hour_dir.mkdir(parents=True, exist_ok=True)
            mgr = MultiScaleModelManager(checkpoint_dir=str(hour_dir), hidden_size=HIDDEN_SIZE)
        else:
            mgr = manager

        y_true_dict = get_true_counts(df, target_ts)
        preds, trues = [], []

        for zid in zones:
            try:
                print(f"-----current zone is {zid}-----")
                pred = mgr.train_and_predict_if_needed(df, zid, target_ts, auto_train=True)
                true_val = float(y_true_dict.get(zid, 0))
                records.append({"target_hour": target_ts, "PULocationID": zid, "y_pred": pred, "y_true": true_val})
                preds.append(pred)
                trues.append(true_val)
            except Exception as e:
                records.append({"target_hour": target_ts, "PULocationID": zid, "y_pred": np.nan, "y_true": np.nan, "error": str(e)})

        if preds:
            preds, trues = np.array(preds), np.array(trues)
            mask = ~np.isnan(preds) & ~np.isnan(trues)
            mean_true = np.mean(trues[mask])
            if mask.any():
                mae = np.mean(np.abs(preds[mask] - trues[mask]))
                rmse = np.sqrt(np.mean((preds[mask] - trues[mask]) ** 2))

                metrics.append({
                    "target_hour": target_ts,
                    "MAE": mae,
                    "RMSE": rmse,
                    "NMAE": mae / (mean_true + 1e-8),
                    "NRMSE": rmse / (mean_true + 1e-8),
                    "mean_true": mean_true
                })
                print(f"✅ MAE={mae:.3f}, RMSE={rmse:.3f}，NMAE={mae / (mean_true + 1e-8)}，NMSE={rmse / (mean_true + 1e-8)}")
            else:
                metrics.append({"target_hour": target_ts, "MAE": np.nan, "RMSE": np.nan})
        else:
            metrics.append({"target_hour": target_ts, "MAE": np.nan, "RMSE": np.nan})

    # 保存结果


    pred_df = pd.DataFrame(records)
    if Path(LOOKUP_PATH).exists():
        lookup = pd.read_csv(LOOKUP_PATH)
        pred_df = pred_df.merge(lookup[["LocationID", "Zone"]],
                                left_on="PULocationID", right_on="LocationID", how="left").drop(columns=["LocationID"])
    pred_df.to_csv("predictions_rolling.csv", index=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("hourly_metrics.csv", index=False)

    all_data = pred_df.dropna(subset=["y_pred", "y_true"])
    if not all_data.empty:
        overall_mae = np.mean(np.abs(all_data["y_pred"] - all_data["y_true"]))
        overall_rmse = np.sqrt(np.mean((all_data["y_pred"] - all_data["y_true"]) ** 2))
    else:
        overall_mae = overall_rmse = np.nan

    with open("overall_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall MAE: {overall_mae}\nOverall RMSE: {overall_rmse}\n")

    print(f"\n🎯 Overall MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)
    df = prepare_df()

    cfg = ManagerConfig(hidden_size=HIDDEN_SIZE)
    manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, cfg=cfg)
    # manager = MultiScaleModelManager(checkpoint_dir=CHECKPOINT_DIR, hidden_size=HIDDEN_SIZE)

    run_rolling(df, manager)



if __name__ == "__main__":
    main()
