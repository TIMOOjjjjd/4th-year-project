import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from persistent_multiscale import MultiScaleModelManager


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    # Config
    target_date = pd.Timestamp('2021-04-05 16:00')
    data_parquet = 'data.parquet'
    excluded_zones = [103, 104, 105, 46, 264, 265]
    checkpoints_dir = 'checkpoints_multiscale'
    hidden_size = 64

    # Load minimal columns and preprocess to hourly
    columns_to_load = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(data_parquet, columns=columns_to_load)
    df = df[~df['PULocationID'].isin(excluded_zones)]
    zone_total_number = len(df['PULocationID'].unique())

    print('-----------------------------------------------------')
    print(zone_total_number)
    print('-----------------------------------------------------')

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['datetime'] = df['pickup_datetime'].dt.floor('H')

    # Manager handles train-once and reuse
    manager = MultiScaleModelManager(checkpoint_dir=checkpoints_dir, hidden_size=hidden_size)

    results = []
    global_predictions = []
    global_true_values = []
    skip_zone_list = []

    # Determine truth for previous hour (for metrics/plot)
    # previous_hour = target_date - pd.Timedelta(hours=1)
    true_counts = (
        df[df['datetime'] == target_date]
        .groupby('PULocationID').size()
        .to_dict()
    )

    unique_zones = df['PULocationID'].unique()
    count = len(unique_zones)

    for zone_id in unique_zones:
        print(f"Processing PULocationID: {zone_id}")
        print(f"There are {count} zones left")
        count -= 1
        try:
            # Train once if missing, then predict
            pred = manager.train_and_predict_if_needed(df, int(zone_id), target_date, auto_train=True)

            # Collect metrics inputs
            true_val = float(true_counts.get(int(zone_id), 0))
            global_predictions.append(pred)
            global_true_values.append(true_val)

            results.append({
                'PULocationID': int(zone_id),
                'Prediction': pred,
                'True Value': true_val
            })
        except Exception as e:
            print(f"Zone {zone_id}: skipped -> {e}")
            skip_zone_list.append(int(zone_id))

    # Append skipped zones as empty entries
    for skipped_id in skip_zone_list:
        results.append({'PULocationID': skipped_id, 'Prediction': None, 'True Value': None})

    # Compute global metrics
    if global_predictions and global_true_values:
        global_predictions = np.array(global_predictions)
        global_true_values = np.array(global_true_values)
        global_mae = np.mean(np.abs(global_predictions - global_true_values))
        global_mse = np.mean((global_predictions - global_true_values) ** 2)
        print(f"Overall Average MAE: {global_mae:.4f}")
        print(f"Overall Average MSE: {global_mse:.4f}")
    else:
        print("No valid predictions available for computing global metrics.")

    # Save raw results
    results_df = pd.DataFrame(results)

    # Map LocationID to Zone names
    lookup_table = pd.read_csv('taxi-zone-lookup.csv')
    results_df = pd.merge(
        results_df,
        lookup_table[['LocationID', 'Zone']],
        left_on='PULocationID',
        right_on='LocationID',
        how='left'
    )

    # Merge by Zone (average duplicates)
    merged_results = results_df.groupby('Zone', as_index=False).agg({
        'PULocationID': 'first',
        'Prediction': 'mean',
        'True Value': 'mean'
    })

    epochs_used = 50  # matches manager default for naming consistency
    out_csv = f'GRU_Merged_{epochs_used}epochs.csv'
    merged_results.to_csv(out_csv, index=False)
    print(f"Merged results saved to {out_csv}")

    # Plot if data available
    if not results_df.empty:
        plot_df = results_df.dropna(subset=['Prediction', 'True Value'])
        if not plot_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(plot_df['Prediction'].values, label='Prediction', linestyle='--', marker='o')
            plt.plot(plot_df['True Value'].values, label='True Value', linestyle='-', marker='x')
            plt.title('Prediction vs True Value')
            plt.xlabel('Taxi_zone ID (index order)')
            plt.ylabel('Passenger Demand')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("No data available for plotting after dropping NaNs.")
    else:
        print("No data available for plotting.")


if __name__ == '__main__':
    main()

