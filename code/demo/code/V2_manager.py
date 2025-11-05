import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from V2_multiscale_confidence import V2MultiScaleConfidence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class V2Manager:
    def __init__(self, checkpoint_dir: str = 'checkpoints_v2', hidden_size: int = 64):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size
        self.epochs = 50
        self.patience = 5
        self.lr = 1e-3
        self.durations = {
            '1d': 24,
            '1w': 24 * 7,
            '1m': 24 * 30,
        }
        self.sequence_length = self.durations['1m']
        self.forecast_length = 1

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f'v2_zone_{zone_id}.pt'

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f'scaler_v2_zone_{zone_id}.pkl'

    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._scaler_path(zone_id).exists()

    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> pd.DataFrame:
        start_date = target_date - pd.Timedelta(hours=self.sequence_length)
        zone_df = df[df['PULocationID'] == zone_id].copy()
        hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')
        rng = pd.date_range(start=start_date, end=target_date, freq='H')
        hourly = (
            hourly.set_index('datetime').reindex(rng).fillna(0)
                  .rename_axis('datetime').reset_index()
        )
        hourly['passenger_count'] = hourly['passenger_count'].astype(float)
        return hourly

    def _build_training_windows(self, hourly: pd.DataFrame, scaler: MinMaxScaler | None = None):
        seq_len = self.sequence_length
        flen = self.forecast_length
        dur = self.durations

        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(hourly[['passenger_count']])
        hourly = hourly.copy()
        hourly['scaled'] = scaler.transform(hourly[['passenger_count']])

        X1d, X1w, X1m, y = [], [], [], []
        for i in range(len(hourly) + 1 - seq_len - flen):
            x_1d = hourly['scaled'].iloc[i + seq_len - dur['1d']: i + seq_len].values
            x_1w = hourly['scaled'].iloc[i + seq_len - dur['1w']: i + seq_len].values
            x_1m = hourly['scaled'].iloc[i: i + seq_len].values
            y_val = hourly['scaled'].iloc[i + seq_len: i + seq_len + flen].values
            if len(y_val) == flen:
                X1d.append(x_1d)
                X1w.append(x_1w)
                X1m.append(x_1m)
                y.append(y_val)

        X_tensor = {
            '1d': torch.tensor(np.array(X1d, np.float32)).unsqueeze(-1).to(device),
            '1w': torch.tensor(np.array(X1w, np.float32)).unsqueeze(-1).to(device),
            '1m': torch.tensor(np.array(X1m, np.float32)).unsqueeze(-1).to(device),
        }
        y_tensor = torch.tensor(np.array(y, np.float32)).to(device)
        return X_tensor, y_tensor, scaler

    def _build_inference_window(self, hourly: pd.DataFrame, scaler: MinMaxScaler):
        dur = self.durations
        seq_len = self.sequence_length
        hourly = hourly.copy()
        hourly['scaled'] = scaler.transform(hourly[['passenger_count']])
        end = len(hourly)
        start = end - seq_len
        x_1d = hourly['scaled'].iloc[end - dur['1d']: end].values
        x_1w = hourly['scaled'].iloc[end - dur['1w']: end].values
        x_1m = hourly['scaled'].iloc[start: end].values
        X = {
            '1d': torch.tensor(x_1d, dtype=torch.float32).view(1, -1, 1).to(device),
            '1w': torch.tensor(x_1w, dtype=torch.float32).view(1, -1, 1).to(device),
            '1m': torch.tensor(x_1m, dtype=torch.float32).view(1, -1, 1).to(device),
        }
        return X

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler):
        torch.save(model.state_dict(), self._model_path(zone_id))
        import pickle
        with open(self._scaler_path(zone_id), 'wb') as f:
            pickle.dump(scaler, f)

    def _load(self, zone_id: int):
        import pickle
        model = V2MultiScaleConfidence(self.hidden_size).to(device)
        state = torch.load(self._model_path(zone_id), map_location=device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    def train_once(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp):
        if self.has_checkpoint(zone_id):
            return
        hourly = self._prepare_zone_series(df, zone_id, target_date)
        X, y, scaler = self._build_training_windows(hourly)
        model = V2MultiScaleConfidence(self.hidden_size).to(device)
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        best, patience = float('inf'), 0
        model.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            cur = loss.item()
            if cur < best:
                best, patience = cur, 0
            else:
                patience += 1
                if patience >= self.patience:
                    break
        self._save(zone_id, model, scaler)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError('Missing checkpoint; call train_once or enable auto_train')
        model, scaler = self._load(zone_id)
        hourly = self._prepare_zone_series(df, zone_id, target_date)
        X = self._build_inference_window(hourly, scaler)
        with torch.no_grad():
            pred_scaled = model(X).cpu().numpy()
        from sklearn.preprocessing import MinMaxScaler as _S
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        return float(pred)

    def train_and_predict_if_needed(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp, auto_train: bool=True) -> float:
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError('No checkpoint and auto_train=False')
            self.train_once(df, zone_id, target_date)
        return self.predict(df, zone_id, target_date)


def _prepare_df_from_parquet(parquet_path: str) -> pd.DataFrame:
    cols = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(parquet_path, columns=cols)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['datetime'] = df['pickup_datetime'].dt.floor('H')
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V2 multiscale train-once manager (no 1h GRU).')
    parser.add_argument('--data', type=str, default='data.parquet')
    parser.add_argument('--target', type=str, default='2021-03-05 12:00')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--checkpoints', type=str, default='checkpoints_v2')
    parser.add_argument('--zones', type=int, nargs='*', default=None)
    parser.add_argument('--no-auto-train', action='store_true')
    args = parser.parse_args()

    target_date = pd.Timestamp(args.target)
    df = _prepare_df_from_parquet(args.data)

    mgr = V2Manager(checkpoint_dir=args.checkpoints, hidden_size=args.hidden)
    zones = args.zones if args.zones else sorted(df['PULocationID'].unique().tolist())
    results = []
    for zid in zones:
        try:
            pred = mgr.train_and_predict_if_needed(df, zid, target_date, auto_train=not args.no_auto_train)
            results.append({'PULocationID': zid, 'Prediction': pred})
            print(f'Zone {zid}: Prediction @ {target_date} = {pred:.4f}')
        except Exception as e:
            print(f'Zone {zid}: skipped -> {e}')
    if results:
        out_csv = Path(args.checkpoints) / f"v2_predictions_{target_date.strftime('%Y%m%d_%H%M')}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f'Saved predictions to {out_csv}')
