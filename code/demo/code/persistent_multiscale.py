import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


# Device selection
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MultiScaleModel(nn.Module):
    """Identical architecture to the one in code/demo.py"""

    def __init__(self, hidden_size):
        super(MultiScaleModel, self).__init__()

        self.hidden_size = hidden_size

        # LSTMs (daily & weekly)
        self.lstm_1d = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_1w = nn.LSTM(1, hidden_size, batch_first=True)

        # Transformer (monthly)
        self.input_projection = nn.Linear(1, hidden_size)
        self.transformer_1m = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=2, batch_first=True)

        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_size * 3, hidden_size)

        # GRU on fused features + raw 1h
        self.gru = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)

        # Final regressor
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x_1h, x_1d, x_1w, x_1m = x["1h"], x["1d"], x["1w"], x["1m"]

        # LSTMs
        _, (h_1d, _) = self.lstm_1d(x_1d)
        h_1d = h_1d[-1]

        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]

        # Transformer (keep same semantics as demo.py)
        x_1m = self.input_projection(x_1m)
        x_1m = x_1m.permute(1, 0, 2)
        h_1m = self.transformer_1m(x_1m, x_1m)[-1]

        # Fuse
        fused_trend = torch.cat([h_1d, h_1w, h_1m], dim=1)
        fused_trend = self.feature_fusion(fused_trend)

        # GRU input
        batch_size, seq_len, _ = x_1h.shape
        fused_trend_expanded = fused_trend.unsqueeze(1).repeat(1, seq_len, 1)
        x_gru_input = torch.cat([x_1h, fused_trend_expanded], dim=2)

        # GRU
        _, h_gru = self.gru(x_gru_input)
        h_gru = h_gru[-1]

        # Output
        output = self.fc(h_gru)
        return output


class MultiScaleModelManager:
    """Train once per zone, save model+scaler, and reuse for inference."""

    def __init__(self, checkpoint_dir: str = "checkpoints_multiscale", hidden_size: int = 64):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_size = hidden_size

        # Training hyperparams (matching demo.py)
        self.epochs = 50
        self.patience = 5
        self.learning_rate = 0.001

        # Time scales
        self.durations = {
            '1h': 1,
            '1d': 24,
            '1w': 24 * 7,
            '1m': 24 * 30
        }
        self.forecast_length = 1
        self.sequence_length = self.durations['1m']

    def _model_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"multiscale_zone_{zone_id}.pt"

    def _scaler_path(self, zone_id: int) -> Path:
        return self.checkpoint_dir / f"scaler_zone_{zone_id}.pkl"

    def has_checkpoint(self, zone_id: int) -> bool:
        return self._model_path(zone_id).exists() and self._scaler_path(zone_id).exists()

    def _save(self, zone_id: int, model: nn.Module, scaler: MinMaxScaler) -> None:
        torch.save(model.state_dict(), self._model_path(zone_id))
        with open(self._scaler_path(zone_id), 'wb') as f:
            pickle.dump(scaler, f)

    def _load(self, zone_id: int) -> tuple[nn.Module, MinMaxScaler]:
        model = MultiScaleModel(self.hidden_size).to(device)
        state = torch.load(self._model_path(zone_id), map_location=device)
        model.load_state_dict(state)
        model.eval()
        with open(self._scaler_path(zone_id), 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler

    def _prepare_zone_series(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> pd.DataFrame:
        """Return full hourly series for the zone from (target_date - 1m) to (target_date - 1h)."""
        assert 'datetime' in df.columns and 'PULocationID' in df.columns

        # Time window
        start_date = target_date - pd.Timedelta(hours=self.sequence_length)

        zone_df = df[df['PULocationID'] == zone_id].copy()
        hourly = zone_df.groupby('datetime').size().reset_index(name='passenger_count')

        # Restrict to needed range and fill missing hours with zeros
        rng = pd.date_range(start=start_date, end=target_date, freq='H')
        hourly = (
            hourly.set_index('datetime')
            .reindex(rng)
            .fillna(0)
            .rename_axis('datetime')
            .reset_index()
        )
        hourly['passenger_count'] = hourly['passenger_count'].astype(float)
        return hourly

    def _build_training_windows(self, hourly: pd.DataFrame, scaler: MinMaxScaler | None = None):
        """Build sliding windows for training; returns tensors X_dict, y_tensor, scaler."""
        seq_len = self.sequence_length
        flen = self.forecast_length
        durations = self.durations

        # Fit or reuse scaler (fit on entire history here like in demo path before target)
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(hourly[['passenger_count']])

        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])

        X_1h, X_1d, X_1w, X_1m, y = [], [], [], [], []
        # We need at least seq_len + flen samples to build one example
        for i in range(len(hourly) + 1 - seq_len - flen):
            x_1h = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1h']: i + seq_len].values
            x_1d = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1d']: i + seq_len].values
            x_1w = hourly['passenger_count_scaled'].iloc[i + seq_len - durations['1w']: i + seq_len].values
            x_1m = hourly['passenger_count_scaled'].iloc[i: i + seq_len].values

            X_1h.append(x_1h)
            X_1d.append(x_1d)
            X_1w.append(x_1w)
            X_1m.append(x_1m)

            y_val = hourly['passenger_count_scaled'].iloc[i + seq_len: i + seq_len + flen].values
            if len(y_val) == flen:
                y.append(y_val)

        X_1h = np.array(X_1h, dtype=np.float32)
        X_1d = np.array(X_1d, dtype=np.float32)
        X_1w = np.array(X_1w, dtype=np.float32)
        X_1m = np.array(X_1m, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X_tensor = {
            '1h': torch.tensor(X_1h, dtype=torch.float32).unsqueeze(-1).to(device),
            '1d': torch.tensor(X_1d, dtype=torch.float32).unsqueeze(-1).to(device),
            '1w': torch.tensor(X_1w, dtype=torch.float32).unsqueeze(-1).to(device),
            '1m': torch.tensor(X_1m, dtype=torch.float32).unsqueeze(-1).to(device),
        }
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        return X_tensor, y_tensor, scaler

    # def _build_inference_window(self, hourly: pd.DataFrame, scaler: MinMaxScaler):
    #     """Build a single last window for predicting the next hour."""
    #     durations = self.durations
    #     seq_len = self.sequence_length
    #
    #     hourly = hourly.copy()
    #     hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])
    #
    #     # Use the last contiguous window
    #     end = len(hourly)
    #     start = end - seq_len
    #
    #     x_1h = hourly['passenger_count_scaled'].iloc[end - durations['1h']: end].values
    #     x_1d = hourly['passenger_count_scaled'].iloc[end - durations['1d']: end].values
    #     x_1w = hourly['passenger_count_scaled'].iloc[end - durations['1w']: end].values
    #     x_1m = hourly['passenger_count_scaled'].iloc[start: end].values
    #
    #     X_tensor = {
    #         '1h': torch.tensor(x_1h, dtype=torch.float32).view(1, -1, 1).to(device),
    #         '1d': torch.tensor(x_1d, dtype=torch.float32).view(1, -1, 1).to(device),
    #         '1w': torch.tensor(x_1w, dtype=torch.float32).view(1, -1, 1).to(device),
    #         '1m': torch.tensor(x_1m, dtype=torch.float32).view(1, -1, 1).to(device),
    #     }
    #     return X_tensor
    def _build_inference_window(self, hourly: pd.DataFrame, scaler: MinMaxScaler):
        durations = self.durations
        seq_len = self.sequence_length

        hourly = hourly.copy()
        hourly['passenger_count_scaled'] = scaler.transform(hourly[['passenger_count']])

        # 🔧 关键：把 end 设为 len(hourly)-1，丢掉最后一个点（target_date）
        end = len(hourly) - 1  # 末尾下标指向 target_date
        start = end - seq_len  # 覆盖到 target_date-1h 共 720 点

        x_1h = hourly['passenger_count_scaled'].iloc[end - durations['1h']: end].values
        x_1d = hourly['passenger_count_scaled'].iloc[end - durations['1d']: end].values
        x_1w = hourly['passenger_count_scaled'].iloc[end - durations['1w']: end].values
        x_1m = hourly['passenger_count_scaled'].iloc[start: end].values

        X_tensor = {
            '1h': torch.tensor(x_1h, dtype=torch.float32).view(1, -1, 1).to(device),
            '1d': torch.tensor(x_1d, dtype=torch.float32).view(1, -1, 1).to(device),
            '1w': torch.tensor(x_1w, dtype=torch.float32).view(1, -1, 1).to(device),
            '1m': torch.tensor(x_1m, dtype=torch.float32).view(1, -1, 1).to(device),
        }
        return X_tensor

    def train_once(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> None:
        """Train and save model+scaler for a zone if no checkpoint exists."""
        if self.has_checkpoint(zone_id):
            return

        hourly = self._prepare_zone_series(df, zone_id, target_date)

        # Need enough history
        # if len(hourly) < self.sequence_length + self.forecast_length:
        #     raise ValueError(f"Zone {zone_id}: insufficient data for training. Got {len(hourly)} hours.")

        X_tensor, y_tensor, scaler = self._build_training_windows(hourly)

        model = MultiScaleModel(self.hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        best_loss = float('inf')
        patience_ctr = 0

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            current = loss.item()
            if current < best_loss:
                best_loss = current
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        # Save after training
        self._save(zone_id, model, scaler)

    def predict(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp) -> float:
        """Predict next hour demand for the zone at target_date using saved model."""
        if not self.has_checkpoint(zone_id):
            raise FileNotFoundError(
                f"No checkpoint for zone {zone_id}. Call train_once() first or enable auto-train.")

        model, scaler = self._load(zone_id)
        hourly = self._prepare_zone_series(df, zone_id, target_date)

        # if len(hourly) < self.sequence_length:
        #     raise ValueError(f"Zone {zone_id}: insufficient data for inference. Got {len(hourly)} hours.")

        X_last = self._build_inference_window(hourly, scaler)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_last).cpu().numpy()  # shape (1, 1)
        # Inverse transform using saved scaler
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        return float(pred)

    def train_and_predict_if_needed(self, df: pd.DataFrame, zone_id: int, target_date: pd.Timestamp,
                                    auto_train: bool = True) -> float:
        """Convenience: train once if missing, then predict."""
        if not self.has_checkpoint(zone_id):
            if not auto_train:
                raise FileNotFoundError(f"No checkpoint for zone {zone_id} and auto_train is False.")
            self.train_once(df, zone_id, target_date)
        return self.predict(df, zone_id, target_date)


def _prepare_df_from_parquet(parquet_path: str) -> pd.DataFrame:
    """Helper to load the same columns and preprocessing as demo.py relies on."""
    columns_to_load = ['pickup_datetime', 'PULocationID']
    df = pd.read_parquet(parquet_path, columns=columns_to_load)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['datetime'] = df['pickup_datetime'].dt.floor('H')
    return df


if __name__ == "__main__":
    # Example CLI usage (minimal):
    #   python code/persistent_multiscale.py
    # It will train once per zone present in the data and then produce a
    # prediction for target_date for each zone without retraining.

    import argparse

    parser = argparse.ArgumentParser(description="Train-once and predict with MultiScaleModel per zone.")
    parser.add_argument("--data", type=str, default="data.parquet", help="Input parquet file path.")
    parser.add_argument("--target", type=str, default="2021-03-05 12:00", help="Target timestamp (YYYY-mm-dd HH:MM)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size.")
    parser.add_argument("--checkpoints", type=str, default="checkpoints_multiscale", help="Checkpoint directory.")
    parser.add_argument("--zones", type=int, nargs="*", default=None, help="Optional list of PULocationID to process.")
    parser.add_argument("--no-auto-train", action="store_true", help="Disable auto-train when checkpoint missing.")
    args = parser.parse_args()

    target_date = pd.Timestamp(args.target)
    df = _prepare_df_from_parquet(args.data)

    manager = MultiScaleModelManager(checkpoint_dir=args.checkpoints, hidden_size=args.hidden)

    # Select zones
    zones = args.zones if args.zones else sorted(df['PULocationID'].unique().tolist())

    results = []
    for zid in zones:
        try:
            pred = manager.train_and_predict_if_needed(df, zid, target_date, auto_train=not args.no_auto_train)
            results.append({"PULocationID": zid, "Prediction": pred})
            print(f"Zone {zid}: Prediction @ {target_date} = {pred:.4f}")
        except Exception as e:
            print(f"Zone {zid}: skipped due to error -> {e}")

    if results:
        out_csv = Path(args.checkpoints) / f"predictions_{target_date.strftime('%Y%m%d_%H%M')}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv}")



