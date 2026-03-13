from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse

HISTORY_FEATURES = ["mean_24h", "mean_168h", "mean_720h"]


class SpatialGraphSAGE(nn.Module):
    """GraphSAGE encoder that produces spatial embeddings and a regression head."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.sage2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        preds = self.out_linear(x).squeeze(-1)
        return x, preds


def _build_zone_graph(
    edge_weight_csv: str,
    taxi_zone_lookup: str,
    excluded_zones: Sequence[int],
) -> Tuple[torch.Tensor, List[str], List[int]]:
    df_adj = pd.read_csv(edge_weight_csv, index_col=0)
    df_adj.index = [str(idx).lstrip("\ufeff") for idx in df_adj.index]
    df_adj.columns = [str(col).lstrip("\ufeff") for col in df_adj.columns]

    zone_lookup_df = pd.read_csv(taxi_zone_lookup).drop_duplicates(subset="LocationID")
    zone_lookup_by_zone = zone_lookup_df.drop_duplicates(subset="Zone")
    zone_to_location = dict(zip(zone_lookup_by_zone["Zone"], zone_lookup_by_zone["LocationID"]))

    zone_names = []
    zone_locations = []
    for zone_name in df_adj.index:
        loc_id = zone_to_location.get(zone_name)
        if loc_id is None or pd.isna(loc_id):
            continue
        loc_id = int(loc_id)
        if loc_id in excluded_zones:
            continue
        zone_names.append(zone_name)
        zone_locations.append(loc_id)

    if not zone_names:
        raise ValueError("No valid zones found for GNN graph construction.")

    df_adj = df_adj.loc[zone_names, zone_names]
    adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj_matrix)
    return edge_index, zone_names, zone_locations


def _compute_prior_scores(df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[int, float]:
    df_hist = df[df["datetime"] <= target_date]
    counts = df_hist.groupby("PULocationID").size().astype(float)
    if counts.empty:
        return {}

    log_counts = np.log1p(counts)
    vmin, vmax = log_counts.min(), log_counts.max()
    if np.isclose(vmin, vmax):
        normalized = pd.Series(1.0, index=log_counts.index)
    else:
        normalized = (log_counts - vmin) / (vmax - vmin)
    scaled = 0.2 + 0.8 * normalized
    return {int(idx): float(val) for idx, val in scaled.items()}


def _compute_history_means(
    zone_hourly_counts: pd.Series, zone_id: int, target_hour: pd.Timestamp
) -> Dict[str, float]:
    means = {name: 0.0 for name in HISTORY_FEATURES}
    try:
        zone_series = zone_hourly_counts.loc[zone_id]
    except KeyError:
        return means

    for feat_name, hours in zip(HISTORY_FEATURES, (24, 24 * 7, 24 * 30)):
        start = target_hour - pd.Timedelta(hours=hours)
        window = zone_series[(zone_series.index >= start) & (zone_series.index < target_hour)]
        total = float(window.sum()) if not window.empty else 0.0
        means[feat_name] = total / float(hours)
    return means


def _build_zone_features(
    df: pd.DataFrame,
    zone_locations: List[int],
    target_date: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    df_hist = df[df["datetime"] <= target_date]
    zone_hourly_counts = df_hist.groupby(["PULocationID", "datetime"]).size().rename("count")
    prior_scores = _compute_prior_scores(df_hist, target_date)

    features = []
    labels = []
    for zone_id in zone_locations:
        history_means = _compute_history_means(zone_hourly_counts, zone_id, target_date)
        mean_hourly = df_hist[df_hist["PULocationID"] == zone_id].groupby("datetime").size()
        mean_val = float(mean_hourly.mean()) if not mean_hourly.empty else 0.0
        features.append(
            [
                np.log1p(history_means["mean_24h"]),
                np.log1p(history_means["mean_168h"]),
                np.log1p(history_means["mean_720h"]),
                prior_scores.get(int(zone_id), 0.2),
            ]
        )
        labels.append(np.log1p(mean_val))

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_spatial_gnn_embeddings(
    df: pd.DataFrame,
    target_date: Optional[pd.Timestamp],
    excluded_zones: Sequence[int],
    device: torch.device,
    edge_weight_csv: str = "edge_weight_matrix_with_flow.csv",
    taxi_zone_lookup: str = "taxi-zone-lookup.csv",
    hidden_dim: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 0.01,
    gnn_epochs: int = 400,
) -> Dict[int, np.ndarray]:
    if "datetime" not in df.columns:
        df = df.copy()
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df["datetime"] = df["pickup_datetime"].dt.floor("H")

    target_date = target_date or df["datetime"].max()
    edge_index, zone_names, zone_locations = _build_zone_graph(
        edge_weight_csv=edge_weight_csv,
        taxi_zone_lookup=taxi_zone_lookup,
        excluded_zones=excluded_zones,
    )
    features, labels = _build_zone_features(df, zone_locations, target_date)

    x_feat = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    data = Data(x=x_feat, edge_index=edge_index, y=y)

    model = SpatialGraphSAGE(in_dim=x_feat.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.SmoothL1Loss()

    data = data.to(device)
    model.train()
    for _ in range(gnn_epochs):
        optimizer.zero_grad()
        embeddings, preds = model(data)
        loss = loss_func(preds, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(data)

    embeddings_np = embeddings.cpu().numpy()
    return {int(zone_id): embeddings_np[idx] for idx, zone_id in enumerate(zone_locations)}


__all__ = ["train_spatial_gnn_embeddings", "HISTORY_FEATURES"]
