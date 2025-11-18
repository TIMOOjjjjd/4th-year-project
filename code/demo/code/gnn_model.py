from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse

HISTORY_FEATURES = ["mean_24h", "mean_168h", "mean_720h"]


class MultiScaleGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with dropout and GELU activations."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        base_pred = getattr(data, "base_pred", x[:, 0])

        x = self.sage1(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.sage2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)

        residual = self.out_linear(x).squeeze(-1)
        refined = base_pred + residual
        return residual, refined


def _prepare_predictions_frame(
    merged_csv_path: Optional[str], predictions_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    if predictions_df is not None and merged_csv_path is not None:
        raise ValueError("Provide either predictions_df or merged_csv_path, not both.")
    if predictions_df is not None:
        df_pred = predictions_df.copy()
    else:
        if merged_csv_path is None:
            raise ValueError("Either predictions_df or merged_csv_path must be provided.")
        df_pred = pd.read_csv(merged_csv_path)

    required_columns = {"PULocationID", "Prediction", "True Value"}
    missing = required_columns - set(df_pred.columns)
    if missing:
        raise ValueError(f"Predictions dataframe missing columns: {missing}")
    return df_pred


def _build_node_weights(
    df: pd.DataFrame,
    node_count: int,
    zone_idx_map: Dict[str, int],
    zone_lookup_df: pd.DataFrame,
    target_date: pd.Timestamp,
    zone_confidence: Optional[Dict[int, float]],
) -> torch.Tensor:
    """Return a node-aligned tensor with either confidence or county weights."""
    if zone_confidence is not None:
        node_weights = torch.zeros((node_count,), dtype=torch.float32)
        location_to_zone = dict(zip(zone_lookup_df["LocationID"], zone_lookup_df["Zone"]))
        for loc_id, weight in zone_confidence.items():
            zone_name = location_to_zone.get(int(loc_id))
            if zone_name is None:
                continue
            idx = zone_idx_map.get(zone_name)
            if idx is None:
                continue
            node_weights[idx] = float(np.clip(weight, 0.0, 1.0))
        return node_weights

    county_code_to_borough = {
        1: "Bronx",
        2: "Brooklyn",
        4: "Queens",
        5: "Staten Island",
        6: "Manhattan",
    }

    zone_lookup_df = zone_lookup_df.copy()
    zone_lookup_df["CountyCode"] = zone_lookup_df["Borough"].map(
        lambda borough: next(
            (k for k, v in county_code_to_borough.items() if v == borough), None
        )
    )
    location_to_county = dict(
        zip(zone_lookup_df["LocationID"], zone_lookup_df["CountyCode"])
    )

    sequence_length = 24 * 30
    start_date = target_date - pd.Timedelta(hours=sequence_length)
    df_window = df[(df["datetime"] >= start_date) & (df["datetime"] <= target_date)]

    county_volume = (
        df_window.groupby("PULocationID").size().reset_index(name="Total_Volume")
    )
    county_volume["County"] = county_volume["PULocationID"].map(location_to_county)
    county_totals = county_volume.groupby("County")["Total_Volume"].sum().to_dict()

    node_weights = torch.zeros((node_count,), dtype=torch.float32)
    zone_to_county = dict(
        zip(zone_lookup_df["Zone"], zone_lookup_df["CountyCode"])
    )

    for zone, idx in zone_idx_map.items():
        county_code = zone_to_county.get(zone)
        if county_code and county_code in county_totals:
            node_weights[idx] = county_totals[county_code]
        else:
            print(
                f"⚠️ Missing county volume data for Zone {zone} "
                f"(County: {county_code}), setting weight to 0"
            )
    max_weight = node_weights.max()
    if max_weight > 0:
        node_weights = node_weights / max_weight
    return node_weights


def run_gnn_pipeline(
    df_temp: pd.DataFrame,
    target_date: pd.Timestamp,
    excluded_zones: Sequence[int],
    device: torch.device,
    merged_csv_path: Optional[str] = None,
    zone_total_number: Optional[int] = None,
    edge_weight_csv: str = "edge_weight_matrix_with_flow.csv",
    taxi_zone_lookup: str = "taxi-zone-lookup.csv",
    final_output_csv: Optional[str] = "final_predictions_multiscale.csv",
    predictions_df: Optional[pd.DataFrame] = None,
    zone_confidence: Optional[Dict[int, float]] = None,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Execute the GraphSAGE refinement stage using GRU predictions as priors and optional
    confidence weights derived from MC Dropout variance.
    """
    df_pred = _prepare_predictions_frame(merged_csv_path, predictions_df)
    df_pred = df_pred[~df_pred["PULocationID"].isin(excluded_zones)].reset_index(drop=True)
    history_feature_cols = [col for col in HISTORY_FEATURES if col in df_pred.columns]

    df_adj = pd.read_csv(edge_weight_csv, index_col=0)
    adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj_matrix)

    zone_names = df_adj.index.tolist()
    zone_idx_map = {zone: idx for idx, zone in enumerate(zone_names)}
    node_count = len(zone_names)

    df = df_temp.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    if "datetime" not in df.columns:
        df["datetime"] = df["pickup_datetime"].dt.floor("H")

    total_zones = zone_total_number or df["PULocationID"].nunique()
    print(f"Total unique zones: {total_zones}")

    zone_lookup_df = pd.read_csv(taxi_zone_lookup).drop_duplicates(subset="LocationID")
    zone_lookup_by_zone = zone_lookup_df.drop_duplicates(subset="Zone")
    zone_to_location = dict(
        zip(zone_lookup_by_zone["Zone"], zone_lookup_by_zone["LocationID"])
    )
    location_to_zone = dict(zip(zone_lookup_df["LocationID"], zone_lookup_df["Zone"]))

    node_weights = _build_node_weights(
        df=df,
        node_count=node_count,
        zone_idx_map=zone_idx_map,
        zone_lookup_df=zone_lookup_df,
        target_date=target_date,
        zone_confidence=zone_confidence,
    )

    df_pred["Zone"] = df_pred["PULocationID"].map(location_to_zone)
    node_pred = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_label = torch.full((node_count,), float("nan"), dtype=torch.float32)
    history_tensor = (
        torch.full((node_count, len(history_feature_cols)), float("nan"), dtype=torch.float32)
        if history_feature_cols
        else None
    )

    for _, row in df_pred.iterrows():
        zone_str = row["Zone"]
        if not isinstance(zone_str, str):
            continue
        idx = zone_idx_map.get(zone_str)
        if idx is None:
            continue
        node_pred[idx] = float(row["Prediction"])
        node_label[idx] = float(row["True Value"])
        if history_tensor is not None:
            values = []
            for col in history_feature_cols:
                val = row.get(col)
                if pd.notna(val):
                    values.append(float(val))
                else:
                    values.append(float("nan"))
            history_tensor[idx] = torch.tensor(values, dtype=torch.float32)

    print("node_pred NaN count:", torch.isnan(node_pred).sum().item())
    print("node_label NaN count:", torch.isnan(node_label).sum().item())

    valid_indices = torch.where(~torch.isnan(node_pred) & ~torch.isnan(node_label))[0]
    print(f"⚠️ 仅保留有效索引数量: {valid_indices.numel()}")
    if valid_indices.numel() == 0:
        empty_df = pd.DataFrame(
            columns=["PULocationID", "ZoneName", "GRU_Pred", "Refined_Pred", "True_Value", "Confidence"]
        )
        metrics = {k: float("nan") for k in ("mae_gru", "mse_gru", "mae_refined", "mse_refined")}
        return empty_df, metrics

    filtered_zone_names = [zone_names[i] for i in valid_indices.tolist()]
    filtered_location_ids = [zone_to_location.get(name, -1) for name in filtered_zone_names]

    old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}
    valid_edges = (torch.isin(edge_index[0], valid_indices)) & (
        torch.isin(edge_index[1], valid_indices)
    )
    edge_index = edge_index[:, valid_edges]
    edge_index = torch.tensor(
        [
            [old_to_new[i.item()] for i in edge_index[0]],
            [old_to_new[j.item()] for j in edge_index[1]],
        ],
        dtype=torch.long,
    )

    node_pred = node_pred[valid_indices]
    node_label = node_label[valid_indices]
    node_weights = node_weights[valid_indices]
    if history_tensor is not None:
        history_tensor = history_tensor[valid_indices]
        history_tensor = torch.nan_to_num(history_tensor, nan=0.0)
        history_tensor = torch.log1p(history_tensor)

    node_pred_cpu = node_pred.clone()
    node_label_cpu = node_label.clone()
    node_weights_cpu = node_weights.clone()

    residual_target = node_label - node_pred

    feat_components = [node_pred.unsqueeze(1), node_weights.unsqueeze(1)]
    if history_tensor is not None and history_tensor.numel() > 0:
        feat_components.append(history_tensor)
    x_feat = torch.cat(feat_components, dim=1)
    data = Data(
        x=x_feat,
        edge_index=edge_index,
        y=residual_target,
        base_pred=node_pred,
        confidence=node_weights,
    )
    if zone_confidence is not None:
        conf_edge_weight = node_weights[edge_index[0]] * node_weights[edge_index[1]]
        data.edge_weight = conf_edge_weight

    dropout = 0.1
    learning_rate = 0.01
    hidden_dim = 256
    gnn_epochs = 300

    in_dim = x_feat.shape[1]
    model = MultiScaleGraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gnn_epochs)
    loss_func = nn.SmoothL1Loss(reduction="none")

    data = data.to(device)
    node_label_device = node_label_cpu.to(device)

    model.train()
    for epoch in range(1, gnn_epochs + 1):
        optimizer.zero_grad()
        residual_pred, refined_pred = model(data)
        loss_per_node = loss_func(residual_pred, data.y)
        diff = refined_pred - node_label_device
        sign_penalty = 0.005 * torch.relu(torch.sign(diff) * diff)

        target_residual = data.y
        significant_target = target_residual.abs() > 5.0
        sign_mismatch_mask = significant_target & ((residual_pred * target_residual) < 0)
        mismatch_penalty = 0.1 * sign_mismatch_mask.float() * torch.abs(residual_pred - target_residual)

        # loss_per_node = loss_per_node + sign_penalty
        if zone_confidence is not None:
            sample_weights = data.confidence
        else:
            sample_weights = torch.ones_like(data.y)
        mean_weight = sample_weights.mean().clamp(min=1e-6)
        normalized_weights = sample_weights / mean_weight
        loss = (loss_per_node * normalized_weights).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{gnn_epochs}, Loss = {loss.item():.4f}, "
                f"LR = {current_lr:.6f}"
            )

    model.eval()
    with torch.no_grad():
        residual_pred, refined_pred = model(data)

    y_true = node_label_cpu.cpu().numpy().squeeze()
    y_refined = refined_pred.cpu().numpy().squeeze()
    node_pred_base = node_pred_cpu.cpu().numpy().squeeze()
    confidence_vals = node_weights_cpu.cpu().numpy().squeeze()

    print("Final shapes:")
    print("node_pred shape:", node_pred_base.shape)
    print("y_refined shape:", y_refined.shape)
    print("y_true shape:", y_true.shape)

    output_df = pd.DataFrame(
        {
            "PULocationID": filtered_location_ids,
            "ZoneName": filtered_zone_names,
            "GRU_Pred": node_pred_base,
            "Refined_Pred": y_refined,
            "True_Value": y_true,
            "Confidence": confidence_vals,
        }
    )

    mae_gru = np.mean(np.abs(output_df["GRU_Pred"] - output_df["True_Value"]))
    mae_refined = np.mean(np.abs(output_df["Refined_Pred"] - output_df["True_Value"]))
    mse_gru = np.mean((output_df["GRU_Pred"] - output_df["True_Value"]) ** 2)
    mse_refined = np.mean((output_df["Refined_Pred"] - output_df["True_Value"]) ** 2)

    print(f"GRU vs True MAE = {mae_gru:.4f}")
    print(f"GRU vs True MSE = {mse_gru:.4f}")
    print(f"Refined vs True MAE = {mae_refined:.4f}")
    print(f"Refined vs True MSE = {mse_refined:.4f}")

    methods = ["GRU", "GNN(Refined)"]
    mae_values = [mae_gru, mae_refined]
    mse_values = [mse_gru, mse_refined]

    if show_plots:
        plt.figure(figsize=(6, 5))
        plt.bar(methods, mae_values, color=["blue", "orange"], alpha=0.7)
        for idx, value in enumerate(mae_values):
            plt.text(idx, value + 0.2, f"{value:.2f}", ha="center", fontsize=12)
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("Comparison of Prediction Errors (MAE)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

        plt.figure(figsize=(6, 5))
        plt.bar(methods, mse_values, color=["blue", "orange"], alpha=0.7)
        for idx, value in enumerate(mse_values):
            plt.text(idx, value + 0.2, f"{value:.2f}", ha="center", fontsize=12)
        plt.ylabel("Mean Squared Error (MSE)")
        plt.title("Comparison of Prediction Errors (MSE)")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

        plt.scatter(output_df["True_Value"], output_df["GRU_Pred"], label="GRU", alpha=0.6)
        plt.scatter(output_df["True_Value"], output_df["Refined_Pred"], label="GNN", alpha=0.6)
        min_val = np.nanmin(
            [output_df["True_Value"].min(), output_df["GRU_Pred"].min(), output_df["Refined_Pred"].min()]
        )
        max_val = np.nanmax(
            [output_df["True_Value"].max(), output_df["GRU_Pred"].max(), output_df["Refined_Pred"].max()]
        )
        plt.plot([min_val, max_val], [min_val, max_val], "k-", label="GROUND TRUTH")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.legend()
        plt.show()

    if final_output_csv:
        output_df.to_csv(
            final_output_csv,
            columns=["PULocationID", "ZoneName", "GRU_Pred", "Refined_Pred", "True_Value", "Confidence"],
            index=False,
            encoding="utf-8",
        )
        print(f"result saved to '{final_output_csv}'")

    metrics = {
        "mae_gru": mae_gru,
        "mse_gru": mse_gru,
        "mae_refined": mae_refined,
        "mse_refined": mse_refined,
    }
    return output_df, metrics
