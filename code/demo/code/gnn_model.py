from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
    merged_csv_path: Optional[str],
    predictions_df: Optional[pd.DataFrame],
    require_label: bool,
) -> pd.DataFrame:
    if predictions_df is not None and merged_csv_path is not None:
        raise ValueError("Provide either predictions_df or merged_csv_path, not both.")
    if predictions_df is not None:
        df_pred = predictions_df.copy()
    else:
        if merged_csv_path is None:
            raise ValueError("Either predictions_df or merged_csv_path must be provided.")
        df_pred = pd.read_csv(merged_csv_path)

    required_columns = {"PULocationID", "Prediction"}
    if require_label:
        required_columns.add("True Value")
    missing = required_columns - set(df_pred.columns)
    if missing:
        raise ValueError(f"Predictions dataframe missing columns: {missing}")
    return df_pred


def _remap_edges_to_valid_nodes(
    edge_index: torch.Tensor, valid_indices: torch.Tensor
) -> torch.Tensor:
    valid_old = [int(idx) for idx in valid_indices.tolist()]
    valid_set = set(valid_old)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_old)}

    remapped_edges: List[Tuple[int, int]] = []
    for src, dst in edge_index.t().tolist():
        src_i, dst_i = int(src), int(dst)
        if src_i in valid_set and dst_i in valid_set:
            remapped_edges.append((old_to_new[src_i], old_to_new[dst_i]))

    if not remapped_edges:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()


def _build_graph_snapshot(
    df_pred: pd.DataFrame,
    edge_index: torch.Tensor,
    zone_names: Sequence[str],
    zone_idx_map: Dict[str, int],
    location_to_zone: Dict[int, str],
    zone_to_location: Dict[str, int],
    history_feature_cols: Sequence[str],
    fallback_weights: Optional[torch.Tensor],
    require_label: bool,
) -> Data:
    node_count = len(zone_names)
    node_pred = torch.full((node_count,), float("nan"), dtype=torch.float32)
    node_label = torch.full((node_count,), float("nan"), dtype=torch.float32)
    if fallback_weights is None:
        node_weights = torch.ones((node_count,), dtype=torch.float32)
    else:
        node_weights = fallback_weights.clone().to(dtype=torch.float32)
    history_tensor = (
        torch.full((node_count, len(history_feature_cols)), float("nan"), dtype=torch.float32)
        if history_feature_cols
        else None
    )

    for _, row in df_pred.iterrows():
        try:
            loc_id = int(row["PULocationID"])
        except (TypeError, ValueError):
            continue

        zone_str = location_to_zone.get(loc_id)
        if not isinstance(zone_str, str):
            continue
        idx = zone_idx_map.get(zone_str)
        if idx is None:
            continue

        if pd.notna(row.get("Prediction")):
            node_pred[idx] = float(row["Prediction"])
        if "True Value" in df_pred.columns and pd.notna(row.get("True Value")):
            node_label[idx] = float(row["True Value"])
        if "Confidence" in df_pred.columns and pd.notna(row.get("Confidence")):
            node_weights[idx] = float(np.clip(row["Confidence"], 0.0, 1.0))

        if history_tensor is not None:
            values = []
            for col in history_feature_cols:
                val = row.get(col)
                values.append(float(val) if pd.notna(val) else float("nan"))
            history_tensor[idx] = torch.tensor(values, dtype=torch.float32)

    valid_mask = ~torch.isnan(node_pred)
    if require_label:
        valid_mask = valid_mask & ~torch.isnan(node_label)
    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        raise ValueError("No valid graph nodes available.")

    remapped_edge_index = _remap_edges_to_valid_nodes(edge_index, valid_indices)

    node_pred = node_pred[valid_indices]
    node_label = node_label[valid_indices]
    node_weights = node_weights[valid_indices]
    if history_tensor is not None:
        history_tensor = history_tensor[valid_indices]
        history_tensor = torch.nan_to_num(history_tensor, nan=0.0)
        history_tensor = torch.log1p(history_tensor)

    feat_components = [node_pred.unsqueeze(1), node_weights.unsqueeze(1)]
    if history_tensor is not None and history_tensor.numel() > 0:
        feat_components.append(history_tensor)
    x_feat = torch.cat(feat_components, dim=1)

    valid_old_indices = [int(idx) for idx in valid_indices.cpu().tolist()]
    filtered_zone_names = [str(zone_names[idx]) for idx in valid_old_indices]
    filtered_location_ids = [
        int(zone_to_location.get(zone_name, -1)) for zone_name in filtered_zone_names
    ]

    if require_label:
        residual_target = node_label - node_pred
    else:
        residual_target = torch.full_like(node_pred, float("nan"))
    data = Data(
        x=x_feat,
        edge_index=remapped_edge_index,
        y=residual_target,
        base_pred=node_pred,
        true_value=node_label,
        confidence=node_weights,
    )
    data.zone_names = tuple(filtered_zone_names)
    data.location_ids = tuple(filtered_location_ids)
    return data


def _build_training_graph(
    train_predictions_df: pd.DataFrame,
    edge_index: torch.Tensor,
    zone_names: Sequence[str],
    zone_idx_map: Dict[str, int],
    location_to_zone: Dict[int, str],
    zone_to_location: Dict[str, int],
    history_feature_cols: Sequence[str],
) -> Optional[Data]:
    if train_predictions_df.empty:
        return None

    if "target_hour" in train_predictions_df.columns:
        grouped_frames = [
            frame for _, frame in train_predictions_df.groupby("target_hour", sort=True)
        ]
    else:
        grouped_frames = [train_predictions_df]

    snapshots: List[Data] = []
    for frame in grouped_frames:
        try:
            snapshot = _build_graph_snapshot(
                df_pred=frame,
                edge_index=edge_index,
                zone_names=zone_names,
                zone_idx_map=zone_idx_map,
                location_to_zone=location_to_zone,
                zone_to_location=zone_to_location,
                history_feature_cols=history_feature_cols,
                fallback_weights=None,
                require_label=True,
            )
        except ValueError:
            continue
        snapshots.append(snapshot)

    if not snapshots:
        return None

    x_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    base_parts: List[torch.Tensor] = []
    confidence_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    offset = 0
    for snapshot in snapshots:
        x_parts.append(snapshot.x)
        y_parts.append(snapshot.y)
        base_parts.append(snapshot.base_pred)
        confidence_parts.append(snapshot.confidence)
        if snapshot.edge_index.numel() > 0:
            edge_parts.append(snapshot.edge_index + offset)
        offset += snapshot.num_nodes

    if edge_parts:
        combined_edge_index = torch.cat(edge_parts, dim=1)
    else:
        combined_edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(
        x=torch.cat(x_parts, dim=0),
        edge_index=combined_edge_index,
        y=torch.cat(y_parts, dim=0),
        base_pred=torch.cat(base_parts, dim=0),
        confidence=torch.cat(confidence_parts, dim=0),
    )


def _build_output_frame(
    inference_data: Data,
    refined_pred: torch.Tensor,
    final_output_csv: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_true = inference_data.true_value.cpu().numpy().reshape(-1)
    y_refined = refined_pred.detach().cpu().numpy().reshape(-1)
    node_pred_base = inference_data.base_pred.cpu().numpy().reshape(-1)
    confidence_vals = inference_data.confidence.cpu().numpy().reshape(-1)

    output_df = pd.DataFrame(
        {
            "PULocationID": list(inference_data.location_ids),
            "ZoneName": list(inference_data.zone_names),
            "GRU_Pred": node_pred_base,
            "Refined_Pred": y_refined,
            "True_Value": y_true,
            "Confidence": confidence_vals,
        }
    )

    valid_metric = output_df[["GRU_Pred", "Refined_Pred", "True_Value"]].notna().all(axis=1)
    if valid_metric.any():
        metric_df = output_df.loc[valid_metric]
        mae_gru = np.mean(np.abs(metric_df["GRU_Pred"] - metric_df["True_Value"]))
        mae_refined = np.mean(np.abs(metric_df["Refined_Pred"] - metric_df["True_Value"]))
        mse_gru = np.mean((metric_df["GRU_Pred"] - metric_df["True_Value"]) ** 2)
        mse_refined = np.mean((metric_df["Refined_Pred"] - metric_df["True_Value"]) ** 2)
    else:
        mae_gru = mae_refined = mse_gru = mse_refined = float("nan")

    print(f"GRU vs True MAE = {mae_gru:.4f}")
    print(f"GRU vs True MSE = {mse_gru:.4f}")
    print(f"Refined vs True MAE = {mae_refined:.4f}")
    print(f"Refined vs True MSE = {mse_refined:.4f}")

    if final_output_csv:
        output_df.to_csv(
            final_output_csv,
            columns=[
                "PULocationID",
                "ZoneName",
                "GRU_Pred",
                "Refined_Pred",
                "True_Value",
                "Confidence",
            ],
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
    df_window = df[(df["datetime"] >= start_date) & (df["datetime"] < target_date)]

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
    train_predictions_df: Optional[pd.DataFrame] = None,
    zone_confidence: Optional[Dict[int, float]] = None,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Train GraphSAGE on historical residuals, then refine the current target hour.

    ``predictions_df`` is the current inference snapshot. Its ``True Value`` column is
    used only for metrics. ``train_predictions_df`` must contain historical snapshots
    with labels and is the only source used to form residual training targets.
    """
    df_pred = _prepare_predictions_frame(
        merged_csv_path=merged_csv_path,
        predictions_df=predictions_df,
        require_label=False,
    )
    df_pred = df_pred[~df_pred["PULocationID"].isin(excluded_zones)].reset_index(
        drop=True
    )
    if "True Value" not in df_pred.columns:
        df_pred["True Value"] = np.nan

    if train_predictions_df is None:
        train_df = pd.DataFrame()
    else:
        train_df = train_predictions_df.copy()
        required_train_cols = {"PULocationID", "Prediction", "True Value"}
        missing = required_train_cols - set(train_df.columns)
        if missing:
            raise ValueError(f"Training dataframe missing columns: {missing}")
        train_df = train_df[~train_df["PULocationID"].isin(excluded_zones)].reset_index(
            drop=True
        )

    available_cols = set(df_pred.columns) | set(train_df.columns)
    history_feature_cols = [col for col in HISTORY_FEATURES if col in available_cols]

    df_adj = pd.read_csv(edge_weight_csv, index_col=0)
    df_adj.index = [str(idx).lstrip("\ufeff") for idx in df_adj.index]
    df_adj.columns = [str(col).lstrip("\ufeff") for col in df_adj.columns]
    df_adj = df_adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)
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

    try:
        inference_data = _build_graph_snapshot(
            df_pred=df_pred,
            edge_index=edge_index,
            zone_names=zone_names,
            zone_idx_map=zone_idx_map,
            location_to_zone=location_to_zone,
            zone_to_location=zone_to_location,
            history_feature_cols=history_feature_cols,
            fallback_weights=node_weights,
            require_label=False,
        )
    except ValueError:
        empty_df = pd.DataFrame(
            columns=[
                "PULocationID",
                "ZoneName",
                "GRU_Pred",
                "Refined_Pred",
                "True_Value",
                "Confidence",
            ]
        )
        metrics = {
            key: float("nan")
            for key in ("mae_gru", "mse_gru", "mae_refined", "mse_refined")
        }
        return empty_df, metrics

    train_data = _build_training_graph(
        train_predictions_df=train_df,
        edge_index=edge_index,
        zone_names=zone_names,
        zone_idx_map=zone_idx_map,
        location_to_zone=location_to_zone,
        zone_to_location=zone_to_location,
        history_feature_cols=history_feature_cols,
    )

    if train_data is None:
        print("No historical GNN labels available; using base predictions unchanged.")
        output_df, metrics = _build_output_frame(
            inference_data=inference_data,
            refined_pred=inference_data.base_pred,
            final_output_csv=final_output_csv,
        )
        return output_df, metrics

    dropout = 0.1
    learning_rate = 0.01
    hidden_dim = 256
    gnn_epochs = 300

    in_dim = train_data.x.shape[1]
    model = MultiScaleGraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gnn_epochs)
    loss_func = nn.SmoothL1Loss(reduction="none")

    train_data = train_data.to(device)

    model.train()
    for epoch in range(1, gnn_epochs + 1):
        optimizer.zero_grad()
        residual_pred, _ = model(train_data)
        loss_per_node = loss_func(residual_pred, train_data.y)
        sample_weights = train_data.confidence.clamp(min=0.05)
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
        inference_data_device = Data(
            x=inference_data.x,
            edge_index=inference_data.edge_index,
            base_pred=inference_data.base_pred,
        ).to(device)
        _, refined_pred = model(inference_data_device)

    output_df, metrics = _build_output_frame(
        inference_data=inference_data,
        refined_pred=refined_pred,
        final_output_csv=final_output_csv,
    )

    methods = ["GRU", "GNN(Refined)"]
    mae_values = [metrics["mae_gru"], metrics["mae_refined"]]
    mse_values = [metrics["mse_gru"], metrics["mse_refined"]]

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

        plot_df = output_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["True_Value", "GRU_Pred", "Refined_Pred"]
        )
        if not plot_df.empty:
            plt.scatter(plot_df["True_Value"], plot_df["GRU_Pred"], label="GRU", alpha=0.6)
            plt.scatter(
                plot_df["True_Value"],
                plot_df["Refined_Pred"],
                label="GNN",
                alpha=0.6,
            )
            min_val = np.nanmin(
                [
                    plot_df["True_Value"].min(),
                    plot_df["GRU_Pred"].min(),
                    plot_df["Refined_Pred"].min(),
                ]
            )
            max_val = np.nanmax(
                [
                    plot_df["True_Value"].max(),
                    plot_df["GRU_Pred"].max(),
                    plot_df["Refined_Pred"].max(),
                ]
            )
            plt.plot([min_val, max_val], [min_val, max_val], "k-", label="GROUND TRUTH")
            plt.xlabel("True Values")
            plt.ylabel("Predictions")
            plt.legend()
            plt.show()

    return output_df, metrics
