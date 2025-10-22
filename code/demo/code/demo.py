import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
# device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)




target_date = pd.Timestamp('2021-03-05 12:00')  # 目标时间点




import torch
import torch.nn as nn

class MultiScaleModel(nn.Module):
    def __init__(self, hidden_size):
        super(MultiScaleModel, self).__init__()

        self.hidden_size = hidden_size

        # Define LSTM (for daily & weekly patterns)
        self.lstm_1d = nn.LSTM(1, hidden_size, batch_first=True)
        self.lstm_1w = nn.LSTM(1, hidden_size, batch_first=True)

        # Define Transformer (for monthly trends)
        self.input_projection = nn.Linear(1, hidden_size)  # Projects input to match hidden size
        self.transformer_1m = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=2, batch_first=True)

        # Feature fusion layer (Combining LSTM + Transformer outputs)
        self.feature_fusion = nn.Linear(hidden_size * 3, hidden_size)

        # Final GRU (now receives raw 1-hour sequence + fused long-term trends)
        self.gru = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)

        # Final prediction layer (Forecasts next hour demand)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Extract different time-scale features
        x_1h, x_1d, x_1w, x_1m = x["1h"], x["1d"], x["1w"], x["1m"]

        # Process 1-day & 1-week data with LSTMs
        _, (h_1d, _) = self.lstm_1d(x_1d)
        h_1d = h_1d[-1]  # Take the last hidden state

        _, (h_1w, _) = self.lstm_1w(x_1w)
        h_1w = h_1w[-1]  # Take the last hidden state

        # Process 1-month data with Transformer
        x_1m = self.input_projection(x_1m)
        x_1m = x_1m.permute(1, 0, 2)  # Adjust dimensions for Transformer
        h_1m = self.transformer_1m(x_1m, x_1m)[-1]

        # Fuse LSTM and Transformer outputs
        fused_trend = torch.cat([h_1d, h_1w, h_1m], dim=1)
        fused_trend = self.feature_fusion(fused_trend)  # Reduce dimensionality to hidden_size

        # Prepare GRU input (1-hour data + fused long-term trend)
        batch_size, seq_len, _ = x_1h.shape
        fused_trend_expanded = fused_trend.unsqueeze(1).repeat(1, seq_len, 1)  # Expand to match sequence length
        x_gru_input = torch.cat([x_1h, fused_trend_expanded], dim=2)  # [batch, seq_len, hidden_size + 1]

        # Process with GRU
        _, h_gru = self.gru(x_gru_input)
        h_gru = h_gru[-1]  # Take last hidden state

        # Final Prediction
        output = self.fc(h_gru)
        return output


# month = '01'
file_name = f'data.parquet'
columns_to_load = ['pickup_datetime', 'PULocationID']
df_temp = pd.read_parquet(file_name, columns=columns_to_load)


# 过滤指定的区域
excluded_zones = [103, 104, 105, 46, 264, 265]
df_temp = df_temp[~df_temp['PULocationID'].isin(excluded_zones)]
zoneTotalNumber = len(df_temp['PULocationID'].unique())
print('-----------------------------------------------------')
print(len(df_temp['PULocationID'].unique()))
print('-----------------------------------------------------')

df_temp['pickup_datetime'] = pd.to_datetime(df_temp['pickup_datetime'])
df_temp['datetime'] = df_temp['pickup_datetime'].dt.floor('H')

df = df_temp


# durations = {
#     '1h': 1,
#     '1d': 24,
#     '1w': 24 * 7,
#     '1m': 24 * 30
# }
# forecast_length = 1
#
# sequence_length = durations['1m']
# sf = sequence_length + forecast_length
#
#
# global_predictions = []
# global_true_values = []
#
#
# start_date = target_date - pd.Timedelta(hours=sequence_length)
#
# # 过滤指定时间范围的数据（不包括 target_date）
# df = df[(df['datetime'] >= start_date) & (df['datetime'] <= target_date)]

durations = {
    '1h': 1,
    '1d': 24,
    '1w': 24 * 7,
    '1m': 24 * 30
}
forecast_length = 1
sequence_length = durations['1m']
sf = sequence_length + forecast_length

global_predictions = []
global_true_values = []

# 目标：输入区间为 [2021-02-03 11:00, 2021-03-05 11:00]
# 原来是 target_date - 720h 得到 2021-02-03 12:00，现在往前多挪 1 小时
start_date = target_date - pd.Timedelta(hours=sequence_length + 1)
print(start_date)  # 将打印 2021-02-03 11:00:00

# 过滤指定时间范围（仍然不包含 target_date=2021-03-05 12:00）
# 这样就会包含到 2021-03-05 11:00 为止
df = df[(df['datetime'] >= start_date) & (df['datetime'] < target_date)]


skipZoneList = []
# 初始化结果列表
results = []



# 检查过滤后的数据
print(f"Filtered data from {start_date} to {target_date - pd.Timedelta(hours=1)}. Total rows: {len(df)}")
minGlobalDate = df['datetime'].min()
maxGlobalDate = df['datetime'].max()
# 遍历所有的 PULocationID
unique_zones = df['PULocationID'].unique()
count = len(df['PULocationID'].unique())
for zoneid in unique_zones:
    print(f"Processing PULocationID: {zoneid}")
    print(f"There are {count} zones left")
    count -= 1
    zone_df = df[df['PULocationID'] == zoneid]

    # 按小时聚合数据
    hourly_demand = zone_df.groupby('datetime').size().reset_index(name='passenger_count')
    hourly_demand = hourly_demand.sort_values('datetime').reset_index(drop=True)




    # 数据归一化

    # max_day = hourly_demand['datetime'].max().day
    scaler = MinMaxScaler()
    hourly_demand['passenger_count_scaled'] = scaler.fit_transform(hourly_demand[['passenger_count']])

    print(len(hourly_demand))

    if len(hourly_demand) < sf/2:
        print("missing critical data!")
        skipZoneList.append(zoneid)
        continue
    # if target_date not in hourly_demand['datetime'].values:
    #     missing_row = pd.DataFrame({'datetime': [target_date], 'passenger_count': [0]})
    #     hourly_demand = (
    #         pd.concat([hourly_demand, missing_row], ignore_index=True)
    #         .sort_values('datetime')
    #         .reset_index(drop=True)
    #     )
    print(f"Length of hourly_demand: {len(hourly_demand)}")
    print(f"Required sf: {sf}")
    if len(hourly_demand) < sf:
        full_datetime_range = pd.date_range(start=minGlobalDate, end=target_date, freq='H')
        hourly_demand = (
            hourly_demand.set_index('datetime')
            .reindex(full_datetime_range)
            .fillna(0)
            .reset_index()
            .rename(columns={'index': 'datetime'})
        )



        # 生成完整时间范围
        full_datetime_range = pd.date_range(
            start=minGlobalDate,
            end=target_date,
            freq='H'
        )

        # 找出缺失的时间点
        existing_times = set(hourly_demand['datetime'])
        full_times = set(full_datetime_range)
        missing_times = sorted(full_times - existing_times)
        print(len(existing_times))
        print(len(full_times))
        print(len(missing_times))

        # print("Missing times:")
        # for t in missing_times:
        #     print(t)

        # print("Before filling missing data:")
        # print(f"hourly_demand shape: {hourly_demand.columns}")
        # print(hourly_demand.tail())  # 打印前几行数据，检查内容
        print(f"hourly_demand columns before fill: {hourly_demand.shape}")

        # 填充数据
        hourly_demand = hourly_demand.set_index('datetime').reindex(full_datetime_range).fillna(0).reset_index()
        hourly_demand.rename(columns={'index': 'datetime'}, inplace=True)  # 确保列名正确

        # print("After filling missing data:")
        # print(f"hourly_demand shape: {hourly_demand.columns}")
        # print(hourly_demand.tail())
        print(f"hourly_demand columns after fill: {hourly_demand.shape}")

        # 确保归一化列存在
        hourly_demand['passenger_count_scaled'] = scaler.fit_transform(hourly_demand[['passenger_count']])

    X_1h, X_1d, X_1w, X_1m, y = [], [], [], [], []

    for i in range(len(hourly_demand) + 1 - sequence_length - forecast_length):
        # 获取不同时间尺度的数据
        x_1h = hourly_demand['passenger_count_scaled'].iloc[
               i + sequence_length - durations['1h']:i + sequence_length].values  # 1小时数据
        x_1d = hourly_demand['passenger_count_scaled'].iloc[
               i + sequence_length - durations['1d']:i + sequence_length].values  # 1天数据
        x_1w = hourly_demand['passenger_count_scaled'].iloc[
               i + sequence_length - durations['1w']:i + sequence_length].values  # 1周数据
        x_1m = hourly_demand['passenger_count_scaled'].iloc[i:i + sequence_length].values  # 1个月数据

        # **不同时间尺度可以有不同的长度，无需统一**
        X_1h.append(x_1h)
        X_1d.append(x_1d)
        X_1w.append(x_1w)
        X_1m.append(x_1m)

        # 目标值（未来1小时预测）
        y_val = hourly_demand['passenger_count_scaled'].iloc[
                i + sequence_length:i + sequence_length + forecast_length].values
        if len(y_val) == forecast_length:
            y.append(y_val)

    # **转换为 NumPy 数组**
    X_1h = np.array(X_1h, dtype=np.float32)
    X_1d = np.array(X_1d, dtype=np.float32)
    X_1w = np.array(X_1w, dtype=np.float32)
    X_1m = np.array(X_1m, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # **转换为 PyTorch Tensor**
    X_1h_tensor = torch.tensor(X_1h, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, sequence_length, 1)
    X_1d_tensor = torch.tensor(X_1d, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, sequence_length, 1)
    X_1w_tensor = torch.tensor(X_1w, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, sequence_length, 1)
    X_1m_tensor = torch.tensor(X_1m, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, sequence_length, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)  # (batch, forecast_length)

    # **打包为字典，方便传递**
    X_tensor = {
        "1h": X_1h_tensor,
        "1d": X_1d_tensor,
        "1w": X_1w_tensor,
        "1m": X_1m_tensor
    }

    print(f"X_1h_tensor shape: {X_1h_tensor.shape}")  # (batch, sequence_length, 1)
    print(f"X_1d_tensor shape: {X_1d_tensor.shape}")  # (batch, sequence_length, 1)
    print(f"X_1w_tensor shape: {X_1w_tensor.shape}")  # (batch, sequence_length, 1)
    print(f"X_1m_tensor shape: {X_1m_tensor.shape}")  # (batch, sequence_length, 1)
    print(f"y_tensor shape: {y_tensor.shape}")  # (batch, forecast_length)






    # 模型定义
    # 模型定义
    hidden_size = 64
    model = MultiScaleModel(hidden_size).to(device)  # 转移模型到 GPU/CPU
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 转移数据到 GPU/CPU
    # 变成 PyTorch Tensor，并存入字典
    X_tensor = {
        "1h": torch.tensor(X_1h, dtype=torch.float32).unsqueeze(-1).to(device),  # (batch, sequence_length, 1)
        "1d": torch.tensor(X_1d, dtype=torch.float32).unsqueeze(-1).to(device),  # (batch, sequence_length, 1)
        "1w": torch.tensor(X_1w, dtype=torch.float32).unsqueeze(-1).to(device),  # (batch, sequence_length, 1)
        "1m": torch.tensor(X_1m, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, sequence_length, 1)
    }

    # 目标值仍然保持原格式
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)  # (batch, forecast_length)

    # **打印检查**
    print(f"X_1h_tensor shape: {X_tensor['1h'].shape}")  # (batch, sequence_length, 1)
    print(f"X_1d_tensor shape: {X_tensor['1d'].shape}")  # (batch, sequence_length, 1)
    print(f"X_1w_tensor shape: {X_tensor['1w'].shape}")  # (batch, sequence_length, 1)
    print(f"X_1m_tensor shape: {X_tensor['1m'].shape}")  # (batch, sequence_length, 1)
    print(f"y_tensor shape: {y_tensor.shape}")  # (batch, forecast_length)

    # 模型训练
    epochs = 50
    patience = 5
    best_loss = float('inf')
    counter = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 模型评估
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()  # 将预测值转移到 CPU
        true_values = y_tensor.cpu().numpy()  # 将真实值转移到 CPU

    # 逆缩放
    predictions = scaler.inverse_transform(predictions)
    true_values = scaler.inverse_transform(true_values)
    global_predictions.extend(predictions.flatten().tolist())
    global_true_values.extend(true_values.flatten().tolist())
    # 保存结果
    for pred, true_val in zip(predictions, true_values):
        results.append({'PULocationID': zoneid, 'Prediction': pred[0], 'True Value': true_val[0]})




print(skipZoneList)
print(f"length{len(skipZoneList)}")
for skipped_id in skipZoneList:
    results.append({'PULocationID': skipped_id, 'Prediction': None, 'True Value': None})

# 计算全局平均 MAE 和 MSE
if global_predictions and global_true_values:
    global_predictions = np.array(global_predictions)
    global_true_values = np.array(global_true_values)
    global_mae = np.mean(np.abs(global_predictions - global_true_values))
    global_mse = np.mean((global_predictions - global_true_values) ** 2)
    print(f"Overall Average MAE: {global_mae:.4f}")
    print(f"Overall Average MSE: {global_mse:.4f}")
else:
    print("No valid predictions available for computing global metrics.")


# 加载结果 CSV
results_df = pd.DataFrame(results)
# output_file = f'GRU_{zoneTotalNumber-len(skipZoneList)}Zones_{epochs}epochs.csv'
# results_df.to_csv(output_file, index=False)

# print(f"Results saved to {output_file}")
# 加载 lookup table 以映射 Zone 名称
lookup_table = pd.read_csv('taxi-zone-lookup.csv')

# 将 PULocationID 映射到 Zone 名称
results_df = pd.merge(
    results_df,
    lookup_table[['LocationID', 'Zone']],
    left_on='PULocationID',
    right_on='LocationID',
    how='left'
)

# 按 Zone 名称分组，合并相同的 Zone
merged_results = results_df.groupby('Zone', as_index=False).agg({
    'PULocationID': 'first',  # 保留第一个 PULocationID 作为参考
    'Prediction': 'mean',    # 计算 Prediction 的平均值
    'True Value': 'mean'     # 计算 True Value 的平均值
})

# 保存合并后的结果
output_file = f'GRU_Merged_{epochs}epochs.csv'
merged_results.to_csv(output_file, index=False)
print(f"Merged results saved to {output_file}")


if not results_df.empty:
    results_df = results_df.dropna(subset=['Prediction', 'True Value'])
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Prediction'], label='Prediction', linestyle='--', marker='o')
    plt.plot(results_df['True Value'], label='True Value', linestyle='-', marker='x')
    plt.title('Prediction vs True Value')
    plt.xlabel('Taxi_zone ID')
    plt.ylabel('Passenger Demand')
    plt.legend()
    plt.grid()
    plt.show()

else:
    print("No data available for plotting.")

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import random

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
# ============================
# 1. 加载初始邻接矩阵
# ============================
edge_weight_csv = "edge_weight_matrix_with_flow.csv"
df_adj = pd.read_csv(edge_weight_csv, index_col=0)
adj_matrix = torch.tensor(df_adj.values, dtype=torch.float32)  # 稠密矩阵

# 转为稀疏图
edge_index, edge_attr = dense_to_sparse(adj_matrix)




# 提取区域名称和索引映射
zone_names = df_adj.index.tolist()
zone_idx_map = {zone: idx for idx, zone in enumerate(zone_names)}
N = len(zone_names)

# ============================
# 2. 统计 County Code 过去30天的总流量并更新节点权重
# ============================




file_name = "data.parquet"
columns_to_load = ['pickup_datetime', 'PULocationID', 'DOLocationID']
# df = pd.read_parquet(file_name, columns=columns_to_load)
df = df_temp
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
# df['datetime'] = df['pickup_datetime'].dt.floor('H')




# 过滤过去 30 天的数据
sequence_length = 24 * 30
start_date = target_date - pd.Timedelta(hours=sequence_length)
df_30_days = df[(df['datetime'] >= start_date) & (df['datetime'] <= target_date)]

# excluded_zones = [103, 104, 105, 46, 264, 265]
# df = df[~df['PULocationID'].isin(excluded_zones)]
# zoneTotalNumber = len(df['PULocationID'].unique())
print(f"Total unique zones: {zoneTotalNumber}")

# 筛选前一个小时的数据，统计 True Value
previous_hour = target_date - pd.Timedelta(hours=1)  # 前一个小时
previous_hour_data = df[df['datetime'] == previous_hour]


true_values_dict = {}
for pulocation_id, group in previous_hour_data.groupby('PULocationID'):
    true_values_dict[pulocation_id] = len(group)  # 统计每个区域的乘客数量作为 True Value

# 定义 County Code 到 Borough 的映射
county_code_to_borough = {
    1: "Bronx",
    2: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
    6: "Manhattan"
}

# 加载 taxi-zone-lookup.csv 并建立 LocationID 到 County Code 的映射
zone_lookup_file = "taxi-zone-lookup.csv"
zone_lookup_df = pd.read_csv(zone_lookup_file).drop_duplicates(subset="LocationID")
location_to_county_code = dict(zip(zone_lookup_df["LocationID"], zone_lookup_df["Borough"].map(
    lambda x: next((k for k, v in county_code_to_borough.items() if v == x), None)
)))



# 按 PULocationID 统计过去30天的总流量，并映射到 County
county_volume = df_30_days.groupby("PULocationID").size().reset_index(name="Total_Volume")
county_volume["County"] = county_volume["PULocationID"].map(location_to_county_code)
county_total_volume = county_volume.groupby("County")["Total_Volume"].sum().to_dict()





# 初始化节点权重矩阵
node_weights = torch.zeros((N,), dtype=torch.float32)

zone_to_county = dict(zip(zone_lookup_df["Zone"], zone_lookup_df["Borough"].map(
    lambda x: next((k for k, v in county_code_to_borough.items() if v == x), None)
)))

for zone, idx in zone_idx_map.items():
    county_code = zone_to_county.get(zone)  # Map zone to county name first
    if county_code and county_code in county_total_volume:
        node_weights[idx] = county_total_volume[county_code]
    else:
        print(f"⚠️ Missing county volume data for Zone {zone} (County: {county_code}), setting weight to 0")
        node_weights[idx] = 0  # Prevent NaN values

# 归一化节点权重到 [0, 1]
node_weights = node_weights / node_weights.max()

# 输出每个节点的权重
print("Node weights:", node_weights)





# 读取预测数据
pred_csv = "GRU_Merged_50epochs.csv"
df_pred = pd.read_csv(pred_csv)

location_to_zone = dict(zip(zone_lookup_df["LocationID"], zone_lookup_df["Zone"]))

df_pred["Zone"] = df_pred["PULocationID"].map(location_to_zone)

node_pred = torch.full((N,), float('nan'), dtype=torch.float32)
node_label = torch.full((N,), float('nan'), dtype=torch.float32)

for _, row in df_pred.iterrows():
    loc_id = int(row["PULocationID"])
    if loc_id in excluded_zones:
        continue
    zone_str = row["Zone"]
    pred_val = row["Prediction"]
    true_val = row["True Value"]
    if isinstance(zone_str, str) and zone_str in zone_idx_map:
        ridx = zone_idx_map[zone_str]
        node_pred[ridx] = float(pred_val)
        node_label[ridx] = float(true_val)


# 检查 NaN 数量
print("node_pred NaN count:", torch.isnan(node_pred).sum().item())
print("node_label NaN count:", torch.isnan(node_label).sum().item())




# 找到非 NaN 的索引
valid_indices = torch.where(~torch.isnan(node_pred) & ~torch.isnan(node_label))[0]
print(f"⚠️ 仅保留有效索引数量: {valid_indices.numel()}")

zone_names = [zone_names[i] for i in valid_indices.tolist()]


# 重新创建 old_index -> new_index 的映射
old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(valid_indices)}

# 仅保留 edge_index 里仍然存在于 valid_indices 的节点
valid_edges = (torch.isin(edge_index[0], valid_indices)) & (torch.isin(edge_index[1], valid_indices))
edge_index = edge_index[:, valid_edges]


# 重新映射 edge_index 确保索引匹配新数据
edge_index = torch.tensor([[old_to_new[i.item()] for i in edge_index[0]],
                           [old_to_new[j.item()] for j in edge_index[1]]], dtype=torch.long)

node_pred = node_pred[valid_indices]
node_label = node_label[valid_indices]
node_weights = node_weights[valid_indices]


# 组合 node_pred 和 node_weights，形成 2 维特征
x_feat = torch.stack([node_pred, node_weights], dim=1)  # shape [N, 2]
data = Data(x=x_feat, edge_index=edge_index)
data.y = node_label  # shape [N]

do=0.1
lr=0.01
hd=256
epochs = 300
class MultiScaleGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(do)  # 添加 Dropout 防止过拟合
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # **GraphSAGE 不需要 edge_weight**
        x1 = self.dropout(F.gelu(self.sage1(x, edge_index)))
        x2 = self.dropout(F.gelu(self.sage2(x1, edge_index)))
        return self.out_linear(x2).squeeze(-1)


# 训练模型
model = MultiScaleGraphSAGE(in_dim=2, hidden_dim=hd).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.SmoothL1Loss()
# 定义 余弦退火策略
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

data = data.to(device)

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    pred = model(data)
    loss = loss_func(pred, data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        # print(f"Epoch {epoch}/{epochs}, Loss = {loss.item():.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs}, Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
# 验证模型
model.eval()
with torch.no_grad():
    refined_pred = model(data)

y_true = data.y.cpu().numpy()
y_refined = refined_pred.cpu().numpy()
node_pred_val = data.x.squeeze(-1).cpu().numpy()





# -----------------------------
# 8) 输出结果
# -----------------------------
# 确保所有数组是 1 维
node_pred_val = data.x[:, 0].cpu().numpy().squeeze()
y_refined = y_refined.squeeze()
y_true = y_true.squeeze()

print("Final shapes:")
print("node_pred_val shape:", node_pred_val.shape)
print("y_refined shape:", y_refined.shape)
print("y_true shape:", y_true.shape)

# 创建 DataFrame
output_df = pd.DataFrame({
    "ZoneName": zone_names,
    "GRU_Pred": node_pred_val,
    "Refined_Pred": y_refined,
    "True_Value": y_true
})

gru_pred = output_df["GRU_Pred"].values
refined_pred = output_df["Refined_Pred"].values
true_val = output_df["True_Value"].values

mae_gru = np.mean(np.abs(gru_pred - true_val))
mae_refined = np.mean(np.abs(refined_pred - true_val))

mse_gru = np.mean((gru_pred - true_val)**2)
mse_refined = np.mean((refined_pred - true_val)**2)

print(f"GRU vs True MAE = {mae_gru:.4f}")
print(f"GRU vs True MSE = {mse_gru:.4f}")
print(f"Refined vs True MAE = {mae_refined:.4f}")
print(f"Refined vs True MSE = {mse_refined:.4f}")

import matplotlib.pyplot as plt



# 误差数据
methods = ["GRU","GNN(Refined)"]
mae_values = [mae_gru,  mae_refined]

# 绘制柱状图
plt.figure(figsize=(6, 5))
plt.bar(methods, mae_values, color=['blue','orange'], alpha=0.7)

# 添加数值标注
for i, v in enumerate(mae_values):
    plt.text(i, v + 0.2, f"{v:.2f}", ha='center', fontsize=12)

plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Comparison of Prediction Errors (MAE)")
# plt.ylim(5, 15)  # 让差距更加明显
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 误差数据
mse_values = [mse_gru,  mse_refined]

# 绘制柱状图
plt.figure(figsize=(6, 5))
plt.bar(methods, mse_values, color=['blue','orange'], alpha=0.7)

# 添加数值标注
for i, v in enumerate(mse_values):
    plt.text(i, v + 0.2, f"{v:.2f}", ha='center', fontsize=12)

plt.ylabel("Mean Absolute Error (MSE)")
plt.title("Comparison of Prediction Errors (MSE)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


plt.scatter(true_val, gru_pred, label="GRU", alpha=0.6)
plt.scatter(true_val, refined_pred, label="GNN", alpha=0.6)

min_val = min(true_val.min(), gru_pred.min(), refined_pred.min())
max_val = max(true_val.max(), gru_pred.max(), refined_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k-', label="GROUND TRUTH")
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.legend()
plt.show()

output_df.to_csv(
    "final_predictions_multiscale.csv",
    columns=["ZoneName", "GRU_Pred", "Refined_Pred", "True_Value"],
    index=False,
    encoding="utf-8"
)
print("result saved to 'final_predictions_multiscale.csv'")
