import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文字体支持
plt.rcParams['font.family'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# -------------------------
# 1. 数据加载与预处理
# -------------------------

# 读取2020年1月至6月的数据
files = [f'yellow_tripdata_2020-0{i}.csv' for i in range(1, 7)]
df_list = [pd.read_csv(file, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False) for file in files]
df = pd.concat(df_list, ignore_index=True)

# 移除PULocationID为264或265的行
df = df[~df['PULocationID'].isin([264, 265])]

# 提取日期和小时信息
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['datetime'] = df['tpep_pickup_datetime'].dt.floor('H')  # 保持为小时级别的 Timestamp 类型

# 选择特定的区域ID
zoneid = 75
df = df[df['PULocationID'] == zoneid]

# 按datetime分组，计算每小时的乘客数量
hourly_demand = df.groupby(['datetime']).size().reset_index(name='passenger_count')

# 按datetime排序
hourly_demand = hourly_demand.sort_values('datetime').reset_index(drop=True)

# -------------------------
# 2. 数据缩放
# -------------------------

# 使用MinMaxScaler进行数据缩放
scaler = MinMaxScaler()
hourly_demand['passenger_count_scaled'] = scaler.fit_transform(hourly_demand[['passenger_count']])

# -------------------------
# 3. 准备LSTM模型的数据
# -------------------------

sequence_length = 24  # 使用过去24小时的数据预测下一小时
X, y, y_datetimes = [], [], []

for i in range(len(hourly_demand) - sequence_length):
    X.append(hourly_demand['passenger_count_scaled'].iloc[i:i + sequence_length].values)
    y.append(hourly_demand['passenger_count_scaled'].iloc[i + sequence_length])
    y_datetimes.append(hourly_demand['datetime'].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)
y_datetimes = np.array(y_datetimes)
y_datetimes = pd.to_datetime(y_datetimes)  # 将 y_datetimes 转换为 Timestamp

# 划分训练集和测试集
split_datetime = pd.to_datetime('2020-06-01 00:00:00')  # 保持为 Timestamp

# 创建布尔索引数组
train_indices = y_datetimes < split_datetime
test_indices = y_datetimes >= split_datetime

# 划分数据
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
test_datetimes = y_datetimes[test_indices]

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# -------------------------
# 5. 定义LSTM模型
# -------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # 使用最后一个时间步的输出
        return x

# -------------------------
# 6. 训练模型
# -------------------------

# 模型参数
input_size = 1
hidden_size = 64  # 增加隐藏层大小，提高模型能力
output_size = 1
num_layers = 2  # 增加LSTM层数

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 20  # 由于数据量较大，可以适当减少训练轮数
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor.to(device))
    loss = criterion(output, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'第 {epoch + 1} 次迭代，损失值: {loss.item():.4f}')

# -------------------------
# 7. 测试模型
# -------------------------

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device))
    predictions = predictions.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

# 逆缩放以恢复原始值
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# -------------------------
# 8. 评估模型性能
# -------------------------

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f'均方误差（MSE）：{mse:.2f}')
print(f'平均绝对误差（MAE）：{mae:.2f}')
print(f'均方根误差（RMSE）：{rmse:.2f}')

# -------------------------
# 9. 可视化预测结果
# -------------------------

plt.figure(figsize=(15, 6))
plt.plot(test_datetimes, predictions, label='预测值', color='b')
plt.plot(test_datetimes, y_test, label='实际值', color='r')
plt.xlabel('日期时间')
plt.ylabel('乘客数量')
plt.title('区域75的乘客数量预测与实际值对比（2020年6月）')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
