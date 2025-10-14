import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
trip_data01 = pd.read_csv('yellow_tripdata_2020-01.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
trip_data02 = pd.read_csv('yellow_tripdata_2020-02.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
trip_data03 = pd.read_csv('yellow_tripdata_2020-03.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
trip_data04 = pd.read_csv('yellow_tripdata_2020-04.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
trip_data05 = pd.read_csv('yellow_tripdata_2020-05.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
trip_data06 = pd.read_csv('yellow_tripdata_2020-06.csv', parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], low_memory=False)
df = pd.concat([trip_data01, trip_data02, trip_data03, trip_data04, trip_data05, trip_data06], ignore_index=True)

# Data preprocessing
# Remove rows with PULocationID 264 or 265
df = df[~df['PULocationID'].isin([264, 265])]

# Extract date and hour from pickup datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['date'] = df['tpep_pickup_datetime'].dt.date
df['hour'] = df['tpep_pickup_datetime'].dt.hour
zoneid = 75

# Filter data for PULocationID = 75
df = df[df['PULocationID'] == zoneid]

# Group by date to get daily passenger count
daily_demand = df.groupby(['date']).size().reset_index(name='passenger_count')

# Scaling data
scaler = MinMaxScaler()
daily_demand['passenger_count_scaled'] = scaler.fit_transform(daily_demand[['passenger_count']])

# Prepare data for LSTM
sequence_length = 5
X, y = [], []
for i in range(len(daily_demand) - sequence_length):
    X.append(daily_demand['passenger_count_scaled'].iloc[i:i + sequence_length].values)
    y.append(daily_demand['passenger_count_scaled'].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last time step's output
        return x

# Model parameters
input_size = 1
hidden_size = 32
output_size = 1
num_layers = 1

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 100
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor.to(device))
    loss = criterion(output, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device))
    predictions = predictions.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

# Inverse scaling to compare with actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted', color='b')
plt.plot(y_test, label='Actual', color='r')
plt.xlabel('Sample Index')
plt.ylabel('Passenger Count')
plt.title('Predicted vs Actual Passenger Count for Zone 75')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
