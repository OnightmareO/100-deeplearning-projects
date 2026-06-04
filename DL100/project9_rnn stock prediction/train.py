import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data import StockDataset
from model import StockPriceRNN
# 完成股票开盘价格的预测
df = pd.read_csv('./data/stock_data/SH600519.csv')
data = df['open'].values

train_size = int(len(data) * 0.8)
train_raw = data[:train_size]
test_raw = data[train_size:]

# 保存训练集的 min/max 以便反归一化
min_val = train_raw.min()
max_val = train_raw.max()

# 训练集：学习数据规律 + 归一化
train_data = (train_raw - min_val) / (max_val - min_val)
# 测试集：评估模型性能 + 归一化（使用训练集的最小值和最大值）
test_data = (test_raw - min_val) / (max_val - min_val)

train_dataset = StockDataset(train_data)
test_dataset = StockDataset(test_data)

'''
#检查数据集
print(f'Length of the train dataset: {len(train_dataset)}')
print(f'Length of the test dataset: {len(test_dataset)}')
sel_idx = 10
x, y = train_dataset[sel_idx]
print(f'Input: {x}')
print(f'Output: {y}')
'''

# 每个滑动窗口当作独立样本，batch_size=16，训练集打乱，测试集不打乱
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

'''
#检查数据加载器和数据集
train_dataset = train_loader.dataset
test_dataset = test_loader.dataset

print('=== Train Loader ===')
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"train_dataset type: {type(train_dataset)}")

print('\n=== Test Loader ===')
print(f"Number of batches in test_loader: {len(test_loader)}")
print(f"Number of samples in test_dataset: {len(test_dataset)}")
print(f"test_dataset type: {type(test_dataset)}")
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockPriceRNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch) # 输出形状 (batch_size, 1)
        #y_batch 形状 (batch_size,)，因此outputs需要squeeze()以匹配形状
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)# 输出形状 (batch_size, 1)
        predictions.extend(outputs.squeeze().cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

predictions = np.array(predictions)
actuals = np.array(actuals)

# 反归一化（min-max 逆变换）
predictions = predictions * (max_val - min_val) + min_val
actuals = actuals * (max_val - min_val) + min_val

mse = np.mean((predictions - actuals) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - actuals))
R2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

print(f'Test MSE: {mse:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Test MAE: {mae:.4f}')
print(f'Test R2: {R2:.4f}')