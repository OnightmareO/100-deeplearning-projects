import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

class StockDataset(Dataset):
    def __init__(self, data, seq_length=40):
        self.data = data
        self.seq_length = seq_length
        x, y = self.create_dataset(self.data, self.seq_length)
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def create_dataset(self, data, seq_length):
        x = []
        y = []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(x), np.array(y)

    
if __name__ == "__main__":
    data = pd.read_csv('./data/stock_data/SH600519.csv')
    print(data['open'].values)