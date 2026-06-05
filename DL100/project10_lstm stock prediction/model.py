import torch
import torch.nn as nn

class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(StockPriceLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 2
        #self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=self.num_layers, 
                             batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x): # x shape: (batch, seq_len=40, input_size=1)
        # 如果输入是 (batch, seq_len) 的形状，补上 feature 维度变为 (batch, seq_len, 1)r
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)   # 返回全部序列 (batch, seq_len=40, hidden_size=50)
        # out[:, -1, :]最后一层只取最后一个时间步(batch, hidden_size=50)
        out = self.fc(out[:, -1, :])
        return out
