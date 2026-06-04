import torch
import torch.nn as nn

class StockPriceRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=80, dropout=0.2):
        super(StockPriceRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x): # x shape: (batch, seq_len=60, input_size=1)
        # 如果输入是 (batch, seq_len) 的形状，补上 feature 维度变为 (batch, seq_len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.rnn1(x)  # 返回全部序列 (batch, seq_len=60, hidden_size=80)
        out = self.dropout(out) #
        out, _ = self.rnn2(out)  # 返回全部序列 (batch, seq_len=60, hidden_size=80)
        out = self.dropout(out)
        # out[:, -1, :]最后一层只取最后一个时间步(batch, hidden_size=80)
        out = self.fc(out[:, -1, :])
        return out