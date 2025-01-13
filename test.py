import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, stride=1)
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.conv1(x.permute(0, 2, 1))  # 入力を [batch, input_size, seq_len] に変更
        x = torch.relu(x).permute(0, 2, 1)  # Conv1Dの出力をLSTMの入力に変換
        print(x.shape)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    
if __name__ == '__main__':
    model = CNNLSTM(4, 32, 2)
    x = torch.rand(10, 100, 4)
    y = model(x)
    print(y.shape)