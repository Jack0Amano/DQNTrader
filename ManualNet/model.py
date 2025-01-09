import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import talib as ta
import CommonAnalyzer as ca
import enum
import keyboard

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TrendModel(nn.Module):
    """
    入力されたデータからWindowのサイズ分のデータの傾向を予測するモデル   
    """
    def __init__(self):
        super(TrendModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class TrendModelAgent:
    """
    トレンドモデルを使って学習と予測を行うエージェント
    """

    def __init__(self, model, window_size):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.data = []
        self.target = []
        self.loss = 0
