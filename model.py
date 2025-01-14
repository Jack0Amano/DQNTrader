import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 相場分類器

class MarketClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(MarketClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=4, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # conv1dの初期化
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        # lstmの初期化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)

    def forward(self, x, hidden_state=None):
        """
        x: [[[Upper_band, Lower_band, price_position, band_width], ...], ...] shape=(batch_size, sequence_length, input_dim)   
        price_position: (close - Lower_band) / (Upper_band - Lower_band) 現在の価格がボリンジャーバンドのどの位置にあるか
        """
        # x.shape = (batch, seq_len, input_size)
        # CNNの入力用に [batch, input_size, seq_len] に変換
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # LSTMの入力用に [batch, seq_len(Conv1dで小さくなった長さ), conv1.out_channels] に変換
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_state is not None:
            h_0, c_0 = hidden_state
            h_0 = h_0.to(x.device)
            c_0 = c_0.to(x.device)
        else:
            # 初期化
            h_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
            c_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        lstm_out, (h_0, c_0) = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # 最後の時系列ステップの出力
        output = self.fc(last_out)
        return output, (h_0, c_0)
    
# ボックス相場分析器
class BoxMarketAnalyzer(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2):
        super(BoxMarketAnalyzer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=16, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # conv1dの初期化
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        # lstmの初期化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)
    
    def forward(self, x, hidden_state=None):
        """
        x: (batch_size, sequence_length, input_dim)  
        return ボックス相場の上昇・下降確率
        """
        # x.shape = (batch, seq_len, input_size)
        # CNNの入力用に [batch, input_size, seq_len] に変換
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # LSTMの入力用に [batch, seq_len(Conv1dで小さくなった長さ), conv1.out_channels] に変換
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_state is not None:
            h_0, c_0 = hidden_state
            h_0 = h_0.to(x.device)
            c_0 = c_0.to(x.device)
        else:
            # 初期化
            h_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
            c_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        out, (h_0, c_0) = self.lstm(x)

        out_logit = self.fc(out[:, -1, :])
        return out_logit, out_logit, (h_0, c_0)
    
# トレンド相場分析器
class TrendMarketAnalyzer(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=128, num_layers=2):
        super(TrendMarketAnalyzer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=8, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # conv1dの初期化
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0)

        # lstmの初期化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0)
    
    def forward(self, x, hidden_state=None):
        """
        x: (batch_size, sequence_length, input_dim)   
        return トレンド方向の上昇・下降確率
        """
        # x.shape = (batch, seq_len, input_size)
        # CNNの入力用に [batch, input_size, seq_len] に変換
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # LSTMの入力用に [batch, seq_len(Conv1dで小さくなった長さ), conv1.out_channels] に変換
        x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_state is not None:
            h_0, c_0 = hidden_state
            h_0 = h_0.to(x.device)
            c_0 = c_0.to(x.device)
        else:
            # 初期化
            h_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
            c_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        out, (h_0, c_0) = self.lstm(x)
        out_logit = self.fc(out[:, -1, :])

        return out_logit, out_logit, (h_0, c_0)
    

# 相場によってモデルを切り替えるハイブリッドモデル
class HybridModel(nn.Module):
    def __init__(self, classifier_input_dim, box_input_dim, trend_input_dim):
        super(HybridModel, self).__init__()
        self.classifier = MarketClassifier(classifier_input_dim).to("cuda")
        self.box_model = BoxMarketAnalyzer(box_input_dim).to("cuda")
        self.trend_model = TrendMarketAnalyzer(trend_input_dim).to("cuda")

    def forward(self, classifier_x, box_x, trend_x, classifier_hidden_state=None, box_hidden_state=None, trend_hidden_state=None):
        """
        classifier: shape=(batch_size, sequence_length, input_dim)   
        box_x: shape=(batch_size, sequence_length, input_dim)   
        trend_x: shape=(batch_size, sequence_length, input_dim)   
        return トレンド方向の[横這い・上昇・下降確率]   
        """
        market_type, classifier_hidden_state = self.classifier(classifier_x, classifier_hidden_state)
        box_output, box_logit, box_hidden_state = self.box_model(box_x, box_hidden_state)
        trend_output, trend_logit, trend_hidden_state = self.trend_model(trend_x, trend_hidden_state)

        market_type_bool = market_type[:, 0] < 0.5
        market_type_bool = market_type_bool.view(-1, 1)
        output = torch.where(market_type_bool, box_output, trend_output)
        output_logit = torch.where(market_type_bool, box_logit, trend_logit)

        return output, output_logit, classifier_hidden_state, box_hidden_state, trend_hidden_state
        
    def test(self, classifier_input_dim, box_input_dim, trend_input_dim, sequence_length):
        batch = 8
        box_x = torch.randn(batch, sequence_length, box_input_dim).to("cuda")
        trend_x = torch.randn(batch, sequence_length, trend_input_dim).to("cuda")
        classifier_x = torch.randn(batch, sequence_length, classifier_input_dim).to("cuda")
        return self.forward(classifier_x, box_x, trend_x)

if __name__ == '__main__':
    model = HybridModel(4, 6, 4).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    output, logit, _, _, _ = model.test(4,6,4,30)
    criterion(logit, torch.randint(0, 3, (8,)).to("cuda"))
    print(output.shape)