import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Qネットワーク定義
class LSTMNetwork(nn.Module):
    def __init__(self, state_channel_size, sequence_length, action_size):
        """
        Qネットワークのコンストラクタ   
        input_size: 状態の次元数 [[close(-sequence_length), ... close(current)],    
                                    [profit(-sequence_length), ... profit(current)]] shape=(2, sequence_length) 
                                    の場合input_size=3
        action_size: 行動の数 [Nothing, BUY, SELL, CHECKOUT]
        """
        super(LSTMNetwork, self).__init__()
        self.conv1 = nn.Conv1d(sequence_length, 128, kernel_size=3, stride=4)
        self.conv2 = nn.Conv1d(128, 32, kernel_size=1, stride=2)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=1,
            hidden_size=action_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True)
        self.output = nn.Linear(8, action_size)
        self.hidden = None

    def forward(self, x, hidden_state=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        batch_size = x.size(0)
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_state is not None:
            h_0, c_0 = hidden_state
        else:
            # 初期化
            h_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
            c_0 = torch.zeros(num_directions * self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        x, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = x[:, -1, :]
        x = self.output(x)
        return x, (h_n, c_n)

    
    def reset_hidden(self):
        """
        隠れ状態をリセットするメソッド   
        """
        self.hidden = None
    
    # Random入力してテストする
    def test(self, input_size, sequence_length, action_size):
        x = torch.randn(64, sequence_length, input_size)
        output = self.forward(x)
        return output

# 優先度付き経験再生バッファの定義
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # バッファの容量
        self.memory = []  # 経験を保存するリスト

    # 経験をバッファに追加
    def add(self, state, action, label, profit):
        experience = (state, action, label, profit)
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        メモリの先頭からバッチサイズ分の経験を取り出すメソッド
        """
        states, actions, label_q_values, profits = zip(*self.memory)
        # 古い経験から優先してサンプリング
        states = np.array(states)
        actions = np.array(actions)
        label_q_values = np.array(label_q_values)
        profits = np.array(profits)
        states = states[0:batch_size]
        actions = actions[0:batch_size]
        label_q_values = label_q_values[0:batch_size]
        return states, actions, label_q_values, profits

    def clear(self, batch_size):
        """
        バッチサイズで指定された分のメモリを先頭から削除するメソッド
        """
        self.memory = self.memory[batch_size:]

    def __len__(self):
        return len(self.memory)
    
# DQNエージェントの定義
class LSTMAgent:
    def __init__(
        self,
        state_channel_size = 3,
        sequence_length = 10,
        action_size = 4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        beta_start=0.4,
        beta_frames=100000,
        label_sequence_length=10
    ):
        self.state_channel_size = state_channel_size  # 状態の次元数
        self.action_size = action_size  # 行動の数
        # 優先度付き経験再生バッファを初期化
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.gamma = gamma  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率の減少率
        self.epsilon_min = 0.05  # 最小探索率
        self.learning_rate = learning_rate  # 学習率
        self.beta_start = beta_start  # 優先度付き経験再生の初期β値
        self.beta_frames = beta_frames  # β値が1に達するまでのフレーム数
        # ポリシーネットワークのインスタンス
        self.model = LSTMNetwork(state_channel_size, sequence_length, action_size)
        # Adamオプティマイザを使用してネットワークのパラメータを最適化
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # modelのhidden_stateを保持するための変数
        self.model_hidden = None
        self.batch_size = batch_size
        self.label_sequence_length = label_sequence_length
        self.criterion = nn.MSELoss()

    # 経験をメモリに追加
    def remember(self, state, action, label_q_values, profit):
        self.memory.add(state, action, label_q_values, profit)

    # 行動を選択
    def action(self, state):
        """
        行動を選択するメソッド   
        state: 現在の状態 [[close(-sequence_length), ... close(current)],    
                                    [profit(-sequence_length), ... profit(current)],   
                                    [order_type(-sequence_length), ... order_type(current)]] shape=(3, sequence_length) 
                                    の場合input_size=3
        """
        # 探索かネットワークの予測に基づいて行動を選択するか決定
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # ランダムに行動を選択
        state = torch.FloatTensor(state).unsqueeze(0)  # 状態をテンソルに変換してネットワークに入力
        with torch.no_grad():  # 勾配計算を無効化
            act_values, (h_n, c_n) = self.model(state)  # 各行動のQ値を計算
        self.model_hidden = (h_n.detach(), c_n.detach())
        return torch.argmax(act_values).item()  # 最大のQ値を持つ行動を選択

    # 経験をリプレイしてネットワークを訓練
    def replay(self):
        if len(self.memory) < self.batch_size + self.label_sequence_length:
            return False  # メモリが十分に溜まるまではリプレイを実行しない
        
        # メモリから優先度に基づいたサンプリングを実行
        states, actions, label_q_values, profits = self.memory.sample(self.batch_size)

        # TODO NextStateが株価予測ではtick+1程度の未来のデータを利用しても教師とはならない。
        # NextStateはある程度の未来のデータを利用して教師とする必要がある。
        # NextStateを数十分間の間の未来のデータの平均値を利用することで教師を作成するとか？ 

        # statesの形は (batch_size, state_size, sequence_length)
        states = torch.from_numpy(np.array(states))
        # actionsの形は (batch_size, action_size) unqueeze(1)で次元を追加して (batch_size, 1, action_size) に変換
        actions = torch.LongTensor(actions).unsqueeze(1)
        # next_statesの形は (batch_size, state_size, sequence_length)
        # label_q_values = torch.from_numpy(np.array(label_q_values))
        # profitsの形は (batch_size,)
        profits = torch.FloatTensor(profits)
        # label_q_valuesの形は 現在の時間で最も良い行動のQ値 (batch_size, (nothing_q_value, buy_q_value, sell_q_value, take_profit_q_value)) 
        label_q_values = torch.from_numpy(np.array(label_q_values))
        # 現在の状態でのQ値を取得
        current_q_values, _ = self.model(states)
        
        loss = self.criterion(current_q_values, label_q_values)

        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward(retain_graph=True)  # 勾配を逆伝播
        self.optimizer.step()  # パラメータを更新

        self.memory.clear(self.batch_size)  # メモリをクリア

        # 探索率を減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return True


if __name__ == "__main__":

    agent = LSTMAgent(state_channel_size=4, sequence_length=512)
    model = agent.model
    
    output, _ = model.test(4, 512, 4)
    print(output.shape)