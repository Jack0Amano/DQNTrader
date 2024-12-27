import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Qネットワーク定義
class QNetwork(nn.Module):
    def __init__(self, state_channel_size, sequence_length, action_size):
        """
        Qネットワークのコンストラクタ   
        input_size: 状態の次元数 [[close(-sequence_length), ... close(current)],    
                                    [profit(-sequence_length), ... profit(current)],   
                                    [order_type(-sequence_length), ... order_type(current)]] shape=(3, sequence_length) 
                                    の場合input_size=3
        action_size: 行動の数 [何もしない, BUY, SELL, CHECKOUT]
        """
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=state_channel_size,
            hidden_size=action_size,
            num_layers=2,
            batch_first=True)
        self.conv1 = nn.Conv1d(sequence_length, 128, kernel_size=1, stride=4)
        self.conv2 = nn.Conv1d(128, 32, kernel_size=1, stride=2)
        self.conv3 = nn.Conv1d(32, action_size, kernel_size=1, stride=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x
    
    # Random入力してテストする
    def test(self, input_size, sequence_length, action_size):
        x = torch.randn(1, sequence_length, input_size)
        print(self.forward(x).shape)

# 優先度付き経験再生バッファの定義
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity  # バッファの容量
        self.alpha = alpha  # 優先度をどの程度重視するかを決めるハイパーパラメータ
        self.memory = []  # 経験を保存するリスト
        self.pos = 0  # 現在の挿入位置
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 優先度のリスト

    # 経験をバッファに追加
    def add(self, state, action, reward, next_state, profit, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        experience = (state, action, reward, next_state, profit, done)

        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    # 優先度に基づいて経験をサンプリング
    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.pos]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        # weightsは各memoryの重要度で shapeは(batch_size,)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, profits, dones = zip(*experiences)
        return states, actions, rewards, next_states, profits, dones, indices, weights

    # 優先度を更新
    def update(self, idx, priority):
        self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)
    
# DQNエージェントの定義
class DQNAgent:
    def __init__(
        self,
        state_channel_size = 3,
        sequence_length = 10,
        action_size = 4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        target_update_freq=10,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
    ):
        self.state_channel_size = state_channel_size  # 状態の次元数
        self.action_size = action_size  # 行動の数
        # 優先度付き経験再生バッファを初期化
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.gamma = gamma  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995  # 探索率の減少率
        self.epsilon_min = 0.01  # 最小探索率
        self.batch_size = batch_size  # ミニバッチのサイズ
        self.learning_rate = learning_rate  # 学習率
        self.target_update_freq = target_update_freq  # ターゲットネットワークの更新頻度
        self.beta_start = beta_start  # 優先度付き経験再生の初期β値
        self.beta_frames = beta_frames  # β値が1に達するまでのフレーム数
        self.frame = 0  # フレーム数をトラッキングするための変数
        # ポリシーネットワークのインスタンス
        self.model = QNetwork(state_channel_size, sequence_length, action_size)
        # ターゲットネットワークのインスタンス　教師用のアクションを作成するためのネットワーク
        self.target_model = QNetwork(state_channel_size, sequence_length, action_size)
        # ターゲットネットワークをポリシーネットワークのパラメータで初期化
        self.update_target_network()
        # Adamオプティマイザを使用してネットワークのパラメータを最適化
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # ターゲットネットワークをポリシーネットワークのパラメータで更新
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 経験をメモリに追加
    def remember(self, state, action, reward, next_state, profit, done):
        self.memory.add(state, action, reward, next_state, profit, done)

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
            act_values = self.model(state)  # 各行動のQ値を計算
        return torch.argmax(act_values).item()  # 最大のQ値を持つ行動を選択

    # 経験をリプレイしてネットワークを訓練
    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # メモリが十分に溜まるまではリプレイを実行しない

        self.frame += 1  # フレーム数をカウント
        # β値をフレーム数に基づいて更新
        beta = min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

        # メモリから優先度に基づいたサンプリングを実行
        states, actions, rewards, next_states, dones, profits, indices, weights = (
            self.memory.sample(self.batch_size, beta)
        )

        # statesの形は (batch_size, state_size, sequence_length)
        states = torch.from_numpy(np.array(states))
        # actionsの形は (batch_size, action_size) unqueeze(1)で次元を追加して (batch_size, 1, action_size) に変換
        actions = torch.LongTensor(actions).unsqueeze(1)
        # rewardsの形は (batch_size,)
        rewards = torch.FloatTensor(rewards)
        # next_statesの形は (batch_size, state_size, sequence_length)
        next_states = torch.from_numpy(np.array(next_states))
        # profitsの形は (batch_size,)
        profits = torch.FloatTensor(profits)
        # donesの形は (batch_size,)
        dones = torch.FloatTensor(dones)
        # weightsの形は (batch_size, )  unqueeze(1)で次元を追加して (batch_size, 1) に変換
        weights = torch.FloatTensor(weights).unsqueeze(1)

        # 現在の状態でのQ値を取得
        current_q_values = self.model(states).gather(1, actions)
        # 次の状態での最大Q値をターゲットネットワークから取得
        # next_q_valuesは (batch_size,)の形で各バッチの最大Q値を保持している
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        # ターゲットQ値を計算 ここでprofitなどの報酬も加算していく rewardsにprofitを組み込む形か？
        target_q_values = rewards + profits + (1 - dones) * self.gamma * next_q_values.unsqueeze(1)

        # TD誤差の二乗に重みを掛けた損失を計算
        loss = (current_q_values - target_q_values).pow(2) * weights
        loss = loss.mean()  # 平均損失を計算

        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # 勾配を逆伝播
        self.optimizer.step()  # パラメータを更新

        # サンプルした経験の優先度を更新
        for idx in indices:
            self.memory.update(idx, loss.item())

        # 探索率を減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ターゲットネットワークの更新
        if self.frame % self.target_update_freq == 0:
            self.update_target_network()


if __name__ == "__main__":
    agent = DQNAgent(sequence_length=512)
    agent.model.test(3, 512, 4)