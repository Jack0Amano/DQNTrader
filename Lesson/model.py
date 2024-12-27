import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Qネットワーク定義
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=255):
        super(QNetwork, self).__init__()
        # 入力層から中間層へ
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 中間層からさらに中間層へ
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 中間層から出力層（行動数）へ
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # ReLU活性化関数を使用して、各層の出力を次の層に渡す
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 最終層の出力を返す


# 優先度付き経験再生バッファの定義
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity  # バッファの容量
        self.alpha = alpha  # 優先度をどの程度重視するかを決めるハイパーパラメータ
        self.memory = []  # 経験を保存するリスト
        self.pos = 0  # 現在の挿入位置
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 優先度のリスト

    # 経験をバッファに追加
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        experience = (state, action, reward, next_state, done)

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
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones, indices, weights

    # 優先度を更新
    def update(self, idx, priority):
        self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


# DQNエージェントの定義
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        target_update_freq=10,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
    ):
        self.state_size = state_size  # 状態の次元数
        self.action_size = action_size  # 行動の数
        # 優先度付き経験再生バッファを初期化
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.gamma = gamma  # 割引率
        self.epsilon = 1.0  # 探索率
        try:
            with open("epsilon.txt", "r", encoding="utf-8") as file:
                epsilon_value_str = file.read().strip()
            if epsilon_value_str:
                self.epsilon = float(epsilon_value_str)
                print("探索率をロードしました。")
            else:
                # ファイルが空だった場合
                print("探索率が見つかりませんでした。")
        except FileNotFoundError:
            # ファイルが存在しない場合
            print("探索率が見つかりませんでした。")
        except ValueError:
            # ファイルから読み込んだ値がfloatに変換できなかった場合の処理
            print("探索率のファイルから読み込んだ値が正しい形式ではありません。")
        self.epsilon_decay = 0.995  # 探索率の減少率
        self.epsilon_min = 0.01  # 最小探索率
        self.batch_size = batch_size  # ミニバッチのサイズ
        self.learning_rate = learning_rate  # 学習率
        self.target_update_freq = target_update_freq  # ターゲットネットワークの更新頻度
        self.beta_start = beta_start  # 優先度付き経験再生の初期β値
        self.beta_frames = beta_frames  # β値が1に達するまでのフレーム数
        self.frame = 0  # フレーム数をトラッキングするための変数
        # ポリシーネットワークのインスタンス
        self.model = QNetwork(state_size, action_size)
        # ターゲットネットワークのインスタンス
        self.target_model = QNetwork(state_size, action_size)
        # ターゲットネットワークをポリシーネットワークのパラメータで初期化
        self.update_target_network()
        # Adamオプティマイザを使用してネットワークのパラメータを最適化
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # ターゲットネットワークをポリシーネットワークのパラメータで更新
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 経験をメモリに追加
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    # ε-グリーディー法で行動を選択
    def action(self, state):
        # 探索かネットワークの予測に基づいて行動を選択するか決定
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # ランダムに行動を選択
        state = torch.FloatTensor(state).unsqueeze(
            0
        )  # 状態をテンソルに変換してネットワークに入力
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
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, beta)
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        # 現在の状態でのQ値を取得
        current_q_values = self.model(states).gather(1, actions)
        # 次の状態での最大Q値をターゲットネットワークから取得
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        # ターゲットQ値を計算
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values.unsqueeze(
            1
        )

        print(next_q_values.unsqueeze(1).shape)

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
            with open("epsilon.txt", "w", encoding="utf-8") as file:
                file.write(str(self.epsilon))

        # ターゲットネットワークの更新
        if self.frame % self.target_update_freq == 0:
            self.update_target_network()