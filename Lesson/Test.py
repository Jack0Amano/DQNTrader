import gym
import math
import random
import numpy as np
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F

GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 1000
TOTAL_TS = 10000
SYNC_FREQ = 10
seed = 2023

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, device="cpu"):
        """
        バッファから指定されたバッチサイズの経験をランダムにサンプリングして返します   
        """
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(5):
            k = np.stack(tuple(item[i] for item in batch), axis=0)
            res.append(torch.tensor(k, device=device))
        return res[0], res[1], res[2], res[3], res[4]
    
device = "cuda" # "cpu"
env = gym.make("CartPole-v1")
n_observations = env.observation_space.shape[0]
# Action: 台を左に押す(0)か 右に押す（1)の二択になります
n_actions = env.action_space.n
buffer = ReplayBuffer(buffer_size=MEMORY_SIZE)
# ネットワークの意味としては、状態を入力として、envに入力する各行動を出力するネットワークです
main_net = nn.Sequential(
    nn.Linear(n_observations, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, n_actions)
).to(device)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(main_net.parameters())

from copy import deepcopy
target_net = deepcopy(main_net)

num_episode = global_ts = 0
ep_rewards = list()
while global_ts <= TOTAL_TS:
    state, done = env.reset(), False
    state = state[0]
    print(state)
    ep_reward, ep_ts = 0.0, 0
    while not done:
        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * global_ts / EPS_DECAY)
        if np.random.rand() < eps:
            action = np.random.choice(n_actions)
        else:
            # stateをネットワークに入力し、各行動のQ値を取得します
            with torch.no_grad():
                q = main_net(torch.tensor(state[None, :], device=device)).detach().cpu().numpy()
            action = np.argmax(q[0])

        # envはシミュレーターであるため動作を入力し、次の状態、報酬、終了したかどうか、追加情報を取得します
        output = env.step(action)
        next_state, reward, done, info, _ = output
        ep_reward += reward
        reward = reward if not done else -reward
        buffer.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            ep_rewards.append(ep_reward)
            if (num_episode == 0) or ((num_episode + 1) % 50 == 0):
                print(f"Ep-{num_episode + 1} {global_ts + 1}/{TOTAL_TS} Eps: {eps:.2f}, Reward: {ep_reward}")
            break
        
        # バッファからランダムにサンプリングしてネットワークを最適化します
        if buffer.size() >= BATCH_SIZE:
            obses_t, actions, rewards, obses_tp1, dones = buffer.sample(BATCH_SIZE, device)
            print(actions.unsqueeze(1))
            # Optimize the model
            with torch.no_grad():
                target_q = target_net(obses_tp1).detach().max(1)[0]
            target = rewards + GAMMA * target_q * (1 - dones.float())
            val_t = main_net(obses_t).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(val_t.float(), target.float())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Periodically sync the target and main networks
            if (global_ts + 1) % SYNC_FREQ == 0:
                target_net.load_state_dict(main_net.state_dict())
        ep_ts += 1
        global_ts += 1
    num_episode += 1