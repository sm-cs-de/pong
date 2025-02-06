# https://atlane.de/deep-q-networks-eine-revolution-im-reinforcement-learning/
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import game.config as cfg


class Buffer:
    def __init__(self, size=1000):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def append(self, state, action, reward, done):
        if len(self.buffer) > 0:
            state_last, action_last, reward_last, next_state_last, done_last = self.buffer[-1]
            self.push(next_state_last, action, reward, state, done)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return np.array(states,dtype=np.float32), np.array(actions,dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(next_states,dtype=np.float32), np.array(dones,dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, lr=1e-2, gamma=0.99, eps_beg=0.2, eps_end=0.01, eps_red=5000, buffer_size=100, buffer_update=10, buffer_batch=10):
        self.gamma = gamma
        self.eps = eps_beg
        self.eps_beg = eps_beg
        self.eps_min = eps_end
        self.eps_red = eps_red

        self.buffer = Buffer(buffer_size)
        self.buffer_update = buffer_update
        self.buffer_batch = buffer_batch

        self.online_net = nn.Sequential(nn.Linear(cfg.HEIGHT+1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(cfg.player_move_distances)))
        self.target_net = nn.Sequential(nn.Linear(cfg.HEIGHT+1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, len(cfg.player_move_distances)))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.step_count = 0

    def reset(self):
        self.eps = self.eps_beg
        self.buffer.buffer.clear()

        def reinit(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)
        self.online_net.apply(reinit)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.step_count = 0

    def get_action(self, state):
        """
        Epsilon-greedy action selection.
        'state' is assumed to be a numpy array of shape (state_dim,).
        """
        self.step_count += 1
        self.eps = max(self.eps_min, self.eps - (1.0 / self.eps_red))

        if random.random() < self.eps:
            return random.randint(0, len(cfg.player_move_distances)-1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)  # shape (1, state_dim)
            with torch.no_grad():
                q_values = self.online_net(state_t)

            return q_values.argmax(dim=1).item()

    def train_step(self):
        """
        Sample a batch from replay_buffer, compute the loss, and update online_net.
        """
        if len(self.buffer) < self.buffer_batch:
            return  # not enough samples

        states, actions, rewards, next_states, dones = self.buffer.sample(self.buffer_batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # print(states_t)
        # print(actions_t)
        # print(rewards_t)
        # print(next_states_t)
        # print(dones_t)
        # print(self.online_net(states_t))

        q_values = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            # DQN max_a' Q_target(s', a')
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

        targets = rewards_t + self.gamma * (1 - dones_t) * next_q_values
        loss = self.loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % self.buffer_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
