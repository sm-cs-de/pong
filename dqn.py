import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import pong_game.config as cfg


class Buffer:
    def __init__(self, size=1000):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def append(self):


    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return np.array(states,dtype=np.float32), np.array(actions,dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(next_states,dtype=np.float32), np.array(dones,dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, lr=1e-3, gamma=0.99, eps_beg=1.0, eps_end=0.01, eps_red=5000, buffer_size=1000, buffer_update=100, buffer_batch=10):
        self.gamma = gamma
        self.eps = eps_beg
        self.eps_min = eps_end
        self.eps_decay = eps_red

        self.buffer = Buffer(buffer_size)
        self.buffer_update = buffer_update
        self.buffer_batch = buffer_batch

        self.online_net = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.target_net = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.step_count = 0

    def get_action(self, state):
        """
        Epsilon-greedy action selection.
        'state' is assumed to be a numpy array of shape (state_dim,).
        """
        self.step_count += 1
        self.eps = max(self.eps_min, self.eps - (1.0 / self.eps_decay))

        if random.random() < self.eps:
            return random.choice(cfg.player_move_dist)
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


# class MockGameEngine:
#     def __init__(self, left_bound=0, right_bound=49, box_width=5):
#         self.left_bound = left_bound
#         self.right_bound = right_bound
#         self.box_width = box_width
#         self.position = 25  # start in the middle
#         self.step_count = 0
#         self.max_steps = 50

    # def get_player_position(self):
    #     return self.position

#     def apply_action(self, action_idx):
#         """
#         action_idx -> pixel offset from the ACTIONS list.
#         Return (reward, done).
#         """
#         ACTIONS = [-2, -1, 0, +1, +2]
#         move = ACTIONS[action_idx]
#
#         # Update position within bounds
#         self.position = max(self.left_bound, min(self.right_bound, self.position + move))
#
#         # Simple termination condition: let's say after max_steps
#         self.step_count += 1
#         done = (self.step_count >= self.max_steps)
#
#         # We'll compute the reward outside or here.
#         # Let's do it here for convenience: sum of distribution under the box.
#         dist = distribution_net()  # new distribution after the move (mock)
#         left_edge = self.position
#         right_edge = min(self.position + self.box_width - 1, self.right_bound)
#
#         reward = dist[left_edge:right_edge + 1].sum()
#
#         return (reward, done), dist
#
#
# # Instantiate everything
# distribution_length = 50
# box_width = 5
# state_dim = distribution_length + 1  # distribution + position
# action_dim = 5  # discrete moves: [-2, -1, 0, +1, +2]
#
# agent = DQN()
# replay_buffer = Buffer()
#
# # Hyperparams
# num_episodes = 30
# batch_size = 32
#
# for episode in range(num_episodes):
#     engine = MockGameEngine(box_width=box_width)
#
#     # Initial distribution & state
#     dist = distribution_net()
#     pos = engine.get_player_position()
#
#     # Construct initial state vector [distribution, position]
#     # e.g., shape: [50 + 1] = [51]
#     state = np.concatenate([dist, [pos]], axis=0)
#
#     done = False
#     total_reward = 0.0
#
#     while not done:
#         # 1) Agent selects action
#         action_idx = agent.select_action(state)
#
#         # 2) Environment (game engine) applies action
#         (reward, done), next_dist = engine.apply_action(action_idx)
#         next_pos = engine.get_player_position()
#
#         # 3) Construct next_state
#         next_state = np.concatenate([next_dist, [next_pos]], axis=0)
#
#         # 4) Store transition in replay buffer
#         replay_buffer.push(state, action_idx, reward, next_state, done)
#
#         # 5) Train step
#         agent.train_step(replay_buffer, batch_size)
#
#         # Update for next iteration
#         state = next_state
#         dist = next_dist
#         total_reward += reward
#
#     print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.3f}")
