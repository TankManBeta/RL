# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2023/11/17 9:53
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# use namedtuple to help manager transitions
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class MemoryPool(object):
    def __init__(self, pool_size):
        self.pool = deque([], maxlen=pool_size)

    def sample(self, batch_size):
        batch_data = random.sample(self.pool, batch_size)
        state, action, reward, next_state, done = zip(*batch_data)
        return state, action, reward, next_state, done

    def push(self, *args):
        self.pool.append(Transition(*args))

    def __len__(self):
        return len(self.pool)


class Network(nn.Module):
    def __init__(self, n_features, n_actions, n_width=128, n_depth=2):
        super(Network, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_width = n_width
        self.n_depth = n_depth
        self.model = self.build_model()

    def build_model(self):
        layer_dim = [self.n_features] + [self.n_width for _ in range(self.n_depth)]
        layers = []
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.n_width, self.n_actions))
        return nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.model(x)


class DQNAgent:
    def __init__(self, n_features, n_actions, gamma=0.98, lr=2e-3, update_interval=20):
        super(DQNAgent, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.update_interval = update_interval
        self.interval_count = 0
        self.evaluate_model = Network(n_features, n_actions).to(DEVICE)
        self.target_model = Network(n_features, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(params=self.evaluate_model.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            state = torch.FloatTensor(state).to(DEVICE)
            actions_value = self.evaluate_model(state)
            action = np.argmax(actions_value.detach().numpy(), axis=-1)
        return action

    def update_params(self):
        self.target_model.load_state_dict(self.evaluate_model.state_dict())

    def learn(self, transition_dict):
        # get transition
        state = transition_dict.state
        action = np.expand_dims(transition_dict.action, axis=-1)
        reward = np.expand_dims(transition_dict.reward, axis=-1)
        next_state = transition_dict.next_state
        done = np.expand_dims(transition_dict.done, axis=-1)

        # process data
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.int64).to(DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        done = torch.tensor(np.array(done), dtype=torch.float).to(DEVICE)

        q_evaluate = self.evaluate_model(state).gather(1, action)
        with torch.no_grad():
            max_next_q = self.target_model(next_state).max(1)[0].view(-1, 1)
            q_target = reward + self.gamma * max_next_q * (1 - done)
        loss = self.loss_function(q_evaluate, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update parameters of the evaluate model if the condition is satisfied
        if self.interval_count % self.update_interval == 0:
            self.update_params()
        self.interval_count += 1

    def save_checkpoint(self, save_path, episode):
        torch.save(self.evaluate_model.state_dict(), f"{save_path}/{episode}-evaluate.pkl")
        torch.save(self.target_model.state_dict(), f"{save_path}/{episode}-target.pkl")
