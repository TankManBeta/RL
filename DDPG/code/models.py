# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/28 21:46
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# use namedtuple to help manager transitions
Transition = namedtuple("Transition", ("state", "action", "reward", "state_next", "done"))


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


class QNet(nn.Module):
    def __init__(self, n_features=3, n_actions=1, n_hidden=64, n_temp=32, n_out=1):
        super(QNet, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.n_temp = n_temp
        self.n_out = n_out
        self.fc_s = nn.Linear(self.n_features, self.n_hidden)
        self.fc_a = nn.Linear(self.n_actions, self.n_hidden)
        self.fc_q = nn.Linear(self.n_hidden * 2, self.n_temp)
        self.fc_out = nn.Linear(self.n_temp, self.n_out)

    def forward(self, x, a):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if isinstance(a, np.ndarray):
            a = torch.FloatTensor(a)
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        out = self.fc_out(q)
        return out


class MuNet(nn.Module):
    def __init__(self, n_features=3, n_hidden=64, n_temp=32, n_out=1, action_bound=2):
        super(MuNet, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_temp = n_temp
        self.n_out = n_out
        self.action_bound = action_bound
        self.fc1 = nn.Linear(self.n_features, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_temp)
        self.fc_mu = nn.Linear(self.n_temp, self.n_out)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * self.action_bound
        return mu


class DDPGAgent:
    def __init__(self, dim_state=3, dim_action=1, dim_hidden=64, dim_temp=32, dim_out=1, action_bound=2, sigma=0.01,
                 lr_mu=3e-4, lr_q=3e-3, tau=0.005, gamma=0.98):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden
        self.dim_temp = dim_temp
        self.dim_out = dim_out
        self.action_bound = action_bound
        self.sigma = sigma
        self.lr_mu = lr_mu
        self.lr_q = lr_q
        self.tau = tau
        self.gamma = gamma

        self.mu = MuNet(self.dim_state, self.dim_hidden, self.dim_temp, self.dim_out, self.action_bound).to(DEVICE)
        self.q = QNet(self.dim_state, self.dim_action, self.dim_hidden, self.dim_temp, self.dim_out).to(DEVICE)
        self.mu_target = MuNet(self.dim_state, self.dim_hidden, self.dim_temp, self.dim_out, self.action_bound).to(DEVICE)
        self.q_target = QNet(self.dim_state, self.dim_action, self.dim_hidden, self.dim_temp, self.dim_out).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())
        self.mu_target.load_state_dict(self.mu.state_dict())
        self.optimizer_mu = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
        self.optimizer_q = optim.Adam(self.q.parameters(), lr=self.lr_q)

    # choose action according to the distribution
    def choose_action(self, state):
        action = self.mu(state).item()
        action = action + self.sigma * np.random.randn(self.dim_action)
        return action

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, transition_dict):
        # process data
        state = torch.tensor(np.array(transition_dict.state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(transition_dict.action), dtype=torch.float).view(-1, 1).to(DEVICE)
        reward = torch.tensor(transition_dict.reward, dtype=torch.float).view(-1, 1).to(DEVICE)
        state_next = torch.tensor(np.array(transition_dict.state_next), dtype=torch.float).to(DEVICE)
        done = torch.tensor(transition_dict.done, dtype=torch.float).view(-1, 1).to(DEVICE)

        q_value = self.q(state, action)
        q_value_next = reward + self.gamma * self.q_target(state_next, self.mu_target(state_next)) * (1-done)
        loss_q = torch.mean(F.mse_loss(q_value, q_value_next))
        self.optimizer_q.zero_grad()
        loss_q.backward()
        self.optimizer_q.step()

        loss_mu = -torch.mean(self.q(state, self.mu(state)))
        self.optimizer_mu.zero_grad()
        loss_mu.backward()
        self.optimizer_mu.step()

        self.soft_update(self.mu, self.mu_target)
        self.soft_update(self.q, self.q_target)

    def save_checkpoint(self, save_path, episode):
        torch.save(self.mu.state_dict(), f"{save_path}/{episode}-mu.pkl")
        torch.save(self.q.state_dict(), f"{save_path}/{episode}-q.pkl")
