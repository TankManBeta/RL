# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/26 15:24
"""
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# use namedtuple to help manager transitions
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "action_prob", "done"))


class DataPool:
    def __init__(self):
        self.pool = deque([])

    def get_data(self):
        state_list, action_list, reward_list, state_next_list, action_prob_list, done_list = [], [], [], [], [], []
        for transition in self.pool:
            state, action, reward, state_next, action_prob, done = transition
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
            state_next_list.append(state_next)
            action_prob_list.append(action_prob)
            done_list.append(done)
        self.clear()
        data = {
            "state": state_list,
            "action": action_list,
            "reward": reward_list,
            "state_next": state_next_list,
            "action_prob": action_prob_list,
            "done": done_list
        }
        return data

    def push(self, *args):
        self.pool.append(Transition(*args))

    def clear(self):
        self.pool = deque([])

    def __len__(self):
        return len(self.pool)


# Policy
class PolicyNet(nn.Module):
    def __init__(self, n_features, n_actions, n_width=300, n_depth=2):
        super(PolicyNet, self).__init__()
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
        out = F.softmax(self.model(x), dim=-1)
        return out


class PGAgent:
    def __init__(self, n_features, n_actions, gamma=0.98, lr=1e-3):
        super(PGAgent, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.policy = PolicyNet(n_features, n_actions).to(DEVICE)
        self.optimizer_policy = optim.Adam(params=self.policy.parameters(), lr=lr)

    # choose action according to the distribution
    def choose_action(self, state):
        pi = self.policy(state)
        dist = Categorical(pi)
        action = dist.sample().item()
        action_prob = pi[action].item()
        return action, action_prob

    def learn(self, transition_dict):
        reward_list = transition_dict["reward"]
        state_list = transition_dict["state"]
        action_list = transition_dict["action"]

        G = 0
        self.optimizer_policy.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(DEVICE)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(DEVICE)
            log_prob = torch.log(self.policy(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer_policy.step()

    def save_checkpoint(self, save_path, episode):
        torch.save(self.policy.state_dict(), f"{save_path}/{episode}-model.pkl")
