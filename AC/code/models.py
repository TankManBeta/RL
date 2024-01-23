# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2023/12/24 15:10
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Actor
class Actor(nn.Module):
    def __init__(self, n_features, n_actions, n_width=300, n_depth=2):
        super(Actor, self).__init__()
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


# Critic
class Critic(nn.Module):
    def __init__(self, n_features, n_actions, n_width=300, n_depth=2):
        super(Critic, self).__init__()
        self.n_features = n_features
        self.n_width = n_width
        self.n_depth = n_depth
        self.n_actions = n_actions
        self.model = self.build_model()

    def build_model(self):
        layer_dim = [self.n_features] + [self.n_width for _ in range(self.n_depth)]
        layers = []
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.n_width, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.model(x)


class ACAgent:
    def __init__(self, n_features, n_actions, gamma=0.98, lr=1e-3):
        super(ACAgent, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.actor = Actor(n_features, n_actions).to(DEVICE)
        self.critic = Critic(n_features, n_actions).to(DEVICE)
        self.optimizer_actor = optim.Adam(params=self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(params=self.critic.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    # choose action according to the distribution
    def choose_action(self, state):
        pi = self.actor(state)
        dist = Categorical(pi)
        action = dist.sample()
        return action.item()

    def learn(self, transition_dict):
        # process data
        state = torch.tensor(np.array(transition_dict["state"]), dtype=torch.float).to(DEVICE)
        action = torch.tensor(transition_dict["action"]).view(-1, 1).to(DEVICE)
        reward = torch.tensor(transition_dict["reward"], dtype=torch.float).view(-1, 1).to(DEVICE)
        state_next = torch.tensor(np.array(transition_dict["state_next"]), dtype=torch.float).to(DEVICE)
        done = torch.tensor(transition_dict["done"], dtype=torch.float).view(-1, 1).to(DEVICE)

        # calculate advantage function
        v = self.critic(state)
        v_next = self.critic(state_next)
        td = self.gamma * v_next * (1 - done) + reward - v
        # get probability
        pi = self.actor(state)
        prob = pi.gather(1, action)
        log_prob = torch.log(prob)
        # loss of actor
        loss_actor = torch.mean(-log_prob * td.detach())
        # loss of critic
        loss_critic = self.loss_function(self.gamma * v_next * (1-done) + reward, v)
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def save_checkpoint(self, save_path, episode):
        torch.save(self.actor.state_dict(), f"{save_path}/{episode}-actor.pkl")
        torch.save(self.critic.state_dict(), f"{save_path}/{episode}-critic.pkl")
