# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/5 13:34
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
            action_list.append([action])
            reward_list.append([reward])
            state_next_list.append(state_next)
            action_prob_list.append([action_prob])
            done_list.append([done])
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


class PPOAgent:
    def __init__(self, n_features, n_actions, gamma=0.98, lr=1e-3, lam=0.95, epsilon_clip=0.2):
        super(PPOAgent, self).__init__()
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.lam = lam
        self.epsilon_clip = epsilon_clip
        self.actor = Actor(n_features, n_actions).to(DEVICE)
        self.critic = Critic(n_features, n_actions).to(DEVICE)
        self.optimizer_actor = optim.Adam(params=self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(params=self.critic.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    def choose_action(self, state):
        pi = self.actor(state)
        dist = Categorical(pi)
        action = dist.sample().item()
        action_prob = pi[action].item()
        return action, action_prob

    def learn(self, transition_dict):
        # process data
        state = torch.tensor(np.array(transition_dict["state"]), dtype=torch.float).to(DEVICE)
        action = torch.tensor(transition_dict["action"]).view(-1, 1).to(DEVICE)
        reward = torch.tensor(transition_dict["reward"], dtype=torch.float).view(-1, 1).to(DEVICE)
        state_next = torch.tensor(np.array(transition_dict["state_next"]), dtype=torch.float).to(DEVICE)
        action_prob = torch.tensor(transition_dict["action_prob"], dtype=torch.float).to(DEVICE)
        done = torch.tensor(transition_dict["done"], dtype=torch.float).view(-1, 1).to(DEVICE)
        # calculate advantage function
        with torch.no_grad():
            v = self.critic(state)
            v_next = self.critic(state_next)
            td = self.gamma * v_next * (1 - done) + reward - v
            td = td.detach().numpy()
            advantage_list = []
            advantage = 0.0
            for td_t in td[::-1]:
                advantage = self.gamma * self.lam * advantage + td_t[0]
                advantage_list.append([advantage])
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float)
            v_target = advantage + v

        pi = self.actor(state)
        dist = Categorical(probs=pi)
        dist_entropy = dist.entropy().mean()
        # dist_entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1, keepdim=True)
        prob = pi.gather(1, action)
        # calculate the importance weight
        ratio = torch.exp(torch.log(prob) - torch.log(action_prob))
        # ppo clip
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage
        # loss of actor
        loss_actor = torch.mean(-torch.min(surr1, surr2)) - 0.01 * dist_entropy
        # loss of critic
        loss_critic = self.loss_function(v_target, self.critic(state))
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def save_checkpoint(self, save_path, episode):
        torch.save(self.actor.state_dict(), f"{save_path}/{episode}-actor.pkl")
        torch.save(self.critic.state_dict(), f"{save_path}/{episode}-critic.pkl")
