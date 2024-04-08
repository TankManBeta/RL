# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/4/8 12:49
"""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Normal

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


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        # 重参数化采样
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if isinstance(a, np.ndarray):
            a = torch.FloatTensor(a)
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACAgent:
    """
    处理连续动作的SAC算法
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy,
                 tau, gamma):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(DEVICE)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(DEVICE)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(DEVICE)
        # 第一个目标Q网络
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(DEVICE)
        # 第二个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(DEVICE)
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        # 可以对alpha求梯度
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        # 目标熵的大小
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(DEVICE)
        action = self.actor(state)[0]
        return [action.item()]

    # 计算目标Q值
    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    # 软更新
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, transition_dict):
        states = torch.tensor(transition_dict.state, dtype=torch.float).to(DEVICE)
        actions = torch.tensor(transition_dict.action, dtype=torch.float).view(-1, 1).to(DEVICE)
        rewards = torch.tensor(transition_dict.reward, dtype=torch.float).view(-1, 1).to(DEVICE)
        next_states = torch.tensor(transition_dict.state_next, dtype=torch.float).to(DEVICE)
        dones = torch.tensor(transition_dict.done, dtype=torch.float).view(-1, 1).to(DEVICE)
        # 对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_checkpoint(self, save_path, episode):
        torch.save(self.actor.state_dict(), f"{save_path}/{episode}-actor.pkl")
        torch.save(self.critic_1.state_dict(), f"{save_path}/{episode}-critic1.pkl")
        torch.save(self.critic_2.state_dict(), f"{save_path}/{episode}-critic2.pkl")
        torch.save(self.critic_2.state_dict(), f"{save_path}/{episode}-target_critic1.pkl")
        torch.save(self.critic_2.state_dict(), f"{save_path}/{episode}-target_critic2.pkl")
