# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/30 18:45
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


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(n_states, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(n_states + n_actions, 256)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class TD3Agent:
    def __init__(self, n_states, n_actions, action_space, gamma=0.99, policy_freq=2, actor_lr=1e-3, critic_lr=1e-3,
                 hidden_dim=256, tau=0.005, policy_noise=0.2, explore_noise=0.1, noise_clip=0.5, explore_steps=10):
        # td target parameter
        self.gamma = gamma
        # learning rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # standard deviation of noise added to the output of the policy
        self.policy_noise = policy_noise
        # maximum value of noise added to the policy
        self.noise_clip = noise_clip
        # exploration ratio
        self.explore_noise = explore_noise
        # update frequency of the target policy network
        self.policy_freq = policy_freq
        # soft update parameters of the target network
        self.tau = tau
        self.sample_count = 0
        # steps to explore
        self.explore_steps = explore_steps
        self.n_actions = n_actions
        self.n_states = n_states
        self.action_space = action_space
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low) / 2, device=DEVICE,
                                         dtype=torch.float32).unsqueeze(dim=0)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low) / 2, device=DEVICE,
                                        dtype=torch.float32).unsqueeze(dim=0)

        self.actor = Actor(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.actor_target = Actor(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic1 = Critic(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.critic2 = Critic(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.critic1_target = Critic(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.critic2_target = Critic(n_states, n_actions, hidden_dim=hidden_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_lr)

    def choose_action(self, state):
        self.sample_count += 1
        # explore
        if self.sample_count < self.explore_steps:
            return self.action_space.sample()
        else:
            state = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(dim=0)
            action = self.actor(state)
            action = self.action_scale * action + self.action_bias
            action = action.detach().cpu().numpy()[0]
            action_noise = np.random.normal(0, self.action_scale.cpu().numpy()[0] * self.explore_noise,
                                            size=self.n_actions)
            action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
            return action

    def learn(self, transition_dict):
        # process data
        state = torch.tensor(np.array(transition_dict.state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(transition_dict.action), dtype=torch.float).view(-1, 1).to(DEVICE)
        reward = torch.tensor(transition_dict.reward, dtype=torch.float).view(-1, 1).to(DEVICE)
        state_next = torch.tensor(np.array(transition_dict.state_next), dtype=torch.float).to(DEVICE)
        done = torch.tensor(transition_dict.done, dtype=torch.float).view(-1, 1).to(DEVICE)

        # construct noise incorporating target actions
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # calculate target actions with added noise
        next_action = (self.actor_target(state_next) + noise).clamp(-self.action_scale + self.action_bias,
                                                                    self.action_scale + self.action_bias)
        # calculate the ratings of two critic networks for the next state and action
        target_q1, target_q2 = self.critic1_target(state_next, next_action).detach(), self.critic2_target(
            state_next, next_action).detach()
        # select a smaller value to calculate the target q value
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + self.gamma * target_q * (1 - done)

        # get q value
        current_q1, current_q2 = self.critic1(state, action), self.critic2(state, action)
        # critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # delay policy update
        if self.sample_count % self.policy_freq == 0:
            # actor loss
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # soft update
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, save_path, episode):
        torch.save(self.actor.state_dict(), f"{save_path}/{episode}-actor.pkl")
        torch.save(self.critic1.state_dict(), f"{save_path}/{episode}-critic1.pkl")
        torch.save(self.critic2.state_dict(), f"{save_path}/{episode}-critic2.pkl")
