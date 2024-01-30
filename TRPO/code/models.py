# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/30 16:31
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim=1):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, lam, kl_constraint, alpha, critic_lr, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(DEVICE)
        self.critic = ValueNet(state_dim, hidden_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lam = lam
        self.kl_constraint = kl_constraint
        self.alpha = alpha

    def choose_action(self, state):
        pi = self.actor(state)
        dist = Categorical(pi)
        action = dist.sample().item()
        return action

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        r2 = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = r2 / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_r2 = torch.dot(r, r)
            if new_r2 < 1e-10:
                break
            beta = new_r2 / r2
            p = r + beta * p
            r2 = new_r2
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_prob, actor):
        log_prob = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_prob - old_log_prob)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_prob, old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
        for i in range(15):
            coefficient = self.alpha ** i
            new_para = old_para + coefficient * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_prob, advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_prob, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coefficient = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_prob, old_action_dists,
                                    descent_direction * max_coefficient)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

    def learn(self, transition_dict):
        state = torch.tensor(np.array(transition_dict["state"]), dtype=torch.float).to(DEVICE)
        action = torch.tensor(transition_dict["action"]).view(-1, 1).to(DEVICE)
        reward = torch.tensor(transition_dict["reward"], dtype=torch.float).view(-1, 1).to(DEVICE)
        next_state = torch.tensor(np.array(transition_dict["next_state"]), dtype=torch.float).to(DEVICE)
        done = torch.tensor(transition_dict["done"], dtype=torch.float).view(-1, 1).to(DEVICE)
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_delta = td_target - self.critic(state)
        from utils import compute_advantage
        advantage = compute_advantage(self.gamma, self.lam, td_delta.cpu()).to(DEVICE)
        old_log_prob = torch.log(self.actor(state).gather(1, action)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(state).detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(state, action, old_action_dists, old_log_prob, advantage)

    def save_checkpoint(self, save_path, episode):
        torch.save(self.actor.state_dict(), f"{save_path}/{episode}-actor.pkl")
        torch.save(self.critic.state_dict(), f"{save_path}/{episode}-critic.pkl")
