# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/30 17:38
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from models import TRPOAgent


def compute_advantage(gamma, lam, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lam * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def run_one_episode(env, agent):
    state, _ = env.reset()
    reward_episode = 0
    done, truncated = False, False
    transition_dict = {"state": [], "action": [], "next_state": [], "reward": [], "done": []}
    while not done and not truncated:
        action = agent.choose_action(state)
        state_next, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        transition_dict["state"].append(state)
        transition_dict["action"].append(action)
        transition_dict["next_state"].append(state_next)
        transition_dict["reward"].append(reward)
        transition_dict["done"].append(done)
        state = state_next
        reward_episode += reward
    agent.learn(transition_dict)
    return reward_episode


def evaluate(env, agent, save_path):
    state, _ = env.reset()
    reward_episode = 0
    frame_list = []
    done, truncated = False, False
    while not done and not truncated:
        prob = agent.actor(state)
        action = torch.argmax(prob).item()
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        reward_episode += reward
        state = next_state
        frame_list.append(env.render())
    # draw frames
    for idx, frame in enumerate(frame_list):
        plt.imshow(frame)
        plt.axis('off')
        plt.savefig(f"{save_path}/{idx}.png")
        plt.show()


def train():
    print("Training starts!!!")
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    prefix_path = Path(f"../results/{now}")
    prefix_path.mkdir(exist_ok=True)
    log_path = prefix_path / "logs"
    log_path.mkdir(exist_ok=True)
    pic_path = prefix_path / "pics"
    pic_path.mkdir(exist_ok=True)
    checkpoint_path = prefix_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    agent = TRPOAgent(observation_n, hidden_dim=128, action_dim=action_n, gamma=0.98, lam=0.95, critic_lr=1e-2,
                      kl_constraint=0.0005, alpha=0.5)
    reward_list = []
    for episode in range(300):
        reward = run_one_episode(env, agent)
        reward_list.append(reward)
        print(f"Episode: {episode}, reward: {reward}")
        writer.add_scalar("Reward/train", reward, global_step=episode)
        if (episode+1) % 10 == 0:
            agent.save_checkpoint(checkpoint_path, episode)
    writer.close()
    plt.plot(list(range(len(reward_list))), reward_list)
    plt.show()
    print("Training ends!!!")
    evaluate(env, agent, pic_path)
