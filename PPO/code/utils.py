# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/1/5 16:13
"""
from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import torch
from models import PPOAgent, DataPool
from torch.utils.tensorboard import SummaryWriter


def run_one_episode(env, agent, data_pool):
    state, _ = env.reset()
    reward_episode = 0
    done, truncated = False, False
    while not done and not truncated:
        action, action_prob = agent.choose_action(state)
        state_next, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        data_pool.push(state, action, reward, state_next, action_prob, done)
        state = state_next
        reward_episode += reward
    data = data_pool.get_data()
    agent.learn(data)
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
    agent = PPOAgent(observation_n, action_n, gamma=0.98, lr=2e-3, lam=0.95, epsilon_clip=0.2)
    reward_list = []
    data_pool = DataPool()
    for episode in range(300):
        reward = run_one_episode(env, agent, data_pool)
        reward_list.append(reward)
        print(f"Episode: {episode}, reward: {reward}")
        writer.add_scalar("Reward/train", reward, global_step=episode)
        if (episode + 1) % 10 == 0:
            agent.save_checkpoint(checkpoint_path, episode)
    writer.close()
    plt.plot(list(range(len(reward_list))), reward_list)
    plt.show()
    print("Training ends!!!")
    evaluate(env, agent, pic_path)
