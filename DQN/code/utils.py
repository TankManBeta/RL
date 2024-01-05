# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2023/11/17 11:17
"""
from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import torch
from models import DQNAgent, MemoryPool, Transition
from torch.utils.tensorboard import SummaryWriter


def run_one_episode(env, agent, memory_pool, batch_size, epsilon):
    state, info = env.reset()
    reward_episode = 0
    done = False
    count = 0
    while count < 200 and not done:
        action = agent.choose_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        memory_pool.push(state, action, reward, next_state, done)
        if len(memory_pool) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory_pool.sample(batch_size)
            T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            agent.learn(T_data)
        state = next_state
        reward_episode += reward
        count += 1
    return reward_episode


def evaluate(env, agent, save_path):
    state, info = env.reset()
    reward_episode = 0
    frame_list = []
    done = False
    count = 0
    while count < 200 and not done:
        action = torch.argmax(agent.evaluate_model(state)).item()
        next_state, reward, done, _, _ = env.step(action)
        reward_episode += reward
        state = next_state
        count += 1
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
    agent = DQNAgent(observation_n, action_n, gamma=0.98, lr=5e-4, update_interval=20)
    memory_pool = MemoryPool(pool_size=50000)
    batch_size = 32
    reward_list = []
    for episode in range(100):
        epsilon = max(0.01, 0.1 - 0.01*(episode/20))
        reward = run_one_episode(env, agent, memory_pool, batch_size, epsilon)
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
