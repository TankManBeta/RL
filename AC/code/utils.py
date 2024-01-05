# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2023/12/24 15:10
"""
from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import torch
from models import ACAgent
from torch.utils.tensorboard import SummaryWriter


def run_one_episode(env, agent):
    state, info = env.reset()
    reward_episode = 0
    done = False
    count = 0
    transition_dict = {"state": [], "action": [], "state_next": [], "reward": [], "done": []}
    while count < 200 and not done:
        action = agent.choose_action(state)
        state_next, reward, done, _, _ = env.step(action)
        transition_dict["state"].append(state)
        transition_dict["action"].append(action)
        transition_dict["state_next"].append(state_next)
        transition_dict["reward"].append(reward)
        transition_dict["done"].append(done)
        state = state_next
        reward_episode += reward
        count += 1
    agent.learn(transition_dict)
    return reward_episode


def evaluate(env, agent, save_path):
    state, info = env.reset()
    reward_episode = 0
    frame_list = []
    done = False
    count = 0
    while count < 200 and not done:
        prob = agent.actor(state)
        action = torch.argmax(prob).item()
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
    agent = ACAgent(observation_n, action_n, gamma=0.98, lr=2e-3)
    reward_list = []
    for episode in range(100):
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
