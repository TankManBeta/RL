# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2023/12/24 15:10
"""
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


def evaluate(env, agent):
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
    for frame in frame_list:
        plt.imshow(frame)
        plt.axis('off')
        plt.show()


def train():
    print("Training starts!!!")
    writer = SummaryWriter(log_dir="../results/logs")
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    observation_n, action_n = env.observation_space.shape[0], env.action_space.n
    agent = ACAgent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01)
    reward_list = []
    for epoch in range(1000):
        reward = run_one_episode(env, agent)
        reward_list.append(reward)
        print(f"Episode: {epoch}, reward: {reward}")
        writer.add_scalar("Reward/train", reward, global_step=epoch)
    writer.close()
    plt.plot(list(range(len(reward_list))), reward_list)
    plt.show()
    print("Training ends!!!")
    evaluate(env, agent)
