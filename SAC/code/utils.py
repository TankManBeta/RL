# -*- coding: utf-8 -*-

"""
    @Author 坦克手贝塔
    @Date 2024/4/8 12:49
"""
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import gym
from models import SACAgent, MemoryPool, Transition


def run_one_episode(env, agent, memory_pool, batch_size):
    state, _ = env.reset()
    reward_episode = 0
    done, truncated = False, False
    while not done and not truncated:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        memory_pool.push(state, action, reward, next_state, done)
        if len(memory_pool) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory_pool.sample(batch_size)
            T_data = Transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            agent.learn(T_data)
        state = next_state
        reward_episode += reward
    return reward_episode


def evaluate(env, agent, save_path):
    state, _ = env.reset()
    reward_episode = 0
    frame_list = []
    done, truncated = False, False
    while not done and not truncated:
        action = agent.actor(state)[0].detach().numpy()
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
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    observation_n = env.observation_space.shape[0]
    action_n = env.action_space.shape[0]
    hidden_dim = 128
    action_bound = env.action_space.high[0]
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    target_entropy = -env.action_space.shape[0]
    gamma = 0.99
    tau = 0.005  # 软更新参数
    agent = SACAgent(observation_n, hidden_dim, action_n, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy,
                     tau, gamma)
    memory_pool = MemoryPool(pool_size=50000)
    batch_size = 32
    reward_list = []
    for episode in range(300):
        reward = run_one_episode(env, agent, memory_pool, batch_size)
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
