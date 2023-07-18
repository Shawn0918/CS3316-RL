from TD3 import TD3
from DDPG import DDPGAgent
from utils import ReplayBuffer

import argparse
from collections import namedtuple, deque
from itertools import count

import csv
import os, sys, random
import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

def output_to_file(rewards_by_100, rewards_by_10, rewards, res_dir, agent_name, env_name, exist):
    reward_10_name = os.path.join(res_dir, env_name + "_" + agent_name + "_rewards_by_10_table.csv")
    reward_100_name = os.path.join(res_dir, env_name + "_" + agent_name + "_rewards_by_100_table.csv")
    reward_name = os.path.join(res_dir, env_name + "_" + agent_name + "_rewards.csv")
    if not exist:
        with open(reward_10_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards_by_10)
        with open(reward_100_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards_by_100)
        with open(reward_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards)
    else:
        with open(reward_10_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards_by_10)
        with open(reward_100_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards_by_100)
        with open(reward_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rewards)


def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect the hyperparameters from user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                   # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)      # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)          # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=float)      # Max time steps to run environment for
    parser.add_argument("--save_models", default=True, action="store_true")            # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)         # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)           # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)          # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)              # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)       # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)         # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)            # Frequency of delayed policy updates
    parser.add_argument("--model_path", default="./models")              # Where models are saved
    parser.add_argument("--res_dir", default="./results")              # Where results are saved
    parser.add_argument("--eval_length", default=10, type=int)           # The number of episodes to evaluate for
    parser.add_argument("--agent", default="TD3", type=str)              # The agent to use ("TD3" or "DDPG")
    parser.add_argument("--save_freq", default=100000, type=int)              # How often to save the model"
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    if args.save_models and not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Set seeds
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set the number of actions and observations
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.agent == "TD3":
        agent = TD3(state_dim, action_dim, max_action, hidden_dim=256, buffer_size=int(1e6), batch_size=args.batch_size, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_freq=args.policy_freq)
    else:
        agent = DDPGAgent(state_dim, action_dim, buffer_size=int(1e6), batch_size=args.batch_size, lr_actor=0.0001, lr_critic=0.0001)
    print("---------------------------------------")
    print(f"Starting training: {args.env_name}")
    print("---------------------------------------")
    
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = False
    episode_reward = 0
    episode_timesteps = 0
    rewards_by_100 = [['episode_num', 'episode_timesteps', 'step', 'reward', 'avg_reward']]
    rewards_by_10 = [['step', 'avg_reward']]
    rewards = [['episode_num', 'step', 'reward']]
    queue10 = deque(maxlen=10)
    queue100 = deque(maxlen=100)
    exist = False  # Used to indicate whether the results file already exists
    obs = env.reset()

    while total_timesteps < args.max_timesteps:
        total_timesteps += 1
        timesteps_since_eval += 1
        episode_timesteps += 1

        # If the episode is done
        if done:
            if args.agent == "TD3":
                if args.env_name == 'Ant-v2' or args.env_name == 'Humanoid-v2':
                    agent.update(episode_timesteps, 100)
            print(f"Total T: {total_timesteps} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward}")
            queue10.append(episode_reward)
            queue100.append(episode_reward)
            rewards_by_10.append([total_timesteps, np.mean(queue10)])
            rewards_by_100.append([episode_num, episode_timesteps, total_timesteps, episode_reward, np.mean(queue100)])
            rewards.append([episode_num, total_timesteps, episode_reward])
            
            # Reset environment
            obs = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            done = False
            episode_num += 1

        # Before start_timesteps, we play random actions
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)
            if args.agent == "TD3":
                if args.expl_noise != 0:
                    action = (action + np.random.normal(0, args.expl_noise, size=action_dim)).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        agent.memory.add(obs, new_obs, action, reward, done_bool)
        
        obs = new_obs
        episode_reward += reward
        if total_timesteps >= args.start_timesteps:
            if args.agent == "TD3":
                if args.env_name != "Ant-v2" and args.env_name != "Humanoid-v2":
                    agent.update(1, args.batch_size)
            else:
                agent.update()

        if total_timesteps % args.save_freq == 0:
            if args.save_models:
                agent.save(args.env_name, args.model_path)
            output_to_file(rewards_by_100, rewards_by_10, rewards, args.res_dir, args.agent, args.env_name, exist)
            exist = True
            rewards_by_10 = []
            rewards_by_100 = []
            rewards = []

    if args.save_models:
        agent.save(args.env_name, args.model_path)

if __name__ == "__main__":
    main()

