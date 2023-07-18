import gym
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from utils import WarpFrame
from cpprb import PrioritizedReplayBuffer, ReplayBuffer
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, NoopResetEnv, MaxAndSkipEnv

# Use argparse to get the arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='VideoPinball-ramNoFrameskip-v4')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gamma', type=float, default=0.99) 
parser.add_argument('--eps_start', type=float, default=1.0)
parser.add_argument('--eps_end', type=float, default=1e-2)
parser.add_argument('--eps_decay', type=float, default=500) 
parser.add_argument('--print_freq', type=int, default=100) # The frequency to print the log
parser.add_argument('--eps_fraction', type=float, default=0.1) # The fraction of the total number of steps to anneal epsilon
parser.add_argument('--target_update', type=int, default=1000) # The frequency to update the target network
parser.add_argument('--eval_freq', type=int, default=10000) # The frequency to evaluate the performance
parser.add_argument('--memory_capacity', type=int, default=100000) # The capacity of replay buffer
parser.add_argument('--learning_freq', type=int, default=4) # The number of steps between every optimization step
parser.add_argument('--lr', type=float, default=0.0000625) 
parser.add_argument('--max_step', type=int, default=10000000) # The maximum number of steps to run the environment
parser.add_argument('--save_freq', type=int, default=100000) # The frequency (measured in the number of steps) to save the model
parser.add_argument('--formal_start', type=int, default=10000) # The number of steps to take random actions before start learning
parser.add_argument('--max_episode_steps', type=int, default=100000) # The maximum number of steps for each episode
parser.add_argument('--model_path', type=str, default='Duel_DQN_model/')
parser.add_argument('--data_path', type=str, default='Duel_DQN_result/')
parser.add_argument('--prioritized_replay', type=bool, default=False)
parser.add_argument('--n_step', type=int, default=1)
args = parser.parse_args()

# Define the hyperparameters
PRINT_FREQ = args.print_freq
FRAC = args.eps_fraction
ENV_NAME = args.env
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
TARGET_UPDATE = args.target_update
MEMORY_CAPACITY = args.memory_capacity
LR = args.lr
MAX_STEP = args.max_step
EVAL_FREQ = args.eval_freq
FORMAL_START = args.formal_start
SAVE_FREQ = args.save_freq
MAX_EPISODE_STEPS = args.max_episode_steps
LEARN_FREQ = args.learning_freq
# Define the path to save the model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# Define the path to save the data
if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Define the environment
if ENV_NAME == 'VideoPinball-ramNoFrameskip-v4':
    env = gym.make(ENV_NAME, obs_type='image')
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
else:
    env = make_atari(ENV_NAME)
    env = wrap_deepmind(env)

obs_size = env.observation_space.shape
obs_size = (obs_size[2],) + (obs_size[0], obs_size[1])
n_actions = env.action_space.n

nstep = args.n_step
rb_dict = {"obs":{"shape": obs_size},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": obs_size},
            "done": {}}
Nstep = {"size": nstep, "rew": "rew", "next": "next_obs"}

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Initialize the weights and biases of the convolutional layers
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
                
        self.a_layer = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.V_layer = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Initialize the weights and biases of the fully connected layers
        for layer in [self.a_layer, self.V_layer]:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    nn.init.kaiming_normal_(sub_layer.weight)
                    nn.init.constant_(sub_layer.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.V_layer(x) + self.a_layer(x) - self.a_layer(x).mean()
        return x 

# Define the Double DQN agent
class DuelDQNAgent():
    def __init__(self, obs_size, n_actions):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.policy_net = DQN(obs_size, n_actions).to(device)
        self.target_net = DQN(obs_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.policy_net.apply(self.policy_net.init_weights)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        if args.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY, rb_dict, Nstep=Nstep)
        else:
            self.memory = ReplayBuffer(MEMORY_CAPACITY, rb_dict)
        self.steps_done = 0
        self.loss_stat = [["step_done", "loss"]]
        self.rewards_stat = [["episode", "rewards"]]
        self.rewards_by_10 = [["step_done", "rewards"]]
        self.rewards_by_100 = [["step_done", "rewards"]]
        self.epsilon = EPS_START
        self.beta = 0.4
        self.beta_step = (1 - self.beta) / MAX_STEP
        self.nstep = args.n_step

    def optimize_model(self):
        batch = self.memory.sample(BATCH_SIZE)
        states = torch.from_numpy(batch['obs']).to(device).type(dtype)
        actions = torch.from_numpy(batch['act']).to(device).type(dlongtype)
        rewards = torch.from_numpy(batch['rew']).to(device).type(dtype)
        next_states = torch.from_numpy(batch['next_obs']).to(device).type(dtype)
        dones = torch.from_numpy(batch['done']).to(device).type(dtype)

        Q_values = torch.gather(self.policy_net(states), 1, actions)
        _, max_actions = self.policy_net(next_states).max(1)
        with torch.no_grad():
            expected_Q_values = rewards + GAMMA*(1 - dones)*torch.gather(self.target_net(next_states), 1, max_actions.unsqueeze(1).to(device))
        loss = nn.MSELoss()(Q_values.squeeze(), expected_Q_values.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) 
        self.optimizer.step()
        
        return loss.item()
    
    def optimize_model_prioritized(self):

        batch = self.memory.sample(BATCH_SIZE, self.beta)
        self.beta += self.beta_step
        weights = torch.from_numpy(batch["weights"].ravel()).to(device).type(dtype)
        states = torch.from_numpy(batch['obs']).to(device).type(dtype)
        actions = torch.from_numpy(batch['act']).to(device).type(dlongtype)
        rewards = torch.from_numpy(batch['rew']).to(device).type(dtype)
        next_states = torch.from_numpy(batch['next_obs']).to(device).type(dtype)
        dones = torch.from_numpy(batch['done']).to(device).type(dtype)

        Q_values = torch.gather(self.policy_net(states), 1, actions)
        _, max_actions = self.policy_net(next_states).max(1)
        with torch.no_grad():
            expected_Q_values = rewards + GAMMA**nstep*(1 - dones)*torch.gather(self.target_net(next_states), 1, max_actions.unsqueeze(1).to(device))
        # loss = nn.MSELoss()(Q_values.squeeze(), expected_Q_values.squeeze())
        squared_diff = (Q_values - expected_Q_values) ** 2
        weighted_squared_diff = squared_diff * weights
        loss = weighted_squared_diff.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        absTD = torch.abs(expected_Q_values - Q_values) 
        # Convert the tensor to numpy array
        absTD = absTD.squeeze().cpu().detach().numpy()
        self.memory.update_priorities(batch["indexes"], absTD)
        return loss.item()
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def initialize(self):
        state = env.reset()
        obs = state.transpose(2,0,1)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device).type(dtype)
        episode_reward = 0
        loss = 0
        done = False
        return obs, obs_tensor, episode_reward, loss, done

    def train(self):
        print("---------------------------------------")
        print(f"Starting training: {ENV_NAME}")
        print("---------------------------------------")
        obs, obs_tensor, episode_reward, loss, done = self.initialize()
        episode = 0
        episode_steps = 0
        deque_100 = deque(maxlen=100)
        deque_10 = deque(maxlen=10)
        eps_timesteps = FRAC * float(MAX_STEP)
        while self.steps_done <= MAX_STEP:
            fraction = min(1.0, float(self.steps_done) / eps_timesteps)
            self.epsilon = EPS_START + fraction * (EPS_END - EPS_START)
            if random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.policy_net(obs_tensor).max(1)[1].item()
            next_state, reward, done, _ = env.step(action) 
            # reward = np.clip(reward, -1.0, 1.0) 
            next_obs = next_state.transpose(2,0,1)
            episode_reward += reward

            next_obs_tensor = torch.from_numpy(next_obs).unsqueeze(0).to(device).type(dtype)
            self.memory.add(obs=obs, act=action, rew=reward, next_obs=next_obs, done=done)
            obs_tensor = next_obs_tensor
            obs = next_obs
            self.steps_done += 1
            if self.steps_done >= FORMAL_START and self.steps_done % LEARN_FREQ == 0 and self.steps_done >= BATCH_SIZE:
                if args.prioritized_replay:
                    loss = self.optimize_model_prioritized()
                else:
                    loss = self.optimize_model()
                self.loss_stat.append([self.steps_done, loss])
            if self.steps_done >= FORMAL_START and self.steps_done % TARGET_UPDATE == 0:
                self.update_target_network()
            if self.steps_done % SAVE_FREQ == 0:
                self.output_res()
            if done:
                self.memory.on_episode_end() # Reset the buffer for next episode
                episode += 1
                episode_steps = 0
                deque_100.append(episode_reward)
                deque_10.append(episode_reward)
                self.rewards_by_100.append([self.steps_done, np.mean(deque_100)])
                self.rewards_by_10.append([self.steps_done, np.mean(deque_10)])
                self.rewards_stat.append([self.steps_done, episode_reward]) 
                if self.steps_done % PRINT_FREQ == 0:
                    print("Episode:", episode, "Reward:", episode_reward)
                obs, obs_tensor, episode_reward, loss, done = self.initialize()

        print("Training completes")

    def output_res(self):
        data1 = np.array(self.loss_stat)
        data2 = np.array(self.rewards_stat)

        rewards_by_100 = np.array(self.rewards_by_100)
        rewards_by_10 = np.array(self.rewards_by_10)

        rewards_by_100_table = pd.DataFrame(data=rewards_by_100[1:,:], columns=rewards_by_100[0,:])
        rewards_by_10_table = pd.DataFrame(data=rewards_by_10[1:,:], columns=rewards_by_10[0,:])
        rewards_by_100_table.to_csv(os.path.join(args.data_path, ENV_NAME + "_rewards_by_100_table.csv"))
        rewards_by_10_table.to_csv(os.path.join(args.data_path, ENV_NAME + "_rewards_by_10_table.csv"))

        loss_table = pd.DataFrame(data=data1[1:,:], columns=data1[0,:])
        rewards_table = pd.DataFrame(data=data2[1:,:], columns=data2[0,:])
 
        loss_table.to_csv(os.path.join(args.data_path, "loss_table.csv"))
        rewards_table.to_csv(os.path.join(args.data_path, ENV_NAME + "_rewards_table.csv"))
        
        torch.save(self.policy_net.state_dict(), os.path.join(args.model_path, ENV_NAME + "_Dueling_DQN"+ str(self.steps_done) + ".pth"))

if __name__ == "__main__":
    agent = DuelDQNAgent(obs_size, n_actions)

    agent.train()