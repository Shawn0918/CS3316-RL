import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import matplotlib.pyplot as plt
from utils import ReplayBuffer

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the environment
env = gym.make('Pendulum-v0')

class ActorNet(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class CriticNet(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(CriticNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPGAgent:
    def __init__(self, state_size, action_size, buffer_size, batch_size=128, gamma=0.99, tau=0.001, lr_actor=0.001, lr_critic=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        self.actor_net = ActorNet(state_size, action_size).to(device)
        self.target_actor_net = ActorNet(state_size, action_size).to(device)
        self.critic_net = CriticNet(state_size, action_size).to(device)
        self.target_critic_net = CriticNet(state_size, action_size).to(device)
        
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        
        self.memory = ReplayBuffer(buffer_size)
        
    def select_action(self, state, noise=0.1):
        state = torch.tensor(state).view(1,-1).float().to(device)
        #state = torch.from_numpy(state).float().to(device)
        self.actor_net.eval()
        with torch.no_grad():
            action = self.actor_net(state).cpu().data.numpy()
        self.actor_net.train()
        action += noise * np.random.randn(self.action_size)
        return np.squeeze(np.clip(action, -1, 1))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update(self):
        state, action, next_state, reward, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Update critic network
        Q_values = self.critic_net(state, action)
        next_actions = self.target_actor_net(next_state)
        next_Q_values = self.target_critic_net(next_state, next_actions.detach())
        target_Q_values = reward + (self.gamma * next_Q_values * (1 - done))
        critic_loss = F.mse_loss(Q_values, target_Q_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Update actor network
        actions = self.actor_net(state)
        actor_loss = -self.critic_net(state, actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Update target networks
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor_net.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic_net.state_dict(), '%s/%s_critic1.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor_net.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic_net.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
    
# Deprecated
def train_ddpg(agent, env, n_episodes=1000, max_t=1000, print_every=100, store_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    neat_res = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode % store_every == 0:
            neat_res.append(score)

    return scores, neat_res
if __name__ == "__main__":
    # Set hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DDPGAgent(state_size, action_size)
    # Train the agent
    scores, neat_res = train_ddpg(agent, env)
    np.save("ddpg.npy", np.array(scores))
    np.save("ddpg_neat.npy", np.array(neat_res))
    plt.plot(scores)
    plt.xlabel("episode")
    plt.ylabel("episode rewards")
    plt.savefig("DDPG_output.png")