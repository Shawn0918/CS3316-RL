import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a class for the Critic network
class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initialize the CriticNetwork class

        Args:
            state_dim (int): dimension of the state
            action_dim (int): dimension of the action
            hidden_dim (int): dimension of the hidden layers
        """
        super(CriticNetwork, self).__init__()

        # Define the layers of the network
        self.linear1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        """
        Forward pass of the network

        Args:
            state (numpy array): current state
            action (numpy array): action taken

        Returns:
            q_value (numpy array): Q-value for the state-action pair
        """
        # Concatenate the state and action
        state_action = torch.cat([state, action], 1)

        # Pass the state-action pair through the first layer
        x = self.linear1(state_action)
        x = F.relu(x)

        # Pass the output of the first layer through the second layer
        x = self.linear2(x)
        x = F.relu(x)

        # Pass the output of the second layer through the third layer
        x = self.linear3(x)

        # Return the Q-value
        return x
    
# Create a class for the Actor network
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        """
        Initialize the ActorNetwork class

        Args:
            state_dim (int): dimension of the state
            action_dim (int): dimension of the action
            hidden_dim (int): dimension of the hidden layers
            max_action (int): maximum value of the action
        """
        super(ActorNetwork, self).__init__()

        # Define the layers of the network
        self.linear1 = torch.nn.Linear(state_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, action_dim)

        # Define the maximum action
        self.max_action = max_action

    def forward(self, state):
        """
        Forward pass of the network

        Args:
            state (numpy array): current state

        Returns:
            action (numpy array): action taken
        """
        # Pass the state through the first layer
        x = self.linear1(state)
        x = F.relu(x)

        # Pass the output of the first layer through the second layer
        x = self.linear2(x)
        x = F.relu(x)

        # Pass the output of the second layer through the third layer
        x = self.linear3(x)

        # Apply a tanh activation to the output of the third layer
        action = self.max_action * torch.tanh(x)

        # Return the action
        return action
    
# Create a class for the TD3 agent
class TD3(object):
    # Initialize the class
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, buffer_size, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim, max_action).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.memory = utils.ReplayBuffer(buffer_size)
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.batch_size = batch_size

    # Select an action
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    

    # Train the agent
    def update(self, n_iter, batch_size=100):
        for i in range(n_iter):
            self.total_it += 1

            # Sample a batch of transitions from the replay buffer
            state, action, next_state, reward, done = self.memory.sample(batch_size)

            # Convert the numpy arrays to PyTorch tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            
            with torch.no_grad():
                # Select the action according to the policy and add clipped noise
                # Select next action according to target policy:
                '''noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(device)
                noise = noise.clamp(-self.noise_clip,self.noise_clip)
                next_action = (self.actor_target(next_state) + noise)
                next_action = next_action.clamp(-self.max_action, self.max_action)
                # Select action according to policy and add clipped noise'''
                noise = (
                    torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (
                    self.actor_target(next_state) + noise
                ).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                q1_next = self.critic1_target(next_state, next_action)
                q2_next = self.critic2_target(next_state, next_action)
                q_next = torch.min(q1_next, q2_next)
                q_target = reward + ((1 - done) * self.discount * q1_next).detach()

            # Get the current Q estimates from critic1
            current_Q1 = self.critic1(state, action)

            # Compute the critic loss
            critic1_loss = F.mse_loss(current_Q1, q_target)

            # Optimize the critic1
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            # Get the current Q estimates from critic2
            current_Q2 = self.critic2(state, action)

            # Compute the critic loss
            critic2_loss = F.mse_loss(current_Q2, q_target)

            # Optimize the critic2
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic1.state_dict(), '%s/%s_critic1.pth' % (directory, filename))
        torch.save(self.critic2.state_dict(), '%s/%s_critic2.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic1.load_state_dict(torch.load('%s/%s_critic1.pth' % (directory, filename)))
        self.critic2.load_state_dict(torch.load('%s/%s_critic2.pth' % (directory, filename)))

