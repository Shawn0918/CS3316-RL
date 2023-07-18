import numpy as np
import torch
import gym

class ReplayBuffer(object):
    def __init__(self, max_size):
        """
        Initialize a ReplayBuffer object

        Args:
            max_size (int): maximum size of the buffer
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, next_state, action, reward, done):
        """
        Add a new experience to the buffer

        Args:
            state (numpy array): current state
            next_state (numpy array): next state
            action (numpy array): action taken
            reward (float): reward received
            done (int): whether the episode terminated
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, next_state, action, reward, done)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer

        Args:
            batch_size (int): size of the batch to sample

        Returns:
            states (numpy array): batch of states
            next_states (numpy array): batch of next states
            actions (numpy array): batch of actions
            rewards (numpy array): batch of rewards
            dones (numpy array): batch of done flags
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []
        for i in ind:
            s, s_, a, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            next_states.append(np.array(s_, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)
    
    def save(self, filename):
        """
        Save the buffer to a file

        Args:
            filename (str): name of the file to save to
        """
        np.save(filename, self.storage)

    def load(self, filename):
        """
        Load the buffer from a file

        Args:
            filename (str): name of the file to load from
        """
        self.storage = np.load(filename, allow_pickle=True).tolist()

class ReplayBufferHindsight(object):
    def __init__(self, max_size=1e6):
        """
        Initialize a ReplayBufferHindsight object

        Args:
            max_size (int): maximum size of the buffer
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, next_state, action, reward, done, goal_state, goal_next_state):
        """
        Add a new experience to the buffer

        Args:
            state (numpy array): current state
            next_state (numpy array): next state
            action (numpy array): action taken
            reward (float): reward received
            done (int): whether the episode terminated
            goal_state (numpy array): goal state
            goal_next_state (numpy array): goal next state
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = (state, next_state, action, reward, done, goal_state, goal_next_state)
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append((state, next_state, action, reward, done, goal_state, goal_next_state))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer

        Args:
            batch_size (int): size of the batch to sample

        Returns:
            states (numpy array): batch of states
            next_states (numpy array): batch of next states
            actions (numpy array): batch of actions
            rewards (numpy array): batch of rewards
            dones (numpy array): batch of done flags
            goal_states (numpy array): batch of goal states
            goal_next_states (numpy array): batch of goal next states
        """
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones, goal_states, goal_next_states = [], [], [], [], [], [], []
        for i in ind:
            s, s_, a, r, d, g_s, g_s_ = self.storage[i]
            states.append(np.array(s, copy=False))
            next_states.append(np.array(s_, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))
            goal_states.append(np.array(g_s, copy=False))    
            goal_next_states.append(np.array(g_s_, copy=False))
        return np.array(states), np.array(next_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1), np.array(goal_states), np.array(goal_next_states)
    
    def save(self, filename):
        """
        Save the buffer to a file

        Args:
            filename (str): name of the file to save to
        """
        np.save(filename, self.storage)

    def load(self, filename):
        """
        Load the buffer from a file

        Args:
            filename (str): name of the file to load from
        """
        self.storage = np.load(filename, allow_pickle=True).tolist()

