import numpy as np
import pandas as pd
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import time
import os

class DQNNetwork(nn.Module):
    """
    Neural network for DQN with additional layer.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # From hidden_dim to hidden_dim//2
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)  # From hidden_dim//2 to output_dim
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    """
    DQN Agent for test case prioritization.
    """
    def __init__(self, state_size, action_size, hidden_dim=256, learning_rate=0.0001, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.98,
                 memory_size=10000, batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state
            action_size: Dimension of the action space
            hidden_dim: Hidden dimension of the network
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            device: Device to use (cuda or cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.device = device
        
        # Create Q networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_dim).to(device)
        self.update_target_network()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # For stats
        self.loss_history = []
    
    def update_target_network(self):
        """Update the target network with the Q network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # def remember(self, state, action, reward, next_state, done):
    #     """
    #     Add a transition to the replay memory.
        
    #     Args:
    #         state: Current state
    #         action: Action taken
    #         reward: Reward received
    #         next_state: Next state
    #         done: Whether the episode is done
    #     """
    #     self.memory.append((state, action, reward, next_state, done))

    def remember(self, state, action, reward, next_state, done):
        """Add a transition to the replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    # def process_state(self, state_dict):
    #     """Process the state dictionary from the environment into a tensor."""
    #     # Instead of directly using the state features, pad/truncate to fixed size
    #     max_size = self.state_size  # This should be a fixed size
        
    #     # Concatenate all relevant features
    #     test_features = state_dict["test_features"]  # Shape: (max_tests, feature_dim)
    #     available_mask = state_dict["available_mask"]  # Shape: (max_tests,)
        
    #     # Flatten and concatenate
    #     flattened_features = test_features.reshape(-1)
    #     state = np.concatenate([flattened_features, available_mask])
        
    #     # Ensure consistent size
    #     if len(state) < max_size:
    #         # Pad with zeros
    #         state = np.pad(state, (0, max_size - len(state)), 'constant')
    #     elif len(state) > max_size:
    #         # Truncate
    #         state = state[:max_size]
            
    #     return torch.FloatTensor(state).to(self.device)

    def process_state(self, state):
        """Process different state formats into tensors"""
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:  # Vector state (pointwise)
                return torch.FloatTensor(state).to(self.device)
            else:  # Matrix state (listwise)
                flattened_state = state.reshape(-1)
                
                # Ensure consistent size
                if len(flattened_state) < self.state_size:
                    state_tensor = np.pad(flattened_state, (0, self.state_size - len(flattened_state)), 'constant')
                else:
                    state_tensor = flattened_state[:self.state_size]
                    
                return torch.FloatTensor(state_tensor).to(self.device)
        else:
            # Original environment with dictionary state
            test_features = state["test_features"]
            available_mask = state["available_mask"]
            flattened_state = np.concatenate([test_features.reshape(-1), available_mask])
            
            # Ensure consistent size
            if len(flattened_state) < self.state_size:
                state_tensor = np.pad(flattened_state, (0, self.state_size - len(flattened_state)), 'constant')
            else:
                state_tensor = flattened_state[:self.state_size]
                
            return torch.FloatTensor(state_tensor).to(self.device)
    
    # def act(self, state, epsilon=None):
    #     """
    #     Select an action using an epsilon-greedy policy.
        
    #     Args:
    #         state: Current state
    #         epsilon: Exploration rate (use agent's epsilon if None)
            
    #     Returns:
    #         Selected action
    #     """
    #     if epsilon is None:
    #         epsilon = self.epsilon
            
    #     # Process state
    #     state_tensor = self.process_state(state)
        
    #     # With probability epsilon, select a random action
    #     if np.random.rand() <= epsilon:
    #         # Only choose from available tests
    #         available_indices = np.where(state["available_mask"] == 1)[0]
    #         if len(available_indices) > 0:
    #             return np.random.choice(available_indices)
    #         else:
    #             return 0  # Fallback
        
    #     # Otherwise, select the action with the highest Q value
    #     self.q_network.eval()
    #     with torch.no_grad():
    #         q_values = self.q_network(state_tensor)
    #     self.q_network.train()
        
    #     # Mask out unavailable actions by setting their Q values to a large negative number
    #     masked_q_values = q_values.cpu().numpy()
    #     masked_q_values[state["available_mask"] == 0] = -1e9
    #     return np.argmax(masked_q_values)

    def act(self, state, epsilon=None):
        """Select an action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
            
        # Choose random action with probability epsilon
        r = np.random.rand()
        if r <= self.epsilon:
            # Handle different environment types
            if isinstance(state, np.ndarray):
                # For matrix/vector state (listwise/pointwise)
                if len(state.shape) == 1:  # Vector state (pointwise)
                    # For pointwise, return a random priority between 0 and 1
                    return np.random.uniform(0, 1, (1,))
                else:  # Matrix state (listwise)
                    # Find indices that don't have padding value (-1)
                    available_indices = np.where(~np.all(state == -1, axis=1))[0]
                    if len(available_indices) > 0:
                        return np.random.choice(available_indices)
                    else:
                        return 0
            else:
                # For dictionary state (original environment)
                available_indices = np.where(state["available_mask"] == 1)[0]
                if len(available_indices) > 0:
                    return np.random.choice(available_indices)
                else:
                    return 0
        
        # Process state based on its type
        state_tensor = self.process_state(state)
        
        # Get Q values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        # Determine action based on environment type
        if isinstance(state, np.ndarray):
            if len(state.shape) == 1:  # Vector state (pointwise)
                # For pointwise, clamp to 0-1 for priority
                action_value = np.clip(q_values.cpu().numpy()[0], 0, 1)
                return np.array([action_value], dtype=np.float32)
            else:  # Matrix state (listwise)
                # Mask tests with padding value
                masked_q_values = q_values.cpu().numpy()
                for i in range(len(state)):
                    if np.all(state[i] == -1):  # If all values are padding
                        masked_q_values[i] = -1e9
                return np.argmax(masked_q_values)
        else:
            # For original environment
            masked_q_values = q_values.cpu().numpy()
            masked_q_values[state["available_mask"] == 0] = -1e9
            return np.argmax(masked_q_values)
        
    def replay(self):
        """Train the agent by replaying experiences from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(self.process_state(state))
            # Handle different action types
            if isinstance(action, np.ndarray) and action.shape[0] == 1:
                # For continuous actions (pointwise)
                actions.append(float(action[0]))
            else:
                # For discrete actions (listwise, pairwise)
                actions.append(int(action))
            rewards.append(reward)
            next_states.append(self.process_state(next_state))
            dones.append(done)
        
        states = torch.stack(states)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Handle different action types for Q-value computation
        if isinstance(minibatch[0][1], np.ndarray) and minibatch[0][1].shape[0] == 1:
            # For continuous actions (pointwise)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            # For continuous actions, we directly compare the output value
            current_q_values = self.q_network(states)[:, 0]
            
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_states)[:, 0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute MSE loss
            loss = F.mse_loss(current_q_values, target_q_values)
        else:
            # For discrete actions (listwise, pairwise)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            
            # Compute current Q values
            current_q_values = self.q_network(states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values)
        
        self.loss_history.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path):
        """
        Save the agent's Q network.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.q_network.state_dict(), path)
    
    def load(self, path):
        """
        Load the agent's Q network.
        
        Args:
            path: Path to load the model
        """
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()