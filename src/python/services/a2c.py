import numpy as np
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

class A2CAgent:
    """A2C Agent for test case prioritization"""
    def __init__(self, state_size, action_size, hidden_dim=64, learning_rate=0.0007, 
                gamma=0.99, device='auto'):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        self.model = None
        self.vec_env = None

        if torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")
    
    def set_env(self, env):
        """Set environment and create A2C model"""
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Create policy kwargs
        policy_kwargs = dict(
            net_arch=[self.hidden_dim, self.hidden_dim]
        )
        
        # Create A2C model
        self.model = A2C(
            "MlpPolicy", 
            self.vec_env, 
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=1,  # Set to 1 to see training progress
            device=self.device,
        )
        
        return self
    
    def act(self, state):
        """Select action using the policy"""
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        # Handle different state formats and ensure float32 dtype
        if isinstance(state, np.ndarray):
            # Convert explicitly to float32 to avoid dtype issues with MPS
            observation = state.astype(np.float32)
        else:
            # For other state types
            observation = state
        
        # Predict action
        action, _ = self.model.predict(observation, deterministic=True)
        
        return action
    
    def learn(self, total_timesteps, callback=None):
        """Train the agent"""
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self
    
    def save(self, path):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        self.model.save(path)
    
    def load(self, path):
        """Load model"""
        self.model = A2C.load(path)
        return self