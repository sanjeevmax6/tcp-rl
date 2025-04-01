import numpy as np
import torch
import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

class PPOAgent:
    """
    PPO Agent for test case prioritization.
    Wraps the Stable Baselines PPO implementation.
    """
    def __init__(self, state_size, action_size, hidden_dim=256, learning_rate=0.0007, 
                 gamma=0.99, device='auto'):
        """
        Initialize the PPO agent.
        
        Args:
            state_size: Dimension of the state (not directly used but kept for API consistency)
            action_size: Dimension of the action space (not directly used but kept for API consistency)
            hidden_dim: Hidden dimension of the network
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            device: Device to use (auto, cuda, or cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device
        
        # The actual model will be created when set_env is called
        self.model = None
        self.vec_env = None
    
    def set_env(self, env):
        """
        Set the environment and create the PPO model.
        
        Args:
            env: Gym environment
        """
        # Create a vectorized environment as required by PPO
        self.vec_env = DummyVecEnv([lambda: env])
        
        policy_kwargs = dict(
            net_arch=[self.hidden_dim, self.hidden_dim]
        )

        # Create the PPO model
        self.model = PPO(
            "MlpPolicy", 
            self.vec_env, 
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=0
        )
        
        return self
    
    def act(self, state, epsilon=None):
        """
        Select an action using the policy.
        
        Args:
            state: Current state
            epsilon: Not used in PPO but kept for API consistency
            
        Returns:
            Selected action
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        # For vectorized environments, state needs to be properly formatted
        # Convert to numpy array if needed
        if isinstance(state, np.ndarray):
            # For matrix state (listwise)
            if len(state.shape) > 1:
                # Ensure state is in correct shape for vectorized env
                # The model expects a batch dimension
                observation = state
            else:
                # For vector state (pointwise)
                # Reshape for vectorized env
                observation = state.reshape(1, -1)
        else:
            # For dictionary state, convert to numpy array
            # This would need customization based on your environment
            observation = state
        
        # Predict action using the PPO model
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Return the action directly - PPO will return the correct format
        return action
        
    def learn(self, total_timesteps, callback=None):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            callback: Optional callback function
            
        Returns:
            Trained model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        return self
    
    def save(self, path):
        """
        Save the agent's model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call set_env first.")
        
        self.model.save(path)
    
    def load(self, path):
        """
        Load the agent's model.
        
        Args:
            path: Path to load the model
        """
        self.model = PPO.load(path)
        return self