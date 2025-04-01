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
from dqn import DQNNetwork, DQNAgent
from env import TestPrioritizationEnv, FixedTestPrioritizationEnv
from rl_eval import visualize_build_learning, evaluate_dqn, visualize_results, analyze_agent_behavior

# def main():
#     """
#     Main function to run the test prioritization RL system.
#     """
#     import argparse
#     parser = argparse.ArgumentParser(description='Test Prioritization using RL')
#     parser.add_argument('--data_path', type=str, required=True, 
#                        help='Path to the CSV file containing test data')
#     parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'analyze'],
#                        help='Mode to run the system in')
#     parser.add_argument('--model_path', type=str, default=None,
#                        help='Path to load a trained model (for evaluate/analyze mode)')
#     parser.add_argument('--num_episodes', type=int, default=1000,
#                        help='Number of episodes to train for')
#     parser.add_argument('--hidden_dim', type=int, default=128,
#                        help='Hidden dimension of the DQN network')
#     parser.add_argument('--learning_rate', type=float, default=0.001,
#                        help='Learning rate for the optimizer')
#     parser.add_argument('--gamma', type=float, default=0.99,
#                        help='Discount factor')
#     parser.add_argument('--epsilon', type=float, default=1.0,
#                        help='Initial exploration rate')
#     parser.add_argument('--epsilon_min', type=float, default=0.01,
#                        help='Minimum exploration rate')
#     parser.add_argument('--epsilon_decay', type=float, default=0.995,
#                        help='Exploration rate decay')
#     parser.add_argument('--save_dir', type=str, default='results',
#                        help='Directory to save results')
    
#     args = parser.parse_args()
    
#     # Create save directory
#     os.makedirs(args.save_dir, exist_ok=True)
    
#     # Preprocess data
#     print(f"Loading data from {args.data_path}...")
#     build_data = preprocess_data(args.data_path)
    
#     # Create environment
#     env = TestPrioritizationEnv(build_data)
    
#     # Calculate state size for DQN
#     sample_state = env.reset()
#     state_tensor = torch.FloatTensor(np.concatenate([
#         sample_state["test_features"].reshape(-1),
#         sample_state["available_mask"]
#     ]))
#     state_size = state_tensor.shape[0]
    
#     # Create agent
#     agent = DQNAgent(
#         state_size=state_size,
#         action_size=env.max_tests,
#         hidden_dim=args.hidden_dim,
#         learning_rate=args.learning_rate,
#         gamma=args.gamma,
#         epsilon=args.epsilon,
#         epsilon_min=args.epsilon_min,
#         epsilon_decay=args.epsilon_decay
#     )
    
#     # Run in the specified mode
#     if args.mode == 'train':
#         print(f"Training DQN agent for {args.num_episodes} episodes...")
#         training_metrics = train_dqn(
#             env=env,
#             agent=agent,
#             num_episodes=args.num_episodes,
#             save_dir=os.path.join(args.save_dir, 'models'),
#             save_metrics_to_csv=True 
#         )
        
#         # Visualize results
#         print("Visualizing results...")
#         visualize_results(
#             training_metrics=training_metrics,
#             save_dir=os.path.join(args.save_dir, 'figures')
#         )
        
#         # Save metrics
#         with open(os.path.join(args.save_dir, 'training_metrics.pkl'), 'wb') as f:
#             import pickle
#             pickle.dump(training_metrics, f)
            
#     elif args.mode == 'evaluate':
#         # Load model
#         if args.model_path is None:
#             args.model_path = os.path.join(args.save_dir, 'models', 'dqn_test_prioritization.pt')
            
#         print(f"Loading model from {args.model_path}...")
#         agent.load(args.model_path)
        
#         # Evaluate on all builds
#         print("Evaluating agent on all builds...")
#         results = evaluate_dqn(env, agent, build_ids=list(build_data.keys()))
        
#         # Print results
#         print(f"\nEvaluation Results:")
#         print(f"Average APFD: {results['avg_apfd']:.4f}")
#         print(f"Average Improvement: {results['avg_improvement']:.4f}")
        
#         # Save results
#         with open(os.path.join(args.save_dir, 'evaluation_results.pkl'), 'wb') as f:
#             import pickle
#             pickle.dump(results, f)
            
#     elif args.mode == 'analyze':
#         # Load model
#         if args.model_path is None:
#             args.model_path = os.path.join(args.save_dir, 'models', 'dqn_test_prioritization.pt')
            
#         print(f"Loading model from {args.model_path}...")
#         agent.load(args.model_path)
        
#         # Analyze agent behavior
#         print("Analyzing agent behavior...")
#         analysis = analyze_agent_behavior(env, agent, num_builds=10)
        
#         # Save analysis
#         with open(os.path.join(args.save_dir, 'agent_analysis.pkl'), 'wb') as f:
#             import pickle
#             pickle.dump(analysis, f)


# if __name__ == "__main__":
#     main()