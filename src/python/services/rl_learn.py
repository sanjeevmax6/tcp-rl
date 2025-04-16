import pandas as pd
import numpy as np
import os
import random
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from .dqn import DQNNetwork, DQNAgent
from .env import TestPrioritizationEnv, FixedTestPrioritizationEnv, ListwiseTestPrioritizationEnv, PairwiseTestPrioritizationEnv, PointwiseTestPrioritizationEnv
from .rl_eval import visualize_build_learning, evaluate_agent, visualize_results, analyze_agent_behavior
from .ppo import PPOAgent
from . import globals
from .dash_display import init_dashboard 
import random

def select_and_print_features(build_data, prefixes=["REC", "TES_PRO", "TES_COM"]):
    """
    Selects features based on specified prefixes and prints a summary table.
    """
    # Get all column names from the first build
    sample_build_id = list(build_data.keys())[0]
    all_columns = build_data[sample_build_id].columns.tolist()
    
    # Filter columns by prefixes
    selected_features = []
    for col in all_columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected_features.append(col)
    
    # Exclude non-feature columns that might have matching prefixes
    exclude_columns = ['Build', 'Test', 'Verdict', 'Duration']
    selected_features = [col for col in selected_features if col not in exclude_columns]
    
    # Group features by prefix
    feature_groups = {}
    for prefix in prefixes:
        feature_groups[prefix] = [col for col in selected_features if col.startswith(prefix)]
    
    # Print summary table
    print("\n" + "="*80)
    print("FEATURE SELECTION SUMMARY".center(80))
    print("="*80)
    print(f"Total selected features: {len(selected_features)}")
    print("-"*80)
    
    for prefix, features in feature_groups.items():
        print(f"{prefix} Features ({len(features)}):")
        # Display in multiple columns for better readability
        features_per_row = 3
        for i in range(0, len(features), features_per_row):
            row_features = features[i:i+features_per_row]
            print("  " + "  |  ".join(row_features))
        print("-"*80)
    
    print("="*80 + "\n")
    
    return selected_features
    
def train_dqn(env, agent, num_episodes=1000, update_frequency=10, eval_frequency=100, 
              save_dir='models', model_name='dqn_test_prioritization.pt', save_metrics_to_csv=True):
    """
    Train the DQN agent.
    
    Args:
        env: Environment
        agent: DQN agent
        num_episodes: Number of episodes to train for
        update_frequency: How often to update the target network
        eval_frequency: How often to evaluate the agent
        save_dir: Directory to save the model
        model_name: Name of the model file
        
    Returns:
        Dictionary of training metrics.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_apfdcs = []
    episode_improvements = []
    episode_build_ids = [] 
    eval_apfds = []
    build_metrics = {}
    
    # Add dictionaries to track per-build metrics
    build_apfd_history = {}
    build_apfdc_history = {}
    build_improvement_history = {}
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print("Starting training...")
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        build_id = env.current_build  # Get the build ID right after reset
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.replay()
            
            # Update target network
            if episode % update_frequency == 0 and done:
                agent.update_target_network()
        
        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        apfdc = env.build_metrics[build_id]['apfdc']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_apfdcs.append(apfdc)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_apfdc_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_apfdc_history[build_id].append(apfdc)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        if episode % 10 == 0:
            # Calculate averages for builds with multiple episodes
            avg_build_improvements = {}
            for b_id, improvements in build_improvement_history.items():
                if len(improvements) >= 3:  # Only consider builds with at least 3 episodes
                    avg_build_improvements[b_id] = sum(improvements[-3:]) / 3  # Average of last 3
            
            # Find best improving builds
            best_builds = sorted(avg_build_improvements.items(), key=lambda x: x[1], reverse=True)[:3]
            
            elapsed = time.time() - start_time
            apfd_str = f"APFD: {apfd:.4f}" if apfd is not None else "APFD: N/A"
            improvement_str = f"Improvement: {improvement:.4f}" if improvement is not None else "Improvement: N/A"
            print(f"Episode {episode}/{num_episodes}, {apfd_str}, "
                f"{improvement_str}, Epsilon: {agent.epsilon:.4f}, "
                f"Time: {elapsed:.1f}s")
            
            if best_builds:
                print(f"Top 3 improving builds: " + 
                      ", ".join([f"{b_id}: {imp:.4f}" for b_id, imp in best_builds]))
        
        # Evaluate the agent
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            eval_results = evaluate_agent(env, agent, list(env.build_data.keys())[:10])
            avg_apfd = eval_results['avg_apfd']
            avg_improvement = eval_results['avg_improvement']
            eval_apfds.append(avg_apfd)
            
            print(f"\nEvaluation at episode {episode}")
            print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
            
            # Save the model if it's the best so far
            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                agent.save(os.path.join(save_dir, f"best_{model_name}"))
                print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save the final model
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")

    if save_metrics_to_csv:
        # Debug - print lengths to see what's happening
        print(f"Debug - array lengths: rewards={len(episode_rewards)}, APFDs={len(episode_apfds)}, "
            f"APFDCs={len(episode_apfdcs)}, improvements={len(episode_improvements)}, "
            f"build_ids={len(episode_build_ids)}")
        
        # Check if we have any data
        if len(episode_rewards) > 0:
            # Make sure all lists have the same length
            min_length = min(len(episode_rewards), len(episode_apfds), len(episode_apfdcs), 
                            len(episode_improvements), len(episode_build_ids))
            
            print(f"Creating DataFrame with {min_length} rows")
            
            metrics_df = pd.DataFrame({
                'Episode': range(min_length),
                'Build': episode_build_ids[:min_length],
                'Reward': episode_rewards[:min_length],
                'APFD': episode_apfds[:min_length],
                'APFDC': episode_apfdcs[:min_length],
                'Improvement': episode_improvements[:min_length]
            })
            
            # Debug - print a few rows to verify
            print(f"DataFrame sample (first 3 rows):")
            print(metrics_df.head(3))
            
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, 'training_metrics.csv')
            metrics_df.to_csv(csv_path, index=False)
            print(f"Saved training metrics to {csv_path} ({metrics_df.shape[0]} rows)")
        else:
            print("WARNING: No training metrics collected - CSV file will be empty")
    
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
        
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_apfdcs': episode_apfdcs,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'eval_apfds': eval_apfds,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }

# Fix for data preprocessing
def fix_verdict_types(build_data, fail_value=0):
    """Ensure verdict values are properly represented in the data"""
    for build_id, df in build_data.items():
        if 'Verdict' in df.columns:
            # Make sure Verdict column is numeric
            if df['Verdict'].dtype == 'object':
                print(f"Converting Verdict column for build {build_id} from {df['Verdict'].dtype} to numeric")
                # Try to convert, preserving NaN
                build_data[build_id]['Verdict'] = pd.to_numeric(df['Verdict'], errors='coerce')
    
    # Double-check failure detection after conversion
    for build_id, df in build_data.items():
        failures = (df['Verdict'] == fail_value).sum()
        if failures > 0:
            print(f"After conversion: Build {build_id} has {failures}/{len(df)} failures")
    
    return build_data

# Below are for pairwise, pointwise, and listwise (Above training functions are deprecated)

#Listwise
def train_listwise_dqn(env, agent, num_episodes=1000, update_frequency=10, eval_frequency=100, 
                       save_dir='models', model_name='listwise_dqn.pt'):
    """
    Train the DQN agent in listwise mode.
    
    Args:
        env: Listwise environment
        agent: DQN agent
        num_episodes: Number of episodes to train for
        update_frequency: How often to update the target network
        eval_frequency: How often to evaluate the agent
        save_dir: Directory to save the model
        model_name: Name of the model file
        
    Returns:
        Dictionary of training metrics.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_improvements = []
    episode_build_ids = [] 
    build_metrics = {}
    
    # Add dictionaries to track per-build metrics
    build_apfd_history = {}
    build_improvement_history = {}
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print("Starting listwise training...")
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        build_id = env.current_build
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.replay()
            
            # Update target network
            if episode % update_frequency == 0 and done:
                agent.update_target_network()

            epsilon_step = (1.0 - agent.epsilon_min) / 500
            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon - epsilon_step)
        
        globals.latest_order = env.selected_tests
        globals.episode_number = episode
        globals.original_order = env.optimal_order
        globals.failure_array = env.failure_array
        globals.subscript_env = "Listwise"

        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        # if episode % 10 == 0:
        #     elapsed = time.time() - start_time
        #     print(f"Episode {episode}/{num_episodes}, APFD: {apfd:.4f}, "
        #           f"Improvement: {improvement:.4f}, Epsilon: {agent.epsilon:.4f}, "
        #           f"Time: {elapsed:.1f}s")
        
        # Evaluate the agent
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            # Identify builds with failures
            builds_with_failures = []
            for b_id, df in env.build_data.items():
                if 'Verdict' in df.columns:
                    failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                    if failures > 0:
                        builds_with_failures.append(b_id)
            
            # Use up to 10 builds with failures for evaluation
            eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
            if eval_builds:
                eval_results = evaluate_agent(env, agent, build_ids=eval_builds)
                avg_apfd = eval_results['avg_apfd']
                avg_improvement = eval_results['avg_improvement']
                
                print(f"\nEvaluation at episode {episode}")
                print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
                
                # Save the model if it's the best so far
                if avg_improvement > best_avg_improvement:
                    best_avg_improvement = avg_improvement
                    best_model_path = os.path.join(save_dir, f"best_{os.path.basename(model_name)}")
                    agent.save(best_model_path)
                    print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save the final model
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Episode': range(len(episode_rewards)),
        'Build': episode_build_ids,
        'Reward': episode_rewards,
        'APFD': episode_apfds,
        'Improvement': episode_improvements
    })
    
    metrics_csv_path = os.path.join(save_dir, f'listwise_training_metrics{os.path.basename(model_name).replace(".pt", "")}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Visualize build learning
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }

#Pairwise
def train_pairwise_dqn(env, agent, num_episodes=1000, update_frequency=10, eval_frequency=100, 
                      save_dir='models', model_name='pairwise_dqn.pt'):
    """Train DQN agent for pairwise test prioritization"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_improvements = []
    episode_build_ids = [] 
    build_metrics = {}
    
    # Track per-build metrics
    build_apfd_history = {}
    build_improvement_history = {}
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print("Starting pairwise training...")
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        globals.original_order = env.test_cases
        build_id = env.current_build
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.replay()
            
            # Update target network
            if episode % update_frequency == 0 and done:
                agent.update_target_network()
            
            epsilon_step = (1.0 - agent.epsilon_min) / 500
            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon - epsilon_step)
        
        # globals.latest_order = env.sorted_test_cases_vector
        # globals.episode_number = episode
        # # globals.original_order = env.test_cases
        # # globals.failure_array = env.failure_array
        # globals.subscript_env = "Pairwise"
        # print("Current Pair", env.current_pair)
        # print("Test Cases", env.test_cases)
        # print("Sorted Test Cases Vector", env.sorted_test_cases_vector)
        
        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes}, APFD: {apfd:.4f}, "
                  f"Improvement: {improvement:.4f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {elapsed:.1f}s")
        
        # Evaluate the agent
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            # Find builds with failures for evaluation
            builds_with_failures = []
            for b_id, df in env.build_data.items():
                if 'Verdict' in df.columns:
                    failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                    if failures > 0:
                        builds_with_failures.append(b_id)
            
            eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
            eval_results = evaluate_agent(env, agent, build_ids=eval_builds)
            avg_apfd = eval_results['avg_apfd']
            avg_improvement = eval_results['avg_improvement']
            
            print(f"\nEvaluation at episode {episode}")
            print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
            
            # Save the model if it's the best so far
            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                best_model_path = os.path.join(save_dir, f"best_{os.path.basename(model_name)}")
                agent.save(best_model_path)
                print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save final model and metrics
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Episode': range(len(episode_rewards)),
        'Build': episode_build_ids,
        'Reward': episode_rewards,
        'APFD': episode_apfds,
        'Improvement': episode_improvements
    })
    
    metrics_csv_path = os.path.join(save_dir, f'pairwise_training_metrics{os.path.basename(model_name).replace(".pt", "")}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Visualize build learning
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }

#Pointwise
def train_pointwise_dqn(env, agent, num_episodes=1000, update_frequency=10, eval_frequency=100, 
                       save_dir='models', model_name='pointwise_dqn.pt'):
    """Train DQN agent for pointwise test prioritization"""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_improvements = []
    episode_build_ids = [] 
    build_metrics = {}
    
    # Track per-build metrics
    build_apfd_history = {}
    build_improvement_history = {}
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print("Starting pointwise training...")
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        build_id = env.current_build
        done = False
        episode_reward = 0
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.replay()
            
            # Update target network
            if episode % update_frequency == 0 and done:
                agent.update_target_network()

            epsilon_step = (1.0 - agent.epsilon_min) / 500
            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon = max(agent.epsilon_min, agent.epsilon - epsilon_step)
        
        # globals.latest_order = env.selected_tests
        # globals.episode_number = episode
        # globals.original_order = env.optimal_order
        # globals.failure_array = env.failure_array
        # globals.subscript_env = "Listwise"
        
        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        if episode % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes}, APFD: {apfd:.4f}, "
                  f"Improvement: {improvement:.4f}, Epsilon: {agent.epsilon:.4f}, "
                  f"Time: {elapsed:.1f}s")
        
        # Evaluate the agent
        if episode % eval_frequency == 0 or episode == num_episodes - 1:
            # Find builds with failures for evaluation
            builds_with_failures = []
            for b_id, df in env.build_data.items():
                if 'Verdict' in df.columns:
                    failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                    if failures > 0:
                        builds_with_failures.append(b_id)
            
            eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
            eval_results = evaluate_agent(env, agent, build_ids=eval_builds)
            avg_apfd = eval_results['avg_apfd']
            avg_improvement = eval_results['avg_improvement']
            
            print(f"\nEvaluation at episode {episode}")
            print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
            
            # Save the model if it's the best so far
            if avg_improvement > best_avg_improvement:
                best_avg_improvement = avg_improvement
                best_model_path = os.path.join(save_dir, f"best_{os.path.basename(model_name)}")
                agent.save(best_model_path)
                print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save final model and metrics
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Episode': range(len(episode_rewards)),
        'Build': episode_build_ids,
        'Reward': episode_rewards,
        'APFD': episode_apfds,
        'Improvement': episode_improvements
    })
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(save_dir, f'pointwise_training_metrics{os.path.basename(model_name).replace(".pt", "")}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Visualize build learning
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }


def train_ppo(env, agent, num_episodes=1000, eval_frequency=100, 
              save_dir='models', model_name='ppo_model.zip'):
    """
    Train the PPO agent.
    
    Args:
        env: Environment
        agent: PPO agent
        num_episodes: Number of episodes to train for
        eval_frequency: How often to evaluate the agent
        save_dir: Directory to save the model
        model_name: Name of the model file
        
    Returns:
        Dictionary of training metrics.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_improvements = []
    episode_build_ids = [] 
    build_metrics = {}
    
    # Add dictionaries to track per-build metrics
    build_apfd_history = {}
    build_improvement_history = {}
    
    # Calculate timesteps based on episodes and environment complexity
    # For PPO, we need to use timesteps instead of episodes
    avg_episode_length = 50  # Estimate average episode length
    total_timesteps = num_episodes * avg_episode_length
    
    # Create the environment with PPO
    agent.set_env(env)
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print(f"Starting PPO training for approximately {num_episodes} episodes...")
    
    # Train in batches to allow for evaluation
    timesteps_per_batch = total_timesteps // (num_episodes // eval_frequency)
    batches = total_timesteps // timesteps_per_batch
    
    for batch in range(batches):
        # Train for a batch of timesteps
        agent.learn(total_timesteps=timesteps_per_batch)
        
        # Evaluate the agent
        current_episode = batch * eval_frequency
        
        # Reset environment to evaluate
        state = env.reset()
        build_id = env.current_build
        done = False
        episode_reward = 0
        
        # Run one episode for evaluation
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Batch {batch+1}/{batches}, APFD: {apfd:.4f}, "
              f"Improvement: {improvement:.4f}, Time: {elapsed:.1f}s")
        
        # Find builds with failures for evaluation
        builds_with_failures = []
        for b_id, df in env.build_data.items():
            if 'Verdict' in df.columns:
                failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                if failures > 0:
                    builds_with_failures.append(b_id)
        
        eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
        eval_results = evaluate_agent(env, agent, build_ids=eval_builds)
        avg_apfd = eval_results['avg_apfd']
        avg_improvement = eval_results['avg_improvement']
        
        print(f"\nEvaluation at batch {batch+1}")
        print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
        
        # Save the model if it's the best so far
        if avg_improvement > best_avg_improvement:
            best_avg_improvement = avg_improvement
            agent.save(os.path.join(save_dir, f"best_{model_name}"))
            print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save the final model
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Batch': range(len(episode_rewards)),
        'Build': episode_build_ids,
        'Reward': episode_rewards,
        'APFD': episode_apfds,
        'Improvement': episode_improvements
    })
    
    metrics_csv_path = os.path.join(save_dir, f'ppo_training_metrics{os.path.basename(model_name).replace(".zip", "")}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Visualize build learning
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }


def train_a2c(env, agent, num_episodes=1000, eval_frequency=100, 
             save_dir='models', model_name='a2c_model.zip'):
    """
    Train the A2C agent.
    
    Args:
        env: Environment
        agent: A2C agent
        num_episodes: Number of episodes to train for
        eval_frequency: How often to evaluate the agent
        save_dir: Directory to save the model
        model_name: Name of the model file
        
    Returns:
        Dictionary of training metrics.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_apfds = []
    episode_improvements = []
    episode_build_ids = [] 
    build_metrics = {}
    
    # Add dictionaries to track per-build metrics
    build_apfd_history = {}
    build_improvement_history = {}
    
    # Calculate timesteps based on episodes and environment complexity
    # For A2C, we need to use timesteps instead of episodes
    avg_episode_length = 50  # Estimate average episode length
    total_timesteps = num_episodes * avg_episode_length
    
    # Initialize the agent with the environment
    print("Setting up A2C agent...")
    agent.set_env(env)
    
    # Training loop
    start_time = time.time()
    best_avg_improvement = -float('inf')
    
    print(f"Starting A2C training for approximately {num_episodes} episodes...")
    
    # Train in batches to allow for evaluation
    timesteps_per_batch = total_timesteps // (num_episodes // eval_frequency)
    batches = total_timesteps // timesteps_per_batch
    
    for batch in range(batches):
        print(f"Training batch {batch+1}/{batches}...")
        # Train for a batch of timesteps
        agent.learn(total_timesteps=timesteps_per_batch)
        
        # Evaluate the agent
        current_episode = batch * eval_frequency
        
        # Reset environment to evaluate
        state = env.reset()
        build_id = env.current_build
        done = False
        episode_reward = 0
        
        # Run one episode for evaluation
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Get build metrics
        build_id = info['build_id']
        episode_build_ids.append(build_id) 
        apfd = env.build_metrics[build_id]['apfd']
        improvement = env.build_metrics[build_id]['improvement']
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_apfds.append(apfd)
        episode_improvements.append(improvement)
        
        # Store build metrics
        if build_id not in build_metrics:
            build_metrics[build_id] = []
        build_metrics[build_id].append(env.build_metrics[build_id].copy())
        
        # Store per-build history metrics
        if build_id not in build_apfd_history:
            build_apfd_history[build_id] = []
            build_improvement_history[build_id] = []
            
        build_apfd_history[build_id].append(apfd)
        build_improvement_history[build_id].append(improvement)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"Batch {batch+1}/{batches}, APFD: {apfd:.4f}, "
              f"Improvement: {improvement:.4f}, Time: {elapsed:.1f}s")
        
        # Find builds with failures for evaluation
        builds_with_failures = []
        for b_id, df in env.build_data.items():
            if 'Verdict' in df.columns:
                failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                if failures > 0:
                    builds_with_failures.append(b_id)
        
        eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
        eval_results = evaluate_agent(env, agent, build_ids=eval_builds)
        avg_apfd = eval_results['avg_apfd']
        avg_improvement = eval_results['avg_improvement']
        
        print(f"\nEvaluation at batch {batch+1}")
        print(f"Avg APFD: {avg_apfd:.4f}, Avg Improvement: {avg_improvement:.4f}")
        
        # Save the model if it's the best so far
        if avg_improvement > best_avg_improvement:
            best_avg_improvement = avg_improvement
            agent.save(os.path.join(save_dir, f"best_{model_name}"))
            print(f"Saved best model with avg improvement: {best_avg_improvement:.4f}")
    
    # Save the final model
    agent.save(os.path.join(save_dir, model_name))
    print(f"Saved final model to {os.path.join(save_dir, model_name)}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Batch': range(len(episode_rewards)),
        'Build': episode_build_ids,
        'Reward': episode_rewards,
        'APFD': episode_apfds,
        'Improvement': episode_improvements
    })
    
    metrics_csv_path = os.path.join(save_dir, f'a2c_training_metrics{os.path.basename(model_name).replace(".zip", "")}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Visualize build learning
    visualize_build_learning(build_apfd_history, build_improvement_history, save_dir)
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_apfds': episode_apfds,
        'episode_improvements': episode_improvements,
        'episode_build_ids': episode_build_ids,
        'build_metrics': build_metrics,
        'build_apfd_history': build_apfd_history,
        'build_improvement_history': build_improvement_history,
        'final_env_metrics': env.get_build_metrics()
    }

def run_all_test_prioritization_approaches(num_episodes=1000, agent_type="dqn", folder_name=None):
    """
    Run and compare test prioritization approaches with either DQN or PPO agents.
    
    Args:
        num_episodes: Number of episodes to train for
        agent_type: Type of agent to use ("dqn" or "ppo")
        folder_name: Name of dataset folder to use
    
    Returns:
        Dictionary of results for each approach
    """
    
    # Initializing the dash dashbord
    init_dashboard()

    # Parameters
    if folder_name:
        data_path = f"./data/final_datasets/{folder_name}/dataset.csv"
    else:
        # Default path as fallback
        data_path = "/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv"
    
    print(f"Using dataset path: {data_path}")

    fail_value = 0
    pass_values = [1, 2]
    
    # Add episode and agent type to postfixes
    episode_postfix = f"_{folder_name}_{num_episodes}ep_{agent_type}"
    comparison_dir = f'models/comparison{episode_postfix}'
    figures_dir = f'figures/comparison{episode_postfix}'
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Convert verdict column to numeric if needed
    if 'Verdict' in df.columns and df['Verdict'].dtype == 'object':
        print(f"Converting Verdict column from {df['Verdict'].dtype} to numeric")
        df['Verdict'] = pd.to_numeric(df['Verdict'], errors='coerce')
    
    # Organize data by build
    build_data = {}
    for build_id, group in df.groupby('Build'):
        build_data[build_id] = group.reset_index(drop=True)
    
    # Apply fixes to ensure verdict types are correct
    build_data = fix_verdict_types(build_data, fail_value)
    
    # Select features for the model
    selected_features = select_and_print_features(build_data, prefixes=["REC", "TES_PRO", "TES_COM"])
    
    # Create directories for results
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Run each approach with the selected agent type
    results = {}
    
    # =====================
    # LISTWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print(f"RUNNING LISTWISE APPROACH WITH {agent_type.upper()}")
    print("="*50)

    # Create listwise environment
    listwise_env = ListwiseTestPrioritizationEnv(
        build_data=build_data,
        feature_columns=selected_features,
        fail_value=fail_value,
        pass_values=pass_values
    )
    
    # Calculate state size
    sample_state = listwise_env.reset()
    flattened_state = sample_state.reshape(-1)
    listwise_state_size = len(flattened_state)
    
    # Initialize agent based on type
    if agent_type.lower() == "dqn":
        listwise_agent = DQNAgent(
            state_size=listwise_state_size,
            action_size=listwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.0001,
            epsilon_decay=0.995
        )
        
        # Train the agent
        print(f"Training the listwise DQN agent for {num_episodes} episodes...")
        listwise_model_name = f'listwise_dqn{episode_postfix}.pt'
        listwise_metrics = train_listwise_dqn(
            env=listwise_env,
            agent=listwise_agent,
            num_episodes=num_episodes,
            update_frequency=10,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=listwise_model_name
        )
    elif agent_type.lower() == "ppo":  # PPO
        listwise_agent = PPOAgent(
            state_size=listwise_state_size,
            action_size=listwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.0007,
            gamma=0.99
        )
        
        # Train the agent
        print(f"Training the listwise PPO agent for {num_episodes} episodes...")
        listwise_model_name = f'listwise_ppo{episode_postfix}.zip'
        listwise_metrics = train_ppo(
            env=listwise_env,
            agent=listwise_agent,
            num_episodes=num_episodes,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=listwise_model_name
        )
    
    elif agent_type.lower() == "a2c":
        try:
            from a2c import A2CAgent
            listwise_agent = A2CAgent(
                state_size=listwise_state_size,
                action_size=listwise_env.max_tests,
                hidden_dim=64,  # Smaller network for faster training
                learning_rate=0.0007,
                gamma=0.99
            )
            
            # Train the agent
            print(f"Training the listwise A2C agent for {num_episodes} episodes...")
            listwise_model_name = f'listwise_a2c{episode_postfix}.zip'
            listwise_metrics = train_a2c(
                env=listwise_env,
                agent=listwise_agent,
                num_episodes=num_episodes,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=listwise_model_name
            )
        except ImportError:
            print("A2C implementation not found. Using DQN instead.")
            listwise_agent = DQNAgent(
                state_size=listwise_state_size,
                action_size=listwise_env.max_tests,
                hidden_dim=128,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995
            )
            
            # Train the agent
            print(f"Training the listwise DQN agent for {num_episodes} episodes...")
            listwise_model_name = f'listwise_dqn{episode_postfix}.pt'
            listwise_metrics = train_listwise_dqn(
                env=listwise_env,
                agent=listwise_agent,
                num_episodes=num_episodes,
                update_frequency=10,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=listwise_model_name
            )
    
    # Evaluate on builds with failures
    print(f"Evaluating the listwise {agent_type.upper()} agent on builds with failures...")
    builds_with_failures = []
    for build_id, df in listwise_env.build_data.items():
        if 'Verdict' in df.columns:
            failures = sum(1 for _, row in df.iterrows() if listwise_env.is_test_failed(row['Verdict']))
            if failures > 0:
                builds_with_failures.append(build_id)
    
    eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
    listwise_results = evaluate_agent(listwise_env, listwise_agent, build_ids=eval_builds)
    
    # Add to results
    results['listwise'] = {
        'env': listwise_env, 
        'agent': listwise_agent, 
        'results': listwise_results
    }
    
    # =====================
    # PAIRWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print(f"RUNNING PAIRWISE APPROACH WITH {agent_type.upper()}")
    print("="*50)
    
    # Create pairwise environment
    pairwise_env = PairwiseTestPrioritizationEnv(
        build_data=build_data,
        feature_columns=selected_features,
        fail_value=fail_value,
        pass_values=pass_values
    )
    
    # Calculate state size
    sample_state = pairwise_env.reset()
    flattened_state = sample_state.reshape(-1)
    pairwise_state_size = len(flattened_state)
    
    # Initialize agent based on type
    if agent_type.lower() == "dqn":
        pairwise_agent = DQNAgent(
            state_size=pairwise_state_size,
            action_size=pairwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        # Train the agent
        print(f"Training the pairwise DQN agent for {num_episodes} episodes...")
        pairwise_model_name = f'pairwise_dqn{episode_postfix}.pt'
        pairwise_metrics = train_pairwise_dqn(
            env=pairwise_env,
            agent=pairwise_agent,
            num_episodes=num_episodes,
            update_frequency=10,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=pairwise_model_name
        )
    elif agent_type.lower() == "ppo":
        pairwise_agent = PPOAgent(
            state_size=pairwise_state_size,
            action_size=pairwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.0007,
            gamma=0.99
        )
        
        # Train the agent
        print(f"Training the pairwise PPO agent for {num_episodes} episodes...")
        pairwise_model_name = f'pairwise_ppo{episode_postfix}.zip'
        pairwise_metrics = train_ppo(
            env=pairwise_env,
            agent=pairwise_agent,
            num_episodes=num_episodes,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=pairwise_model_name
        )
    elif agent_type.lower() == "a2c":
        try:
            from a2c import A2CAgent
            pairwise_agent = A2CAgent(
                state_size=pairwise_state_size,
                action_size=pairwise_env.max_tests,
                hidden_dim=64,
                learning_rate=0.0007,
                gamma=0.99
            )
            
            # Train the agent
            print(f"Training the pairwise A2C agent for {num_episodes} episodes...")
            pairwise_model_name = f'pairwise_a2c{episode_postfix}.zip'
            pairwise_metrics = train_a2c(
                env=pairwise_env,
                agent=pairwise_agent,
                num_episodes=num_episodes,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=pairwise_model_name
            )
        except ImportError:
            print("A2C implementation not found. Using DQN instead.")
            pairwise_agent = DQNAgent(
                state_size=pairwise_state_size,
                action_size=pairwise_env.max_tests,
                hidden_dim=128,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995
            )
            
            # Train the agent
            print(f"Training the pairwise DQN agent for {num_episodes} episodes...")
            pairwise_model_name = f'pairwise_dqn{episode_postfix}.pt'
            pairwise_metrics = train_pairwise_dqn(
                env=pairwise_env,
                agent=pairwise_agent,
                num_episodes=num_episodes,
                update_frequency=10,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=pairwise_model_name
            )
    
    # Evaluate on builds with failures
    print(f"Evaluating the pairwise {agent_type.upper()} agent on builds with failures...")
    builds_with_failures = []
    for build_id, df in pairwise_env.build_data.items():
        if 'Verdict' in df.columns:
            failures = sum(1 for _, row in df.iterrows() if pairwise_env.is_test_failed(row['Verdict']))
            if failures > 0:
                builds_with_failures.append(build_id)
    
    eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
    pairwise_results = evaluate_agent(pairwise_env, pairwise_agent, build_ids=eval_builds)
    
    # Add to results
    results['pairwise'] = {
        'env': pairwise_env, 
        'agent': pairwise_agent, 
        'results': pairwise_results
    }
    
    # =====================
    # POINTWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print(f"RUNNING POINTWISE APPROACH WITH {agent_type.upper()}")
    print("="*50)
    
    # Create pointwise environment
    pointwise_env = PointwiseTestPrioritizationEnv(
        build_data=build_data,
        feature_columns=selected_features,
        fail_value=fail_value,
        pass_values=pass_values
    )
    
    # Calculate state size
    sample_state = pointwise_env.reset()
    flattened_state = sample_state.reshape(-1)
    pointwise_state_size = len(flattened_state)
    
    # Initialize agent based on type
    if agent_type.lower() == "dqn":
        pointwise_agent = DQNAgent(
            state_size=pointwise_state_size,
            action_size=pointwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        
        # Train the agent
        print(f"Training the pointwise DQN agent for {num_episodes} episodes...")
        pointwise_model_name = f'pointwise_dqn{episode_postfix}.pt'
        pointwise_metrics = train_pointwise_dqn(
            env=pointwise_env,
            agent=pointwise_agent,
            num_episodes=num_episodes,
            update_frequency=10,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=pointwise_model_name
        )
    elif agent_type.lower() == "ppo":
        pointwise_agent = PPOAgent(
            state_size=pointwise_state_size,
            action_size=pointwise_env.max_tests,
            hidden_dim=128,
            learning_rate=0.0007,
            gamma=0.99
        )
        
        # Train the agent
        print(f"Training the pointwise PPO agent for {num_episodes} episodes...")
        pointwise_model_name = f'pointwise_ppo{episode_postfix}.zip'
        pointwise_metrics = train_ppo(
            env=pointwise_env,
            agent=pointwise_agent,
            num_episodes=num_episodes,
            eval_frequency=100,
            save_dir=comparison_dir,
            model_name=pointwise_model_name
        )
    elif agent_type.lower() == "a2c":
        try:
            from a2c import A2CAgent
            pointwise_agent = A2CAgent(
                state_size=pointwise_state_size,
                action_size=pointwise_env.max_tests,
                hidden_dim=64,
                learning_rate=0.0007,
                gamma=0.99
            )
            
            # Train the agent
            print(f"Training the pointwise A2C agent for {num_episodes} episodes...")
            pointwise_model_name = f'pointwise_a2c{episode_postfix}.zip'
            pointwise_metrics = train_a2c(
                env=pointwise_env,
                agent=pointwise_agent,
                num_episodes=num_episodes,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=pointwise_model_name
            )
        except ImportError:
            print("A2C implementation not found. Using DQN instead.")
            pointwise_agent = DQNAgent(
                state_size=pointwise_state_size,
                action_size=pointwise_env.max_tests,
                hidden_dim=128,
                learning_rate=0.001,
                gamma=0.99,
                epsilon=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995
            )
            
            # Train the agent
            print(f"Training the pointwise DQN agent for {num_episodes} episodes...")
            pointwise_model_name = f'pointwise_dqn{episode_postfix}.pt'
            pointwise_metrics = train_pointwise_dqn(
                env=pointwise_env,
                agent=pointwise_agent,
                num_episodes=num_episodes,
                update_frequency=10,
                eval_frequency=100,
                save_dir=comparison_dir,
                model_name=pointwise_model_name
            )
    
    # Evaluate on builds with failures
    print(f"Evaluating the pointwise {agent_type.upper()} agent on builds with failures...")
    builds_with_failures = []
    for build_id, df in pointwise_env.build_data.items():
        if 'Verdict' in df.columns:
            failures = sum(1 for _, row in df.iterrows() if pointwise_env.is_test_failed(row['Verdict']))
            if failures > 0:
                builds_with_failures.append(build_id)
    
    eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
    pointwise_results = evaluate_agent(pointwise_env, pointwise_agent, build_ids=eval_builds)
    
    # Add to results
    results['pointwise'] = {
        'env': pointwise_env, 
        'agent': pointwise_agent, 
        'results': pointwise_results
    }
    
    # =====================
    # COMPARE APPROACHES
    # =====================
    print("\n" + "="*50)
    print(f"COMPARING ALL APPROACHES WITH {agent_type.upper()}")
    print("="*50)
    
    # Print comparison table
    print(f"{'Approach':<15} {'Avg APFD':<15} {'Avg Improvement':<20}")
    print("-" * 50)
    print(f"{'Listwise':<15} {results['listwise']['results']['avg_apfd']:<15.4f} {results['listwise']['results']['avg_improvement']:<20.4f}")
    print(f"{'Pairwise':<15} {results['pairwise']['results']['avg_apfd']:<15.4f} {results['pairwise']['results']['avg_improvement']:<20.4f}")
    print(f"{'Pointwise':<15} {results['pointwise']['results']['avg_apfd']:<15.4f} {results['pointwise']['results']['avg_improvement']:<20.4f}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot APFD comparison
    plt.subplot(2, 1, 1)
    approaches = ['Listwise', 'Pairwise', 'Pointwise']
    apfds = [
        results['listwise']['results']['avg_apfd'], 
        results['pairwise']['results']['avg_apfd'], 
        results['pointwise']['results']['avg_apfd']
    ]
    
    plt.bar(approaches, apfds)
    plt.title(f'Average APFD Comparison ({agent_type.upper()}, {num_episodes} episodes)')
    plt.ylabel('APFD')
    plt.ylim(0, 1)
    
    # Plot Improvement comparison
    plt.subplot(2, 1, 2)
    improvements = [
        results['listwise']['results']['avg_improvement'], 
        results['pairwise']['results']['avg_improvement'], 
        results['pointwise']['results']['avg_improvement']
    ]
    
    plt.bar(approaches, improvements)
    plt.title(f'Average Improvement Comparison ({agent_type.upper()}, {num_episodes} episodes)')
    plt.ylabel('Improvement over Original Order')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.tight_layout()
    comparison_fig_path = os.path.join(figures_dir, f'approach_comparison{episode_postfix}.png')
    plt.savefig(comparison_fig_path)
    
    print(f"Saved comparison visualization to {comparison_fig_path}")
    
    return results

def run_reinforcement_learning(folder_name, output_path, test_count=50, agent_type="dqn", num_episodes=1000):
    """
    Main function to run reinforcement learning for test prioritization.
    
    Args:
        folder_name: Name of the dataset folder
        output_path: Path to save results
        test_count: Number of builds to test
        agent_type: Type of RL agent to use (dqn, ppo, a2c)
        num_episodes: Number of training episodes
    """
    # Convert output_path to string if it's a Path object (comes from argparse)
    if isinstance(output_path, Path):
        output_path = str(output_path)
    
    print(f"\n{'='*80}\nRunning Reinforcement Learning for Test Prioritization\n{'='*80}")
    print(f"Dataset: {folder_name}")
    print(f"Output path: {output_path}")
    print(f"Agent type: {agent_type}")
    print(f"Test count: {test_count}")
    print(f"Training episodes: {num_episodes}")
    
    # Run the comprehensive function that implements all three approaches
    results = run_all_test_prioritization_approaches(
        num_episodes=num_episodes,
        agent_type=agent_type,
        folder_name=folder_name
    )
    
    print(f"\nReinforcement learning completed for {folder_name} with all three approaches.")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test prioritization experiments')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ppo', 'a2c'],
                        help='Agent type to use (dqn or or ppo or a2c)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train for')
    parser.add_argument('--folder', type=str, required=True,
                        help='Dataset folder name (required)')
    
    args = parser.parse_args()

    # Run the experiments with the specified agent type
    results = run_all_test_prioritization_approaches(
        num_episodes=args.episodes,
        agent_type=args.agent,  # Added the missing comma here
        folder_name=args.folder
    )
    
    print(f"\nExperiment completed with {args.agent.upper()} agent on dataset from {args.folder}")