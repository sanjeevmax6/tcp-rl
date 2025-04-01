import pandas as pd
import numpy as np
import os
import random
import torch
import time
import matplotlib.pyplot as plt
from dqn import DQNNetwork, DQNAgent
from env import TestPrioritizationEnv, FixedTestPrioritizationEnv, ListwiseTestPrioritizationEnv, PairwiseTestPrioritizationEnv, PointwiseTestPrioritizationEnv
from rl_eval import visualize_build_learning, evaluate_dqn, visualize_results, analyze_agent_behavior

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
            eval_results = evaluate_dqn(env, agent, list(env.build_data.keys())[:10])
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

# Main function to run the fixed version
def run_fixed_test_prioritization():
    # Parameters
    data_path = "/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv"  # Replace with your actual data path
    fail_value = 0  # 0 indicates failure in your dataset
    pass_values = [1, 2]  # 1 and 2 indicate passing tests
    num_episodes = 1000
    
    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)
    
    # Add explicit verdict type conversion to ensure numeric comparison works
    if 'Verdict' in df.columns and df['Verdict'].dtype == 'object':
        print(f"Converting Verdict column from {df['Verdict'].dtype} to numeric")
        df['Verdict'] = pd.to_numeric(df['Verdict'], errors='coerce')
    
    # Organize data by build
    build_data = {}
    for build_id, group in df.groupby('Build'):
        build_data[build_id] = group.reset_index(drop=True)
    
    # Step 2: Apply fixes to ensure verdict types are correct
    build_data = fix_verdict_types(build_data, fail_value)
    
    # Step 3: Create the fixed environment
    selected_features = select_and_print_features(build_data, prefixes=["REC", "TES_PRO", "TES_COM"])

    # Step 4: Create the fixed environment with selected features
    env = FixedTestPrioritizationEnv(
        build_data=build_data,
        feature_columns=selected_features,
        fail_value=fail_value,
        pass_values=pass_values
    )
    
    # Step 4: Initialize the DQN agent
    # Use a build with failures for sample state
    sample_state = env.reset()
    state_tensor = np.concatenate([
        sample_state["test_features"].reshape(-1),
        sample_state["available_mask"]
    ])
    state_size = state_tensor.shape[0]
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=env.max_tests,
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Create directories for results
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Step 5: Train the agent using ONLY builds with failures
    print(f"Training the agent for {num_episodes} episodes on builds with failures...")
    training_metrics = train_dqn(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        update_frequency=10,
        eval_frequency=50,
        save_dir='models'
    )
    
    # Step 6: Visualize the results
    print("Visualizing results...")
    visualize_results(training_metrics, save_dir='figures')
    
    # Step 7: Evaluate on builds with failures
    print("Evaluating the agent on builds with failures...")
    # Take a subset for evaluation (max 10 builds)
    builds_with_failures = []
    for build_id, df in env.build_data.items():
        if 'Verdict' in df.columns:
            failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
            if failures > 0:
                builds_with_failures.append(build_id)

    # Use random sampling to avoid problematic builds
    if builds_with_failures:
        eval_builds = random.sample(builds_with_failures, min(10, len(builds_with_failures)))
        results = evaluate_dqn(env, agent, build_ids=eval_builds)
        
        print("\nEvaluation Results on Builds with Failures:")
        print(f"Average APFD: {results['avg_apfd']:.4f}")
        print(f"Average Improvement: {results['avg_improvement']:.4f}")
    else:
        print("No builds with failures found for evaluation.")
    
    # Save the agent
    agent.save('models/dqn_1000.pt')
    print("Saved final model to models/fixed_dqn_agent.pt")
    
    return env, agent, results

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
                eval_results = evaluate_dqn(env, agent, build_ids=eval_builds)
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
            eval_results = evaluate_dqn(env, agent, build_ids=eval_builds)
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
            eval_results = evaluate_dqn(env, agent, build_ids=eval_builds)
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

def run_all_test_prioritization_approaches(num_episodes=1000):
    """Run and compare all test prioritization approaches"""
    # Parameters
    data_path = "/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv"
    fail_value = 0
    pass_values = [1, 2]
    
    # Add episode postfix to all directories and filenames
    episode_postfix = f"_{num_episodes}ep"
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
    
    # =====================
    # LISTWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print("RUNNING LISTWISE APPROACH")
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
    
    # Initialize DQN agent
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
    print(f"Training the listwise agent for {num_episodes} episodes...")
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
    print("Evaluating the listwise agent on builds with failures...")
    builds_with_failures = []
    for build_id, df in listwise_env.build_data.items():
        if 'Verdict' in df.columns:
            failures = sum(1 for _, row in df.iterrows() if listwise_env.is_test_failed(row['Verdict']))
            if failures > 0:
                builds_with_failures.append(build_id)
    
    eval_builds = builds_with_failures[:min(10, len(builds_with_failures))]
    listwise_results = evaluate_dqn(listwise_env, listwise_agent, build_ids=eval_builds)
    
    # =====================
    # PAIRWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print("RUNNING PAIRWISE APPROACH")
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
    pairwise_state_size = len(sample_state)
    
    # Initialize DQN agent
    pairwise_agent = DQNAgent(
        state_size=pairwise_state_size,
        action_size=2,  # Binary choice for pairwise
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train the agent
    print(f"Training the pairwise agent for {num_episodes} episodes...")
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
    print("Evaluating the pairwise agent on builds with failures...")
    pairwise_results = evaluate_dqn(pairwise_env, pairwise_agent, build_ids=eval_builds)
    
    # =====================
    # POINTWISE APPROACH
    # =====================
    print("\n" + "="*50)
    print("RUNNING POINTWISE APPROACH")
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
    pointwise_state_size = len(sample_state)
    
    # Initialize DQN agent for continuous actions
    pointwise_agent = DQNAgent(
        state_size=pointwise_state_size,
        action_size=1,  # Continuous priority score
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train the agent
    print(f"Training the pointwise agent for {num_episodes} episodes...")
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
    print("Evaluating the pointwise agent on builds with failures...")
    pointwise_results = evaluate_dqn(pointwise_env, pointwise_agent, build_ids=eval_builds)
    
    # =====================
    # COMPARE APPROACHES
    # =====================
    print("\n" + "="*50)
    print("COMPARING ALL APPROACHES")
    print("="*50)
    
    # Print comparison table
    print(f"{'Approach':<15} {'Avg APFD':<15} {'Avg Improvement':<20}")
    print("-" * 50)
    print(f"{'Listwise':<15} {listwise_results['avg_apfd']:<15.4f} {listwise_results['avg_improvement']:<20.4f}")
    print(f"{'Pairwise':<15} {pairwise_results['avg_apfd']:<15.4f} {pairwise_results['avg_improvement']:<20.4f}")
    print(f"{'Pointwise':<15} {pointwise_results['avg_apfd']:<15.4f} {pointwise_results['avg_improvement']:<20.4f}")
    
    # Create comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot APFD comparison
    plt.subplot(2, 1, 1)
    approaches = ['Listwise', 'Pairwise', 'Pointwise']
    apfds = [listwise_results['avg_apfd'], pairwise_results['avg_apfd'], pointwise_results['avg_apfd']]
    
    plt.bar(approaches, apfds)
    plt.title(f'Average APFD Comparison ({num_episodes} episodes)')
    plt.ylabel('APFD')
    plt.ylim(0, 1)
    
    # Plot Improvement comparison
    plt.subplot(2, 1, 2)
    improvements = [listwise_results['avg_improvement'], pairwise_results['avg_improvement'], 
                   pointwise_results['avg_improvement']]
    
    plt.bar(approaches, improvements)
    plt.title(f'Average Improvement Comparison ({num_episodes} episodes)')
    plt.ylabel('Improvement over Original Order')
    plt.axhline(y=0, color='r', linestyle='-')
    
    plt.tight_layout()
    comparison_fig_path = os.path.join(figures_dir, f'approach_comparison{episode_postfix}.png')
    plt.savefig(comparison_fig_path)
    
    print(f"Saved comparison visualization to {comparison_fig_path}")
    
    return {
        'listwise': {'env': listwise_env, 'agent': listwise_agent, 'results': listwise_results},
        'pairwise': {'env': pairwise_env, 'agent': pairwise_agent, 'results': pairwise_results},
        'pointwise': {'env': pointwise_env, 'agent': pointwise_agent, 'results': pointwise_results}
    }

if __name__ == "__main__":
    # Specify the number of episodes as an argument
    num_episodes = 5000
    comparison_results = run_all_test_prioritization_approaches(num_episodes)