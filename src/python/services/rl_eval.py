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

# def preprocess_data_custom_verdict(data_path, fail_value=0, pass_values=[1, 2]):
#     """
#     Preprocess the data with custom verdict mapping and organize it by build.
    
#     Args:
#         data_path: Path to the CSV file.
#         fail_value: Value indicating a failed test
#         pass_values: List of values indicating passed tests
        
#     Returns:
#         Dictionary mapping build IDs to DataFrames of test data.
#     """
#     # Load data
#     df = pd.read_csv(data_path)
    
#     # Ensure required columns are present
#     required_columns = ['Build', 'Test', 'Verdict']
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Required column '{col}' not found in the dataset.")
    
#     # Verify that Verdict values match our expectations
#     unique_verdicts = df['Verdict'].unique()
#     print(f"Unique verdict values in dataset: {unique_verdicts}")
    
#     valid_verdicts = [fail_value] + pass_values
#     for verdict in unique_verdicts:
#         if verdict not in valid_verdicts:
#             print(f"WARNING: Unexpected verdict value {verdict} found in dataset!")
    
#     # Count failures
#     num_failures = (df['Verdict'] == fail_value).sum()
#     print(f"Number of failing tests (Verdict = {fail_value}): {num_failures} out of {len(df)} "
#          f"({num_failures/len(df):.2%})")
    
#     # Organize data by build
#     build_data = {}
#     for build_id, group in df.groupby('Build'):
#         build_data[build_id] = group.reset_index(drop=True)
        
#         # Check if this build has any failures
#         failures = (group['Verdict'] == fail_value).sum()
#         if failures > 0:
#             print(f"Build {build_id}: {failures}/{len(group)} tests failed ({failures/len(group):.2%})")
    
#     print(f"Loaded {len(build_data)} builds with {len(df)} test cases.")
#     return build_data

def visualize_build_learning(build_apfd_history, build_improvement_history, save_dir):
    """
    Visualize learning curves for builds that appear multiple times in the training.
    
    Args:
        build_apfd_history: Dictionary of build IDs to lists of APFD values
        build_improvement_history: Dictionary of build IDs to lists of improvement values
        save_dir: Directory to save figures
    """
    # Find builds with sufficient history
    frequent_builds = [bid for bid, history in build_apfd_history.items() 
                      if len(history) >= 5]  # At least 5 appearances
    
    if not frequent_builds:
        print("No builds appear frequently enough to track learning progress")
        return
    
    # Create plot for APFD learning curves
    plt.figure(figsize=(12, 8))
    
    for build_id in frequent_builds[:10]:  # Limit to top 10 to avoid overcrowding
        apfds = build_apfd_history[build_id]
        plt.plot(range(len(apfds)), apfds, label=f'Build {build_id}')
    
    plt.title('APFD Learning Curves for Frequent Builds')
    plt.xlabel('Episode Appearance')
    plt.ylabel('APFD')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'build_specific_learning.png'))
    
    # Create plot for improvement learning curves
    plt.figure(figsize=(12, 8))
    
    for build_id in frequent_builds[:10]:
        improvements = build_improvement_history[build_id]
        plt.plot(range(len(improvements)), improvements, label=f'Build {build_id}')
    
    plt.title('Improvement Learning Curves for Frequent Builds')
    plt.xlabel('Episode Appearance')
    plt.ylabel('Improvement over Original Order')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, 'build_specific_improvements.png'))
    
    print(f"Saved build-specific learning visualizations")


def evaluate_dqn(env, agent, build_ids=None, num_builds=10):
    """
    Evaluate the DQN agent on specific builds.
    
    Args:
        env: Environment
        agent: DQN agent
        build_ids: List of build IDs to evaluate on. If None, random builds are selected.
        num_builds: Number of builds to evaluate on if build_ids is None.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Select builds to evaluate on
    if build_ids is None:
        # Identify builds with failures (without relying on env.builds_with_failures)
        builds_with_failures = []
        for build_id, df in env.build_data.items():
            if 'Verdict' in df.columns:
                failures = sum(1 for _, row in df.iterrows() if env.is_test_failed(row['Verdict']))
                if failures > 0:
                    builds_with_failures.append(build_id)
        
        if builds_with_failures:
            build_ids = builds_with_failures[:min(num_builds, len(builds_with_failures))]
        else:
            build_ids = random.sample(list(env.build_data.keys()), min(num_builds, len(env.build_data)))
    else:
        # Filter out builds that don't exist in the dataset
        build_ids = [b for b in build_ids if b in env.build_data]
        
        # If no builds are left, select random ones
        if not build_ids:
            build_ids = random.sample(list(env.build_data.keys()), min(num_builds, len(env.build_data)))
    
    evaluation_metrics = {}
    apfds = []
    improvements = []
    
    for build_id in build_ids:
        try:
            state = env.reset(build_id=build_id)
            done = False
            
            while not done:
                # Get action from agent
                action = agent.act(state, epsilon=0.0)  # No exploration during evaluation
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                state = next_state
            
            # Check if build_metrics exists for this build
            if build_id in env.build_metrics:
                # Store metrics for this build
                build_metrics = env.build_metrics[build_id]
                evaluation_metrics[build_id] = build_metrics.copy()
                
                apfds.append(build_metrics['apfd'])
                improvements.append(build_metrics['improvement'])
            else:
                print(f"Warning: No metrics found for build {build_id}")
        except Exception as e:
            print(f"Error evaluating build {build_id}: {str(e)}")
            continue
    
    # Calculate average metrics
    if not apfds:
        print("WARNING: No valid APFD values collected during evaluation!")
        for build_id in build_ids:
            if build_id in evaluation_metrics:
                print(f"  Build {build_id} metrics: {evaluation_metrics[build_id]}")
            else:
                print(f"  Build {build_id}: No metrics available")
        
        # Default values when no valid data
        avg_apfd = 0.0
        avg_improvement = 0.0
    else:
        avg_apfd = sum(apfds) / len(apfds)
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    
    return {
        'build_metrics': evaluation_metrics,
        'avg_apfd': avg_apfd,
        'avg_improvement': avg_improvement,
        'evaluated_builds': build_ids
    }


def visualize_results(training_metrics, save_dir='figures'):
    """
    Visualize the results of training.
    
    Args:
        training_metrics: Dictionary of training metrics
        save_dir: Directory to save figures
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    episode_rewards = training_metrics['episode_rewards']
    episode_apfds = training_metrics['episode_apfds']
    episode_improvements = training_metrics['episode_improvements']
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(save_dir, 'episode_rewards.png'))
    
    # Plot episode APFDs
    plt.figure(figsize=(10, 6))
    plt.plot(episode_apfds)
    plt.title('Episode APFDs')
    plt.xlabel('Episode')
    plt.ylabel('APFD')
    plt.savefig(os.path.join(save_dir, 'episode_apfds.png'))
    
    # Plot episode improvements
    plt.figure(figsize=(10, 6))
    plt.plot(episode_improvements)
    plt.title('Episode Improvements')
    plt.xlabel('Episode')
    plt.ylabel('Improvement')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.savefig(os.path.join(save_dir, 'episode_improvements.png'))
    
    # Plot histograms of improvements
    plt.figure(figsize=(10, 6))
    plt.hist(episode_improvements, bins=20)
    plt.title('Histogram of Improvements')
    plt.xlabel('Improvement')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.savefig(os.path.join(save_dir, 'improvement_histogram.png'))
    
    # Plot final evaluation results
    final_metrics = training_metrics['final_env_metrics']
    builds = list(final_metrics.keys())
    apfds = [metrics['apfd'] for metrics in final_metrics.values() if metrics['apfd'] is not None]
    original_apfds = [metrics['original_apfd'] for metrics in final_metrics.values() if metrics['original_apfd'] is not None]
    improvements = [metrics['improvement'] for metrics in final_metrics.values() if metrics['improvement'] is not None]
    
    plt.figure(figsize=(12, 6))
    
    # Plot APFD comparison
    plt.subplot(1, 2, 1)
    x = range(len(apfds))
    plt.bar(x, original_apfds, alpha=0.5, label='Original')
    plt.bar(x, apfds, alpha=0.5, label='DQN')
    plt.xlabel('Build Index')
    plt.ylabel('APFD')
    


    plt.title('APFD Comparison')
    plt.legend()
    
    # Plot Improvements
    plt.subplot(1, 2, 2)
    plt.bar(x, improvements)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Build Index')
    plt.ylabel('Improvement')
    plt.title('APFD Improvement')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'apfd_comparison.png'))
    
    # If there are many builds, create a summary stats plot
    if len(builds) > 10:
        # Create box plots
        plt.figure(figsize=(10, 6))
        improvement_data = []
        labels = []
        
        # Group builds by improvement ranges
        ranges = [(-float('inf'), 0), (0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, float('inf'))]
        range_labels = ['Negative', '0-5%', '5-10%', '10-20%', '>20%']
        
        for (lower, upper), label in zip(ranges, range_labels):
            group_data = [imp for imp in improvements if lower <= imp < upper]
            if group_data:
                improvement_data.append(group_data)
                labels.append(f"{label} (n={len(group_data)})")
        
        plt.boxplot(improvement_data, labels=labels)
        plt.title('Improvement Distribution by Range')
        plt.ylabel('APFD Improvement')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, 'improvement_boxplot.png'))
    
    print(f"Saved result visualizations to {save_dir}")


def analyze_agent_behavior(env, agent, num_builds=5):
    """
    Analyze the agent's behavior on specific builds to understand its prioritization strategy.
    
    Args:
        env: Environment
        agent: DQN agent
        num_builds: Number of builds to analyze
        
    Returns:
        Dictionary of analysis results
    """
    # Select random builds
    build_ids = random.sample(list(env.build_data.keys()), min(num_builds, len(env.build_data)))
    
    results = {}
    for build_id in build_ids:
        # Run the agent on this build
        state = env.reset(build_id=build_id)
        done = False
        
        selected_tests = []
        selected_features = []
        
        while not done:
            # Get action from agent
            action = agent.act(state, epsilon=0.0)  # No exploration
            
            # Store selected test features
            test_idx = env.test_indices[action]
            selected_tests.append(test_idx)
            selected_features.append(env._extract_features(test_idx))
            
            # Take action
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        # Get the test data for this build
        test_data = env.build_data[build_id]
        
        # Analyze features of early vs late selected tests
        n_early = min(10, len(env.selected_tests) // 3)
        early_tests = env.selected_tests[:n_early]
        late_tests = env.selected_tests[-n_early:]
        
        early_features = np.array([env._extract_features(idx) for idx in early_tests])
        late_features = np.array([env._extract_features(idx) for idx in late_tests])
        
        # Calculate average feature values
        avg_early = np.mean(early_features, axis=0)
        avg_late = np.mean(late_features, axis=0)
        
        # Calculate feature importance (difference between early and late)
        feature_importance = avg_early - avg_late
        
        # Store results
        results[build_id] = {
            'early_tests': early_tests,
            'late_tests': late_tests,
            'feature_importance': dict(zip(env.feature_columns, feature_importance)),
            'apfd': env.build_metrics[build_id]['apfd'],
            'improvement': env.build_metrics[build_id]['improvement']
        }
        
        # Print some stats
        print(f"\nAnalysis for Build {build_id}")
        print(f"APFD: {env.build_metrics[build_id]['apfd']:.4f}")
        print(f"Improvement: {env.build_metrics[build_id]['improvement']:.4f}")
        print("Top 5 most important features for early selection:")
        
        # Sort features by importance
        sorted_features = sorted(zip(env.feature_columns, feature_importance), 
                                key=lambda x: abs(x[1]), reverse=True)
        
        for feature, importance in sorted_features[:5]:
            direction = "Higher" if importance > 0 else "Lower"
            print(f"  {feature}: {importance:.4f} ({direction} in early tests)")
    
    return results