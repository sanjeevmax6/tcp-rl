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

class TestPrioritizationEnv(gym.Env):
    """
    Environment for test case prioritization using RL.
    The agent selects one test at a time from the remaining pool of tests.
    """
    def __init__(self, build_data, feature_columns=None, fail_value=0, pass_values=[1, 2]):
        """
        Initialize the environment.
        
        Args:
            build_data: Dictionary mapping build IDs to DataFrames containing test data
            feature_columns: List of column names to use as features
        """
        self.build_data = build_data
        self.current_build = None
        self.test_indices = None
        self.selected_tests = None
        self.original_test_order = None
        
        # Store results
        self.build_metrics = {}
        
        # Find the maximum number of tests in any build
        self.max_tests = max(len(tests) for tests in build_data.values())
        
        # Define feature columns if not provided
        if feature_columns is None:
            # Use only REC, TES_PRO, and TES_COM features
            sample_build_id = list(build_data.keys())[0]
            all_columns = build_data[sample_build_id].columns.tolist()
            
            self.feature_columns = [col for col in all_columns if 
                                col.startswith("REC") or 
                                col.startswith("TES_PRO") or 
                                col.startswith("TES_COM")]
            
            # Exclude non-feature columns that might have matching prefixes
            exclude_columns = ['Build', 'Test', 'Verdict', 'Duration']
            self.feature_columns = [col for col in self.feature_columns if col not in exclude_columns]
        else:
            self.feature_columns = feature_columns
            
        # Normalize feature names if they don't exist in the dataset
        for build_id, df in self.build_data.items():
            valid_columns = [col for col in self.feature_columns if col in df.columns]
            if len(valid_columns) < len(self.feature_columns):
                print(f"Warning: Some feature columns not found in dataset. Using {len(valid_columns)} features.")
                self.feature_columns = valid_columns
                break
                
        self.feature_dim = len(self.feature_columns)

        # Add to TestPrioritizationEnv.__init__ after feature_columns are defined
        self.feature_means = {}
        self.feature_stds = {}
        self.normalize_features = True

        # Compute statistics for normalization across all builds
        if self.normalize_features:
            all_features = []
            for build_id, df in self.build_data.items():
                for col in self.feature_columns:
                    if col in df.columns:
                        values = df[col].values
                        all_features.append(values)
            
            all_features = np.concatenate(all_features)
            self.feature_means = {col: np.mean(all_features) for col in self.feature_columns}
            self.feature_stds = {col: np.std(all_features) + 1e-5 for col in self.feature_columns}  # Add small epsilon
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Features of remaining tests
            "test_features": spaces.Box(
                low=-float('inf'), 
                high=float('inf'), 
                shape=(self.max_tests, self.feature_dim)
            ),
            # Mask indicating which tests are still available
            "available_mask": spaces.Box(
                low=0, 
                high=1, 
                shape=(self.max_tests,), 
                dtype=np.int8
            ),
            # Number of tests selected so far
            "tests_selected": spaces.Discrete(self.max_tests + 1),
            # Features of already selected tests
            "selected_test_features": spaces.Box(
                low=-float('inf'), 
                high=float('inf'), 
                shape=(self.max_tests, self.feature_dim)
            ),
            # Results of already selected tests (1 for failed, 0 for passed)
            "selected_test_results": spaces.Box(
                low=0, 
                high=1, 
                shape=(self.max_tests,), 
                dtype=np.int8
            )
        })
        
        # Define action space (selecting one of the max_tests)
        self.action_space = spaces.Discrete(self.max_tests)
    
    def is_test_failed(self, verdict):
        """Check if a test has failed based on its verdict value"""
        return verdict == self.fail_value
    
    def reset(self, build_id=None, ensure_failures=False):
        """
        Reset the environment to start a new episode.
        
        Args:
            build_id: Optional build ID to use. If None, a random build is selected.
            ensure_failures: If True, only select builds with failing tests
            
        Returns:
            The initial observation.
        """
        # If we need to ensure failures, create a list of builds with failures
        if ensure_failures and build_id is None:
            builds_with_failures = []
            for bid, df in self.build_data.items():
                if 'Verdict' in df.columns and (df['Verdict'] == self.fail_value).any():
                    builds_with_failures.append(bid)
            
            if builds_with_failures:
                build_id = random.choice(builds_with_failures)
                print(f"Selected build {build_id} with failures")
            else:
                print("WARNING: No builds with failures found.")
        
        # Randomly select a build if not specified
        if build_id is None:
            build_id = random.choice(list(self.build_data.keys()))
        
        self.current_build = build_id
        self.test_data = self.build_data[build_id]
        
        # Get all test indices
        self.test_indices = list(range(len(self.test_data)))
        self.original_test_order = self.test_indices.copy()
        self.selected_tests = []
        
        # Initialize metrics for this build if not already present
        if self.current_build not in self.build_metrics:
            self.build_metrics[self.current_build] = {
                'apfd': None,
                'apfdc': None,
                'original_apfd': None,
                'original_apfdc': None,
                'improvement': None,
                'test_ordering': None,
                'num_failures': None,
                'total_tests': len(self.test_indices)
            }
        
        return self._get_observation()

    
    def step(self, action):
        # Get current state for comparison
        prev_selected_count = len(self.selected_tests)
        
        # Select the test
        selected_test = self.test_indices[action]
        self.test_indices.pop(action)
        self.selected_tests.append(selected_test)
        
        # Calculate immediate reward
        reward = 0
        
        # Check if selected test is a failure
        verdict = self.test_data.iloc[selected_test]['Verdict']
        if self.is_test_failed(verdict):
            # Reward for finding a failure is inversely proportional to position
            # Earlier failures get higher rewards
            position_factor = 1.0 - (len(self.selected_tests) / len(self.test_data))
            reward = 1.0 * position_factor
        
        # Small negative reward for each step to encourage efficiency
        reward -= 0.01
        
        # Check if all tests have been selected
        done = (len(self.test_indices) == 0)
        
        if done:
            # Calculate final metrics
            apfd, apfdc, num_failures = self._calculate_metrics()
            
            # Store metrics for this build
            self.build_metrics[self.current_build]['apfd'] = apfd
            self.build_metrics[self.current_build]['apfdc'] = apfdc
            self.build_metrics[self.current_build]['test_ordering'] = self.selected_tests.copy()
            self.build_metrics[self.current_build]['num_failures'] = num_failures
            
            # Calculate original order metrics if not already calculated
            if self.build_metrics[self.current_build].get('original_apfd') is None:
                orig_apfd, orig_apfdc, _ = self._calculate_metrics_for_order(self.original_test_order)
                self.build_metrics[self.current_build]['original_apfd'] = orig_apfd
                self.build_metrics[self.current_build]['original_apfdc'] = orig_apfdc
            
            # Calculate improvement - safely with error checking
            original_apfd = self.build_metrics[self.current_build].get('original_apfd')
            if original_apfd is not None and apfd is not None:
                improvement = apfd - original_apfd
                self.build_metrics[self.current_build]['improvement'] = improvement
                # Add terminal reward component based on improvement
                reward += max(improvement * 5.0, 0)  # Bonus for improvement, minimum 0
            else:
                self.build_metrics[self.current_build]['improvement'] = 0.0
        
        return self._get_observation(), reward, done, {'build_id': self.current_build}
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Dictionary containing the observation.
        """
        # Create arrays for test features and mask
        test_features = np.zeros((self.max_tests, self.feature_dim))
        available_mask = np.zeros(self.max_tests)
        
        # Fill in features for remaining tests
        for i, test_idx in enumerate(self.test_indices):
            if i >= self.max_tests:
                break  # Safety check
            
            # Extract features for this test
            test_features[i] = self._extract_features(test_idx)
            available_mask[i] = 1  # Mark as available
        
        # Features of already selected tests
        selected_features = np.zeros((self.max_tests, self.feature_dim))
        selected_results = np.zeros(self.max_tests)
        
        for i, test_idx in enumerate(self.selected_tests):
            if i >= self.max_tests:
                break
            
            selected_features[i] = self._extract_features(test_idx)
            # Get the test result (1 for fail, 0 for pass)
            verdict = self.test_data.iloc[test_idx]['Verdict']
            selected_results[i] = 1 if verdict == 'FAIL' else 0
    
        if len(self.test_indices) > 1:
            diff_features = np.zeros((min(self.max_tests, len(self.test_indices)), self.feature_dim))
            
            # Calculate differences from the mean of all available tests
            mean_features = np.mean(test_features[:len(self.test_indices)], axis=0)
            
            for i, test_idx in enumerate(self.test_indices):
                if i >= self.max_tests:
                    break
                
                # How different is this test from the average?
                diff_features[i] = test_features[i] - mean_features
        else:
            diff_features = np.zeros((self.max_tests, self.feature_dim))
        
        return {
            "test_features": test_features,
            "available_mask": available_mask,
            "tests_selected": len(self.selected_tests),
            "selected_test_features": selected_features,
            "selected_test_results": selected_results,
            "feature_differences": diff_features
        }
    
    def _extract_features(self, test_idx):
        """Extract normalized features for a test case."""
        # Get the test data
        test = self.test_data.iloc[test_idx]
        
        # Extract and normalize selected features
        features = []
        for col in self.feature_columns:
            if col in test and not pd.isna(test[col]):
                value = float(test[col])
                # Normalize if enabled
                if self.normalize_features:
                    value = (value - self.feature_means[col]) / self.feature_stds[col]
            else:
                value = 0.0
            features.append(value)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_metrics(self):
        """
        Calculate APFD and APFDC for the current test ordering.
        
        Returns:
            APFD score, APFDC score, and number of failures.
        """
        return self._calculate_metrics_for_order(self.selected_tests)
    
    def _calculate_metrics_for_order(self, test_order):
        """
        Calculate APFD and APFDC for a given test ordering.
        
        Args:
            test_order: List of test indices in the order they are executed.
            
        Returns:
            APFD score, APFDC score, and number of failures.
        """
        # Get failures (positions of failing tests)
        failures = []
        for i, test_idx in enumerate(test_order):
            verdict = self.test_data.iloc[test_idx]['Verdict']
            if verdict == 'FAIL':
                failures.append(i)
        
        total_tests = len(test_order)
        total_failures = len(failures)
        
        # Calculate APFD (Average Percentage of Fault Detection)
        if total_failures == 0:
            apfd = 1.0  # No failures, so perfect score
        else:
            apfd = 1 - (sum(failures) / (total_tests * total_failures)) + (1 / (2 * total_tests))
        
        # Calculate APFDC (Average Percentage of Fault Detection with Cost)
        if total_failures == 0:
            apfdc = 1.0
        else:
            # Get test costs (using execution time as cost)
            costs = []
            for test_idx in test_order:
                cost = self.test_data.iloc[test_idx].get('REC_RecentAvgExeTime', None)
                if cost is None or pd.isna(cost):
                    cost = 1.0  # Default cost if not available
                costs.append(cost)
                
            total_cost = sum(costs)
            
            # Calculate APFDC using costs
            fault_detection_cost = 0
            for i in failures:
                fault_detection_cost += sum(costs[:i+1])
                
            if total_cost == 0:
                apfdc = 1.0
            else:
                apfdc = 1 - (fault_detection_cost / (total_cost * total_failures)) + (costs[0] / (2 * total_cost))
        
        return apfd, apfdc, total_failures
    
    def get_build_metrics(self):
        """
        Get the metrics for all builds.
        
        Returns:
            Dictionary mapping build IDs to metrics.
        """
        return self.build_metrics

    def print_failure_stats(self):
        """Print statistics about failing tests in the dataset"""
        build_failure_counts = {}
        total_tests = 0
        total_failures = 0
        
        for build_id, df in self.build_data.items():
            if 'Verdict' in df.columns:
                failures = (df['Verdict'] == self.fail_value).sum()
                build_failure_counts[build_id] = {
                    'total_tests': len(df),
                    'failures': failures,
                    'failure_rate': failures / len(df) if len(df) > 0 else 0
                }
                total_tests += len(df)
                total_failures += failures


class FixedTestPrioritizationEnv(TestPrioritizationEnv):
    """A fixed version of the environment with robust failure detection but without forced failure inclusion"""
    
    def __init__(self, build_data, feature_columns=None, fail_value=0, pass_values=[1, 2]):
        self.fail_value = fail_value
        self.pass_values = pass_values

        super().__init__(build_data, feature_columns, fail_value, pass_values)
        
        # Log total number of builds in the dataset
        print(f"Environment initialized with {len(self.build_data)} builds")
        
        # Optional: Count failures across all builds for monitoring purposes only
        total_failures = 0
        total_tests = 0
        for build_id, df in self.build_data.items():
            if 'Verdict' in df.columns:
                failures = sum(1 for _, row in df.iterrows() if self.is_test_failed(row['Verdict']))
                total_failures += failures
                total_tests += len(df)
                
        if total_tests > 0:
            print(f"Dataset contains {total_failures}/{total_tests} failures ({total_failures/total_tests:.2%})")

    def is_test_failed(self, verdict):
        """Robust check if a test has failed based on its verdict value"""
        # Try multiple comparison approaches to catch potential type issues
        if isinstance(verdict, (int, float, np.integer, np.floating)):
            # Numeric comparison
            return float(verdict) == float(self.fail_value)
        elif isinstance(verdict, str):
            # String comparison
            return verdict == str(self.fail_value)
        else:
            # Last resort - convert both to strings
            return str(verdict) == str(self.fail_value)
    
    def reset(self, build_id=None):
        """Reset the environment to start a new episode with any random build"""
        # Pass the build_id parameter to the parent class reset
        return super().reset(build_id=build_id)
    
    def _calculate_metrics_for_order(self, test_order):
        """Calculate APFD and APFDC with debugging output"""
        # Get failures (positions of failing tests)
        failures = []
        
        # Debug info
        detected_failures = 0
        
        for i, test_idx in enumerate(test_order):
            verdict = self.test_data.iloc[test_idx]['Verdict']
            if self.is_test_failed(verdict):
                failures.append(i)
                detected_failures += 1
        
        # Only print detailed debug info if verbose logging is enabled
        # You could add a verbose flag to the class if needed
        if detected_failures > 0:
            print(f"Build {self.current_build}: {detected_failures} failures in {len(test_order)} tests")
            print(f"  Failure positions: {failures}")
        
        # Continue with regular calculation
        total_tests = len(test_order)
        total_failures = len(failures)
        
        # Calculate APFD (Average Percentage of Fault Detection)
        if total_failures == 0:
            apfd = 1.0  # No failures, so perfect score
        elif total_failures == total_tests:
            apfd = 0.0  # All tests fail, worst case scenario
        else:
            apfd = 1 - (sum(failures) / (total_tests * total_failures)) + (1 / (2 * total_tests))
        
        # Calculate APFDC (Average Percentage of Fault Detection with Cost)
        if total_failures == 0:
            apfdc = 1.0
        else:
            # Get test costs (using execution time as cost)
            costs = []
            for test_idx in test_order:
                cost = self.test_data.iloc[test_idx].get('REC_RecentAvgExeTime', None)
                if cost is None or pd.isna(cost):
                    cost = 1.0  # Default cost if not available
                costs.append(cost)
                
            total_cost = sum(costs)
            
            # Calculate APFDC using costs
            fault_detection_cost = 0
            for i in failures:
                fault_detection_cost += sum(costs[:i+1])
                
            if total_cost == 0:
                apfdc = 1.0
            else:
                apfdc = 1 - (fault_detection_cost / (total_cost * total_failures)) + (costs[0] / (2 * total_cost))
        
        return apfd, apfdc, total_failures


class ListwiseTestPrioritizationEnv(gym.Env):
    """Environment for listwise test case prioritization using RL."""
    def __init__(self, build_data, feature_columns=None, fail_value=0, pass_values=[1, 2]):
        """Initialize the environment."""
        self.build_data = build_data
        self.current_build = None
        self.fail_value = fail_value
        self.pass_values = pass_values
        
        # Find the maximum number of tests in any build
        self.max_tests = max(len(tests) for tests in build_data.values())
        
        # Define feature columns
        if feature_columns is None:
            sample_build_id = list(build_data.keys())[0]
            all_columns = build_data[sample_build_id].columns.tolist()
            self.feature_columns = [col for col in all_columns if 
                                col.startswith("REC") or 
                                col.startswith("TES_PRO") or 
                                col.startswith("TES_COM")]
            exclude_columns = ['Build', 'Test', 'Verdict', 'Duration']
            self.feature_columns = [col for col in self.feature_columns if col not in exclude_columns]
        else:
            self.feature_columns = feature_columns
            
        self.feature_dim = len(self.feature_columns)

        # Compute statistics for normalization
        self.feature_means = {}
        self.feature_stds = {}
        self.normalize_features = True
        if self.normalize_features:
            all_features = []
            for build_id, df in self.build_data.items():
                for col in self.feature_columns:
                    if col in df.columns:
                        values = df[col].values
                        all_features.append(values)
            
            all_features = np.concatenate(all_features)
            self.feature_means = {col: np.mean(all_features) for col in self.feature_columns}
            self.feature_stds = {col: np.std(all_features) + 1e-5 for col in self.feature_columns}
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.max_tests)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(self.max_tests, self.feature_dim)
        )
        
        # Initialize tracking variables
        self.test_indices = None
        self.selected_tests = None
        self.original_test_order = None
        self.build_metrics = {}
        self.optimal_order = None
        self.padding_value = -1
        
    def is_test_failed(self, verdict):
        """Check if a test has failed based on its verdict value"""
        if isinstance(verdict, (int, float, np.integer, np.floating)):
            return float(verdict) == float(self.fail_value)
        elif isinstance(verdict, str):
            return verdict == str(self.fail_value)
        else:
            return str(verdict) == str(self.fail_value)
    
    def get_optimal_order(self):
        """Calculate the optimal test case ordering (failing tests first)"""
        # Sort test cases by verdict (failures first)
        test_cases = self.test_data.copy()
        test_cases['IsFailure'] = test_cases['Verdict'].apply(self.is_test_failed)
        
        # Sort by IsFailure (True first, which sorts failures to the top)
        sorted_tests = test_cases.sort_values(by='IsFailure', ascending=False)
        return list(sorted_tests.index)
    
    def reset(self, build_id=None):
        """Reset the environment for a new episode"""
        # Randomly select a build if not specified
        if build_id is None:
            build_id = random.choice(list(self.build_data.keys()))
        
        self.current_build = build_id
        self.test_data = self.build_data[build_id]
        
        # Get all test indices
        self.test_indices = list(range(len(self.test_data)))
        self.original_test_order = self.test_indices.copy()
        self.selected_tests = []
        
        # Calculate optimal order for this build
        self.optimal_order = self.get_optimal_order()
        
        # Initialize metrics for this build
        if self.current_build not in self.build_metrics:
            self.build_metrics[self.current_build] = {
                'apfd': None, 'apfdc': None, 'original_apfd': None,
                'original_apfdc': None, 'improvement': None,
                'test_ordering': None, 'num_failures': None,
                'total_tests': len(self.test_indices)
            }
        
        # Export test case features as observation
        self.current_obs = self._export_test_cases()
        self.initial_observation = np.copy(self.current_obs)
        
        return self.current_obs
    
    def _export_test_cases(self):
        """Export test cases with their features"""
        test_features = np.zeros((self.max_tests, self.feature_dim))
        
        # Fill in features for all tests
        for i, test_idx in enumerate(self.test_indices):
            if i >= self.max_tests:
                break
            test_features[i] = self._extract_features(test_idx)
        
        return test_features
    
    def _extract_features(self, test_idx):
        """Extract normalized features for a test case"""
        test = self.test_data.iloc[test_idx]
        features = []
        
        for col in self.feature_columns:
            if col in test and not pd.isna(test[col]):
                value = float(test[col])
                if self.normalize_features:
                    value = (value - self.feature_means[col]) / self.feature_stds[col]
            else:
                value = 0.0
            features.append(value)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, test_case_index):
        """Calculate reward based on optimal test ordering"""
        if test_case_index >= len(self.test_data) or self.selected_tests.count(test_case_index) > 0:
            return 0  # Invalid action
        
        # Current position in our ordering
        assigned_rank = len(self.selected_tests)
        
        # Find optimal position for this test case
        optimal_rank = self.optimal_order.index(test_case_index)
        
        # Normalize ranks to [0,1]
        total_tests = len(self.test_data)
        normalized_optimal_rank = optimal_rank / total_tests
        normalized_assigned_rank = assigned_rank / total_tests
        
        # Reward is better when assigned rank is closer to optimal rank
        reward = 1 - (normalized_assigned_rank - normalized_optimal_rank)**2
        
        # Additional reward for selecting failing tests early
        verdict = self.test_data.iloc[test_case_index]['Verdict']
        if self.is_test_failed(verdict):
            # Position factor - higher reward for earlier positions
            position_factor = 1.0 - (assigned_rank / total_tests)
            reward += 2.0 * position_factor
        
        return reward
    
    def step(self, action):
        """Take an action in the environment"""
        # Get current state for comparison
        reward = self._calculate_reward(action)
        
        # Select the test if valid action
        if action < len(self.test_indices):
            selected_test = self.test_indices[action]
            self.test_indices.pop(action)
            self.selected_tests.append(selected_test)
        
        # Check if all tests have been selected
        done = (len(self.test_indices) == 0)
        
        if done:
            # Calculate final metrics
            apfd, apfdc, num_failures = self._calculate_metrics()
            
            # Store metrics for this build
            self.build_metrics[self.current_build]['apfd'] = apfd
            self.build_metrics[self.current_build]['apfdc'] = apfdc
            self.build_metrics[self.current_build]['test_ordering'] = self.selected_tests.copy()
            self.build_metrics[self.current_build]['num_failures'] = num_failures
            
            # Calculate original order metrics if not already calculated
            if self.build_metrics[self.current_build].get('original_apfd') is None:
                orig_apfd, orig_apfdc, _ = self._calculate_metrics_for_order(self.original_test_order)
                self.build_metrics[self.current_build]['original_apfd'] = orig_apfd
                self.build_metrics[self.current_build]['original_apfdc'] = orig_apfdc
            
            # Calculate improvement
            original_apfd = self.build_metrics[self.current_build].get('original_apfd')
            if original_apfd is not None and apfd is not None:
                improvement = apfd - original_apfd
                self.build_metrics[self.current_build]['improvement'] = improvement
        
        # Update observation
        self.current_obs = self._next_observation(action)
        
        return self.current_obs, reward, done, {'build_id': self.current_build}
    
    def _next_observation(self, action):
        """Update observation after taking an action"""
        if action < len(self.current_obs):
            # Mark the selected test as unavailable by setting its features to padding value
            self.current_obs[action] = np.repeat(self.padding_value, self.feature_dim)
        return self.current_obs
    
    def _calculate_metrics(self):
        """Calculate APFD and APFDC for the current test ordering"""
        return self._calculate_metrics_for_order(self.selected_tests)
    
    def _calculate_metrics_for_order(self, test_order):
        """Calculate APFD and APFDC for a given test ordering"""
        failures = []
        for i, test_idx in enumerate(test_order):
            verdict = self.test_data.iloc[test_idx]['Verdict']
            if self.is_test_failed(verdict):
                failures.append(i)
        
        total_tests = len(test_order)
        total_failures = len(failures)
        
        # Calculate APFD
        if total_failures == 0:
            apfd = 1.0  # No failures, so perfect score
        else:
            apfd = 1 - (sum(failures) / (total_tests * total_failures)) + (1 / (2 * total_tests))
        
        # Calculate APFDC
        if total_failures == 0:
            apfdc = 1.0
        else:
            # Get test costs
            costs = []
            for test_idx in test_order:
                cost = self.test_data.iloc[test_idx].get('REC_RecentAvgExeTime', 1.0)
                if pd.isna(cost):
                    cost = 1.0
                costs.append(cost)
                
            total_cost = sum(costs)
            
            # Calculate APFDC using costs
            fault_detection_cost = 0
            for i in failures:
                fault_detection_cost += sum(costs[:i+1])
                
            if total_cost == 0:
                apfdc = 1.0
            else:
                apfdc = 1 - (fault_detection_cost / (total_cost * total_failures)) + (costs[0] / (2 * total_cost))
        
        return apfd, apfdc, total_failures
    
    def get_build_metrics(self):
        """Get metrics for all builds"""
        return self.build_metrics