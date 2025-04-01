import os
import numpy as np
import pandas as pd
from ..decay_dataset_factory import DecayDatasetFactory
from ..dataset_factory import DatasetFactory
from ..feature_extractor.feature import Feature
from ..module_factory import ModuleFactory
import sys
from ..ranklib_learner import RankLibLearner
from ..code_analyzer.code_analyzer import AnalysisLevel
from ..results.results_analyzer import ResultAnalyzer
from ..hyp_param_opt import HypParamOpt
from .data_service import DataService
from pathlib import Path
from enum import Enum
import logging
from .rl_data_processing import load_and_preprocess_data
from .rl_env import TCPEnv
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
import torch
# from .rl_visualization import evaluate_model_per_build, calculate_apfd, calculate_apfdc, save_results_to_csv, plot_metrics_over_builds
# from .rl_trainer import DictFeatureExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter

class Experiment(Enum):
    FULL = "FULL"
    WO_IMP = "WO_IMP"
    WO_TES_COM = "WO_TES_COM"
    WO_TES_PRO = "WO_TES_PRO"
    WO_TES_CHN = "WO_TES_CHN"
    WO_REC = "WO_REC"
    WO_COV = "WO_COV"
    WO_COD_COV_COM = "WO_COD_COV_COM"
    WO_COD_COV_PRO = "WO_COD_COV_PRO"
    WO_COD_COV_CHN = "WO_COD_COV_CHN"
    WO_DET_COV = "WO_DET_COV"
    W_Code = "W_Code"
    W_Execution = "W_Execution"
    W_Coverage = "W_Coverage"


class ExperimentsService:
    @staticmethod
    def run_best_ranker_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info("Reading the dataset.")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        results_path = args.output_path / "tsp_accuracy_results"
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        logging.info("Finished reading the dataset.")

        logging.info(
            f"***** Running {args.experiment.value} experiment for {dataset_path.parent.name} *****"
        )
        if args.experiment == Experiment.FULL:
            learner.run_accuracy_experiments(
                outliers_dataset_df, "full-outliers", results_path
            )
            learner.test_heuristics(outliers_dataset_df, results_path / "full-outliers")
        elif args.experiment == Experiment.WO_IMP:
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
                "wo-impacted-outliers",
                results_path,
            )
        elif (
            args.experiment.value.startswith("WO_")
            and args.experiment != Experiment.WO_IMP
        ):
            feature_groups_names = {
                "TES_COM": Feature.TES_COM,
                "TES_PRO": Feature.TES_PRO,
                "TES_CHN": Feature.TES_CHN,
                "REC": Feature.REC,
                "COV": Feature.COV,
                "COD_COV_COM": Feature.COD_COV_COM,
                "COD_COV_PRO": Feature.COD_COV_PRO,
                "COD_COV_CHN": Feature.COD_COV_CHN,
                "DET_COV": Feature.DET_COV,
            }
            feature_group = args.experiment.value[3:]
            names = feature_groups_names[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(names, axis=1),
                f"wo-{feature_group}-outliers",
                results_path,
            )
        elif args.experiment.value.startswith("W_"):
            test_code_features = Feature.TES_COM + Feature.TES_PRO + Feature.TES_CHN
            test_execution_features = Feature.REC
            test_coverage_features = (
                Feature.COV
                + Feature.COD_COV_COM
                + Feature.COD_COV_PRO
                + Feature.COD_COV_CHN
                + Feature.DET_COV
            )
            high_level_feature_groups = {
                "Code": test_code_features,
                "Execution": test_execution_features,
                "Coverage": test_coverage_features,
            }
            non_feature_cols = [
                Feature.BUILD,
                Feature.TEST,
                Feature.VERDICT,
                Feature.DURATION,
            ]
            feature_group = args.experiment.value[2:]
            names = high_level_feature_groups[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df[non_feature_cols + names],
                f"W-{feature_group}-outliers",
                results_path,
            )
        logging.info("Done run_best_ranker_experiments")

    @staticmethod
    def run_all_tcp_rankers(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        rankers = {
            0: ("MART", {"tree": 30}),
            6: (
                "LambdaMART",
                {"tree": 30, "metric2T": "NDCG@10", "metric2t": "NDCG@10"},
            ),
            2: ("RankBoost", {}),
            4: ("CoordinateAscent", {}),
            7: ("ListNet", {}),
            8: ("RandomForest", {}),
        }
        results_path = args.output_path / "tcp_rankers"
        for id, info in rankers.items():
            name, params = info
            logging.info(
                f"***** Running {name} full feature set without Outliers experiments *****"
            )
            learner.run_accuracy_experiments(
                outliers_dataset_df, name, results_path, ranker=(id, params)
            )

    @staticmethod
    def run_decay_test_experiments(args):
        logging.info(f"Running decay tests for {args.output_path.name}")
        repo_miner_class = ModuleFactory.get_repository_miner(AnalysisLevel.FILE)
        repo_miner = repo_miner_class(args)
        change_history_df = repo_miner.load_entity_change_history()
        dataset_factory = DatasetFactory(
            args,
            change_history_df,
            repo_miner,
        )
        dataset_df = pd.read_csv(args.output_path / "dataset.csv")
        decay_ds_factory = DecayDatasetFactory(dataset_factory, args)
        models_path = args.output_path / "tsp_accuracy_results" / "full-outliers"
        decay_ds_factory.create_decay_datasets(dataset_df, models_path)

        learner = RankLibLearner(args)
        datasets_path = args.output_path / "decay_datasets"
        learner.run_decay_test_experiments(datasets_path, models_path)
        logging.info(f"All finished and results are saved at {datasets_path}")
        print()

    @staticmethod
    def analyze_results(args):
        result_analyzer = ResultAnalyzer(args)
        result_analyzer.analyze_results()

    @staticmethod
    def hyp_param_opt(args):
        optimizer = HypParamOpt(args)
        logging.info(f"***** Running {args.output_path.name} hypopt *****")
        build_ds_path = Path(args.output_path / "hyp_param_opt" / str(args.build))
        optimizer.run_optimization(build_ds_path, args.comb_index)
    
    def run_rl_learning():

        # # Function to load and preprocess data
        # def load_and_preprocess_data(data_path):
        #     """Load and preprocess the dataset for the environment"""
        #     df = pd.read_csv(data_path)
            
        #     # Extract feature columns (excluding Build, Test, Verdict, and Duration)
        #     feature_cols = [col for col in df.columns if col not in ['Build', 'Test', 'Verdict', 'Duration']]
            
        #     return df, feature_cols

        # # Main training function
        # def train_dqn_model(device, data_path, log_dir="./logs", total_timesteps=100000):
        #     """Train a DQN model on the test prioritization task"""
        #     # Load and preprocess data
        #     data, feature_cols = load_and_preprocess_data(data_path)
            
        #     # Split data into train and evaluation sets
        #     builds = sorted(data['Build'].unique())
        #     train_size = int(0.8 * len(builds))
        #     train_builds = builds[:train_size]
        #     eval_builds = builds[train_size:]
            
        #     print(f"Training on {len(train_builds)} builds, evaluating on {len(eval_builds)} builds")
            
        #     # Create environments
        #     def make_train_env():
        #         env = TestCasePrioritizationEnv(
        #             dataset=data[data['Build'].isin(train_builds)],
        #             feature_columns=feature_cols
        #         )
        #         return Monitor(env)
            
        #     def make_eval_env():
        #         env = TestCasePrioritizationEnv(
        #             dataset=data[data['Build'].isin(eval_builds)],
        #             feature_columns=feature_cols
        #         )
        #         return Monitor(env)
            
        #     # Create vectorized environments
        #     train_env = DummyVecEnv([make_train_env])
        #     train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
            
        #     eval_env = DummyVecEnv([make_eval_env])
        #     # Use same normalization stats as training environment
        #     eval_env = VecNormalize(
        #         eval_env,
        #         norm_obs=True,
        #         norm_reward=True,
        #         training=False  # Don't update normalization stats during evaluation
        #     )
            
        #     # Define model with custom policy
        #     policy_kwargs = {
        #         "features_extractor_class": DictFeatureExtractor,
        #         "features_extractor_kwargs": {"features_dim": 128},
        #         "net_arch": [128, 128]  # Additional layers after feature extraction
        #     }
            
        #     # Create DQN model
        #     model = DQN(
        #         "MultiInputPolicy",
        #         train_env,
        #         policy_kwargs=policy_kwargs,
        #         learning_rate=1e-4,
        #         buffer_size=10000,
        #         learning_starts=1000,
        #         batch_size=64,
        #         gamma=0.99,
        #         train_freq=4,
        #         gradient_steps=1,
        #         target_update_interval=100,
        #         exploration_fraction=0.2,
        #         exploration_final_eps=0.05,
        #         tensorboard_log=log_dir,
        #         verbose=1,
        #         device = device,
        #     )
            
        #     # Create callbacks
        #     callback = SaveBestAPFDCCallback(eval_env, check_freq=5000, log_dir=log_dir)
            
        #     # Train the model
        #     print("Starting training...")
        #     model.learn(
        #         total_timesteps=total_timesteps,
        #         callback=callback,
        #         tb_log_name="dqn_tcp"
        #     )
            
        #     # Save final model
        #     model.save(os.path.join(log_dir, "final_model"))
        #     train_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
            
        #     print("Training complete!")
        #     return model, train_env, eval_env
        
        # # Evaluation function
        # def evaluate_model(model, env, num_episodes=50):
        #     """Evaluate a trained model on the environment"""
        #     apfdcs = []
        #     rewards = []
            
        #     for _ in range(num_episodes):
        #         obs, _ = env.reset()
        #         episode_reward = 0
        #         done = False
                
        #         while not done:
        #             action, _ = model.predict(obs, deterministic=True)
        #             obs, reward, terminated, truncated, info = env.step(action)
        #             episode_reward += reward
        #             done = terminated or truncated
                    
        #             if done and "apfdc" in info:
        #                 apfdcs.append(info["apfdc"])
                
        #         rewards.append(episode_reward)
            
        #     mean_apfdc = np.mean(apfdcs)
        #     mean_reward = np.mean(rewards)
            
        #     print(f"Evaluation results over {num_episodes} episodes:")
        #     print(f"Mean APFDC: {mean_apfdc:.4f}")
        #     print(f"Mean reward: {mean_reward:.4f}")
            
        #     return mean_apfdc, mean_reward

        # # Callback for saving best models and logging
        # class SaveBestAPFDCCallback(BaseCallback):
        #     """
        #     Callback for saving best models based on APFDC metric.
        #     """
            
        #     def __init__(self, eval_env, check_freq=1000, log_dir="./logs", verbose=1):
        #         super(SaveBestAPFDCCallback, self).__init__(verbose)
        #         self.check_freq = check_freq
        #         self.log_dir = log_dir
        #         self.eval_env = eval_env
        #         self.best_apfdc = -np.inf
        #         self.save_path = os.path.join(log_dir, "best_model")
                
        #         # Create log directory if it doesn't exist
        #         os.makedirs(log_dir, exist_ok=True)
                
        #         # Create Tensorboard writer
        #         self.writer = SummaryWriter(log_dir=log_dir)
            
        #     def _on_step(self):
        #         if self.n_calls % self.check_freq == 0:
        #             # Run evaluation episodes
        #             apfdcs = []
        #             for _ in range(10):  # Evaluate on 10 episodes
        #                 reset_result = self.eval_env.reset()
        #                 # Handle different return formats
        #                 if isinstance(reset_result, tuple):
        #                     obs = reset_result[0]  # Get first element (observation)
        #                 else:
        #                     obs = reset_result  # Just use the result directly
        #                 done = False
        #                 while not done:
        #                     action, _ = self.model.predict(obs, deterministic=True)
        #                     obs, _, terminated, truncated, info = self.eval_env.step(action)
        #                     done = terminated or truncated
        #                     if done and "apfdc" in info:
        #                         apfdcs.append(info["apfdc"])
                    
        #             # Calculate average APFDC
        #             mean_apfdc = np.mean(apfdcs) if apfdcs else 0.0
                    
        #             # Log to Tensorboard
        #             self.writer.add_scalar("eval/mean_apfdc", mean_apfdc, self.n_calls)
                    
        #             # Save best model
        #             if mean_apfdc > self.best_apfdc:
        #                 self.best_apfdc = mean_apfdc
        #                 self.model.save(self.save_path)
        #                 if self.verbose > 0:
        #                     print(f"New best model with APFDC: {mean_apfdc:.4f}")
                        
        #                 # Save APFDC value
        #                 with open(os.path.join(self.log_dir, "best_apfdc.txt"), "w") as f:
        #                     f.write(f"{mean_apfdc:.6f}")
                
        #         return True



        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # if device == "cpu" and torch.backends.mps.is_available():
        #     device = "mps"
        # # print(f"Using device: {device}")
        
        # # Set path to your dataset
        # data_path = '/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv'
        
        # # Train model
        # model, train_env, eval_env = train_dqn_model(
        #     device=device,
        #     data_path=data_path,
        #     log_dir="./logs/tcp_dqn",
        #     total_timesteps=100000,
        # )
        
        # # Evaluate trained model
        # evaluate_model(model, eval_env, num_episodes=50)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        data, feature_cols = load_and_preprocess_data('/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv')

        # Create and wrap the environment for training
        def make_env():
            env = TCPEnv(
                dataset=data,
                feature_columns=feature_cols
            )
            return Monitor(env)  # Adds episode stats monitoring

        env = make_env()

        # device = "mps" if torch.backends.mps.is_available() else "cpu"
        # print(f"Using device: {device}")

        # data, feature_cols = load_and_preprocess_data('/Users/sanjeev/TCP-CI/TCP-CI-dataset/datasets/Angel-ML@angel/dataset.csv')

        # # Create and wrap the environment for training
        # def make_env():
        #     env = TestCasePrioritizationEnv(
        #         # dataset=data[data['Build'].isin(train_builds)],
        #         dataset=data,
        #         feature_columns=feature_cols
        #     )
        #     return Monitor(env)  # Adds episode stats monitoring

        # env = DummyVecEnv([make_env])
        # env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize observations and rewards

        # # Train DQN agent
        # dqn_agent = DQN(
        #     policy="MlpPolicy",
        #     env=env,
        #     learning_rate=1e-3,
        #     buffer_size=50000,
        #     learning_starts=1000,
        #     batch_size=32,
        #     tau=1.0,  # Target network update rate
        #     gamma=0.99,  # Discount factor
        #     train_freq=4,  # Update the model every 4 steps
        #     gradient_steps=1,
        #     target_update_interval=1000,
        #     exploration_fraction=0.1,
        #     exploration_initial_eps=1.0,
        #     exploration_final_eps=0.05,
        #     verbose=1,
        #     device=device,
        # )

        # # Train for a number of timesteps
        # dqn_agent.learn(total_timesteps=50000, log_interval=100)

        # # Save the trained model
        # dqn_agent.save("dqn_tcp_model")

        # dqn_agent = DQN.load("dqn_tcp_model")

        # # After training is complete and the model is saved:
        # print("Training complete. Evaluating model on each build...")

        # # Get chronologically ordered evaluation builds (use test builds or all builds)
        # eval_builds = sorted(data['Build'].unique())[:10]  # Use last 30 builds for evaluation
        # print(f"Evaluating on {len(eval_builds)} builds")

        # # Evaluate the model on each build
        # evaluation_results = evaluate_model_per_build(dqn_agent, data, feature_cols, eval_builds)
        # print(evaluation_results)

        # Save results to CSV
        # results_df = save_results_to_csv(evaluation_results)

        # # Create visualizations
        # plot_metrics_over_builds(results_df)

        # # Print summary statistics
        # print("\nSummary Statistics:")
        # print(f"Average APFD: {results_df['apfd_model'].mean():.4f} (Random: {results_df['apfd_random'].mean():.4f})")
        # print(f"Average APFDC: {results_df['apfdc_model'].mean():.4f} (Random: {results_df['apfdc_random'].mean():.4f})")
        # print(f"APFD Improvement: {(results_df['apfd_model'].mean() / results_df['apfd_random'].mean() - 1) * 100:.2f}%")
        # print(f"APFDC Improvement: {(results_df['apfdc_model'].mean() / results_df['apfdc_random'].mean() - 1) * 100:.2f}%")
