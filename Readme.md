# TCP-RL: Test Case Prioritization with Reinforcement Learning

_Enhancing test case prioritization in Continuous Integration contexts through reinforcement learning techniques_

## Project Overview

This project extends the work on scalable and accurate test case prioritization in Continuous Integration (CI) contexts by incorporating reinforcement learning (RL) techniques. While traditional approaches rely on heuristics or machine learning ranking models, our reinforcement learning approach enables more dynamic and adaptive prioritization strategies that can better identify fault-revealing test cases early in the test execution sequence.

![TCP-RL Framework](https://github.com/YourUsername/TCP-RL/raw/main/figures/tcp_rl_framework.png)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Dataset Creation](#dataset-creation)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Traditional Learning](#traditional-learning)
  - [Other Experiments](#other-experiments)
- [Results and Evaluation](#results-and-evaluation)
- [Original Work](#original-work)
- [License](#license)

## Features

- **Multiple RL Approaches**: Supports listwise, pairwise, and pointwise reinforcement learning strategies for test prioritization
- **Various RL Agents**: Implementation includes DQN (Deep Q-Network), PPO (Proximal Policy Optimization), and A2C (Advantage Actor-Critic) agents
- **Comprehensive Evaluation**: Automatic evaluation metrics including APFD (Average Percentage of Fault Detection) and improvement over original order
- **Visualization Tools**: Built-in tools for visualizing agent learning and performance
- **Integrated with Traditional Methods**: Can be used alongside traditional test prioritization techniques for comparison

## Installation

### Prerequisites

- Python 3.7+
- Java (for traditional RankLib models)
- Understand (for static code analysis)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/TCP-RL.git
   cd TCP-RL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install additional dependencies for reinforcement learning:
   ```bash
   pip install torch stable-baselines3
   ```

## Project Structure

```
TCP-RL/
├── data/                   # For datasets
│   └── final_datasets/     # Processed datasets ready for experiments
├── src/                    # Source code
│   └── python/
│       ├── code_analyzer/  # Static code analysis utilities
│       ├── services/       # Core services including RL modules
│       │   ├── rl_learn.py # RL training implementation
│       │   ├── dqn.py      # DQN agent implementation
│       │   ├── ppo.py      # PPO agent implementation
│       │   ├── a2c.py      # A2C agent implementation
│       │   ├── env.py      # RL environment implementations
│       │   └── rl_eval.py  # RL evaluation utilities
│       ├── entities/       # Data structures
│       └── feature_extractor/ # Feature extraction utilities
├── models/                 # Saved models from RL training
├── figures/                # Generated visualizations
├── results/                # Experiment results
└── main.py                 # Main entry point
```

## Usage

### Dataset Creation

Before running experiments, you need to create or prepare datasets:

```bash
# Process travis torrent logs
python main.py tr_torrent -i ../travistorrent-tools -o ../tr-torrent -r apache@commons

# Create dataset from GitHub repository
python main.py dataset -s apache/commons -c ../rtp-torrent -o ./datasets/apache@commons
```

### Reinforcement Learning

Run reinforcement learning experiments with:

```bash
# Basic RL experiment with DQN agent
python main.py learn --reinforcement-learning --folder-name Angel-ML@angel --output-path ./results --test-count 50 --episodes 1000 --agent-type dqn

# Use PPO agent instead
python main.py learn --reinforcement-learning --folder-name Angel-ML@angel --output-path ./results --agent-type ppo --episodes 500

# Use A2C agent
python main.py learn --reinforcement-learning --folder-name Angel-ML@angel --output-path ./results --agent-type a2c --episodes 800
```

Command line arguments:
- `--reinforcement-learning`: Flag to use reinforcement learning methods
- `--folder-name`: Dataset folder name (required)
- `--output-path`: Directory to save results
- `--test-count`: Number of builds to test (default: 50)
- `--agent-type`: RL agent to use: "dqn", "ppo", or "a2c" (default: "dqn")
- `--episodes`: Number of training episodes (default: 1000)

### Traditional Learning

Run traditional machine learning ranking experiments:

```bash
# Run with best ranking model (Random Forest)
python main.py learn -o ./datasets/apache@commons -t 50 -r best -e FULL

# Run with all ranking models
python main.py learn -o ./datasets/apache@commons -t 50 -r all
```

### Other Experiments

```bash
# Run decay test experiments
python main.py decay_test -o ./datasets/apache@commons -p ./datasets/apache@commons/commons

# Analyze results
python main.py results -d ./datasets/ -o ./results/
```

## Results and Evaluation

The reinforcement learning approach generates several evaluation metrics and visualizations:

- **APFD Scores**: Average percentage of fault detection for each prioritization method
- **Improvement**: Relative improvement over original test order
- **Learning Curves**: Visualizations showing the agent's learning progress
- **Comparison Charts**: Visual comparison between different approaches (listwise, pairwise, and pointwise)

Results are saved in the following directories:
- Models: `models/`
- Visualizations: `figures/`
- CSV metrics: `models/comparison*/`

## Original Work

This project extends the work on "Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts" by Saboor Yaraghi, et al. The original work is available at: [10.1109/TSE.2022.3184842](https://doi.org/10.1109/TSE.2022.3184842)

If using this work in academic research, please cite:

```bibtex
@article{yaraghi2021tcp,
    title={Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts},
    author={Saboor Yaraghi, Ahmadreza and Bagherzadeh, Mojtaba and Kahani, Nafiseh and Briand, Lionel},
    journal={IEEE Transactions on Software Engineering},
    year={2022},
    doi={10.1109/TSE.2022.3184842}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

[Your Name](mailto:your.email@example.com)