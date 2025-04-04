# TCP-RL: Real-Time Test Case Prioritization for CI/CD using Reinforcement Learning

_Enhancing test case prioritization in Continuous Integration contexts through reinforcement learning techniques_

[Sanjeev Vijayakumar](mailto:sv8958@rit.edu)

## Project Overview

This project extends the work on scalable and accurate test case prioritization in Continuous Integration (CI) contexts by incorporating reinforcement learning (RL) techniques. While traditional approaches rely on heuristics or machine learning ranking models, this reinforcement learning approach enables more dynamic and adaptive prioritization strategies that can better identify fault-revealing test cases early in the test execution sequence.

![Key Motivation](https://github.com/sanjeevmax6/TCP-RL/raw/main/figures/entity_changes_histogram.png)

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
- **Various RL Agents**: Implementation includes DQN (Deep Q-Network), PPO (Proximal Policy Optimization), and A2C (Advantage Actor-Critic) agents. [NOTE: PPO and A2C are still being built]
- **Comprehensive Evaluation**: Automatic evaluation metrics including APFD (Average Percentage of Fault Detection) and APFDC (Average Percentage of Fault Detection with execution costs as well)
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

![Dataset preparation](https://github.com/sanjeevmax6/TCP-RL/raw/main/figures/dataset_processing.png)

For this project to run, the final datasets for 3 repositories is already prepared and resides in the 'data' folder. However if you prefer to generate the datasets from raw_datasets within the 'data' folder, you need to install Understand and Java first.

### Understand
Understand is a code analysis enterprise software with a wide variety of [supported languages](https://support.scitools.com/support/solutions/articles/70000582794-supported-languages) which provides static dependencies available in a source code between files, functions, classes, etc. For more details on the feature of this software, visit [this link](https://scitools.com/features). In this project, we utilize Understand to create static dependency graphs to collect a part of our features. 

In this section, we will explain how to install and set up Understand to obtain a file with `.und` format which is the output of Understand's analysis. Note that this project needs Understand's database for extracting features and will not work without it.

#### Installing Understand's CLI
You can download the latest stable version of Understand from [this link](https://licensing.scitools.com/download). To run this project, you need to add the `und` command to your PATH environment variable so the `und` command is recognized in the shell. `und` is located in the `bin` directory of Understand's software.

```bash
export PATH="$PATH:/path/to/understand/scitools/bin/linux64"
```

Finally, run the following command to make sure `und` is successfully installed:

```bash
$ und version
(Build 1029)
```

#### Note
This project has been tested on *Build 1029* of Understand on Linux (specifically Ubuntu). It may require minor compatibility changes if it is used on other Understand builds or other operating systems.

#### Adding Understand Python Package/Library
Unlike typical projects, Understand does not provide its Python library in the well-known pip package installer, and you need to manually add the package to your Python environment. The instructions for adding the package are explained in [this link](https://support.scitools.com/support/solutions/articles/70000582852-getting-started-with-the-python-api).

### Java
This project uses [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib) for training and testing machine learning ranking models. RankLib is a library of learning-to-rank algorithms, and it is written in Java. Hence, this project requires Java for running training and testing experiments. This project is trained and tested on OpenJDK version `1.8.0_292` and `11.0.11`.

![TCP-RL Framework](https://github.com/sanjeevmax6/TCP-RL/raw/main/figures/methodology.png)

## Project Structure

```
TCP-RL/
├── data/                   # For datasets
│   └── final_datasets/     # Processed datasets ready for experiments
├── src/                    # Source code
│   └── python/
│       ├── code_analyzer/                  # Static code analysis utilities for Understand
|       |__ commit_classifer                # utils for classification of commits while preparing dataset
│       ├── entities/                       # Data structures for build runs using travis torrent tool
│       └── feature_extractor/              # Feature extraction utilities
│       ├── results/                        # for ML based models
│       ├── services/                       # Core services including ML and RL modules
│       │   ├── rl_learn.py                 # RL training implementation
│       │   ├── dqn.py                      # DQN agent implementation
│       │   ├── ppo.py                      # PPO agent implementation
│       │   ├── a2c.py                      # A2C agent implementation
│       │   ├── env.py                      # RL environment implementations
│       │   └── rl_eval.py                  # RL evaluation utilities
│       │   └── data_collection_service.py  # main file to prepare dataset
│       │   └── experiments_service.py      # main file that calls ML models
├── models/                # Saved models from RL training
├── figures/               # Generated visualizations
├── results/               # Experiment results
└── main.py                # Main entry point
```

## Usage

### Dataset Creation (Optional as this has already been done)

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

### Other Experiments for ML

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

[Sanjeev Vijayakumar](mailto:sv8958@rit.edu)