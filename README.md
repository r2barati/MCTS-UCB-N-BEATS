# MCTS-UCB-N-BEATS
Using MCTS UCB for Hyperparameter Tuning of N-BEATS

# Hyperparameter Tuning for Time Series Forecasting with N-BEATS

## Background

### 1. Problem Definition and Theoretical Framework in Hyperparameter Tuning

Hyperparameter tuning is a critical aspect of machine learning, particularly in time series forecasting models like N-BEATS. The goal is to optimize the model's performance through an effective selection of hyperparameters. The problem can be mathematically defined as:

```math
h* = arg min_{h∈H} L(M_h(D_{train}), D_{test})
```

where `M_h` represents a model parameterized by hyperparameters `h`, `D_train` is the training dataset, and `D_test` is the testing dataset. The function `L` typically denotes a loss metric such as Mean Absolute Percentage Error (MAPE), which evaluates the accuracy of the model's predictions.

#### Loss Functions and Model Evaluation

The choice of the loss function is pivotal, as it influences the configurations deemed optimal. For time series forecasting, MAPE is often used due to its scale-independence and interpretability, defined as:

```math
MAPE = (100 / n) * Σ|((A_t - F_t) / A_t)|
```

where `A_t` are the actual values and `F_t` are the forecast values.

#### Hyperparameter Space

The hyperparameter space `H` encompasses all possible combinations that can be adjusted in the model, represented as a Cartesian product of all possible values of each hyperparameter.

#### Optimization Strategies

The optimization process involves navigating through `H` to find the set of hyperparameters `h` that minimizes `L`. Strategies for this search include Grid Search, Random Search, and Monte Carlo Tree Search (MCTS), each with distinct mechanisms and benefits.

### 2. Overview of the N-BEATS Model

The N-BEATS model employs a fully-connected neural network approach structured around blocks, each comprising several layers. The model's architecture allows it to adapt flexibly to different temporal patterns and complexities in the data.

### 3. Hyperparameter Optimization Methods

#### Grid Search

Grid Search systematically explores a predefined grid of hyperparameters, evaluating each combination to find the one that offers the best performance.

#### Random Search

Random Search selects random configurations from the hyperparameter space, which can provide a good solution more efficiently than Grid Search by avoiding exhaustive exploration.

#### Monte Carlo Tree Search (MCTS)

MCTS uses a decision tree where each node represents a hyperparameter choice, balancing exploration of new configurations with exploitation of known effective ones. Nodes are expanded based on the Upper Confidence Bound (UCB) formula, making it suitable for complex spaces:

```math
UCB1 = X_j + C * sqrt((2 * log(N)) / n_j)
```

### Comparative Analysis

While Grid Search is thorough, it may not be practical for large spaces. Random Search offers a balance between efficiency and coverage, whereas MCTS provides a strategic exploration that can effectively identify optimal configurations.

## Implementation Details

This repository contains implementations of Grid Search, Random Search, and two variations of Monte Carlo Tree Search (UCB with c = 0.5 and c = 1.44) for the N-BEATS model. Detailed code examples are provided to demonstrate the application of these methods in optimizing hyperparameters for time series forecasting.

### Getting Started

To run the provided examples, please refer to the individual notebook files for each method, which include comprehensive instructions and explanations of the underlying processes.

## Contributing

Contributions to improve the implementations or extend the repository's capabilities are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is released under the MIT License. Details can be found in the LICENSE file.

This introduction sets the stage for the detailed implementation notebooks, providing a theoretical foundation and context for the optimization techniques used in the project. It prepares users to understand the importance and implications of each method's application to the N-BEATS model.

# Monte Carlo Tree Search for N-BEATS Model Optimization

This repository contains a Python implementation of a Monte Carlo Tree Search (MCTS) applied to the hyperparameter optimization of the N-BEATS forecasting model from the Darts library. The project leverages PyTorch for neural network operations and utilizes various optimization algorithms.

## Features

- Implementation of the MCTS algorithm to explore and optimize model parameters.
- Use of PyTorch and Darts libraries for constructing and training the N-BEATS model.
- Extensive use of logging and tqdm for tracking the progress of computations.

## Requirements

To run this project, you will need the following libraries:

- `math`
- `random`
- `time`
- `pandas`
- `torch`
- `darts`

You can install the required Python packages using the following command:

```bash
pip install pandas torch darts tqdm
```

Ensure that you have a compatible version of PyTorch installed, suited to your system's specifications (CPU or GPU).

## Usage

To execute the MCTS for N-BEATS model optimization, run the script from the command line:

```bash
python mcts_nbeats_optimization.py
```

This will initiate a series of MCTS iterations to find optimal model settings, saving the results and iteration times to a CSV file.

## Structure

The main components of the script include:

- `Node` class for managing tree nodes.
- `MCTree` class for handling the tree structure and MCTS operations.
- Execution loop to perform MCTS iterations.
- Logging and progress tracking using `tqdm`.

The optimization results are stored in 'mcts_search_results_low.csv' after running the script.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
```
