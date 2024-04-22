# MCTS-UCB-N-BEATS
Using MCTS UCB for Hyperparameter Tuning of N-BEATS

# N-BEATS Hyperparameter Tuning using MCTS-UCB

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

This introduction sets the stage for the detailed implementation notebooks, providing a theoretical foundation and context for the optimization techniques used in the project. It prepares users to understand the importance and implications of each method's application to the N-BEATS model.

### Monte Carlo Tree Search for N-BEATS Model Optimization

This repository contains a Python implementation of a Monte Carlo Tree Search (MCTS) applied to the hyperparameter optimization of the N-BEATS forecasting model from the Darts library. The project leverages PyTorch for neural network operations and utilizes various optimization algorithms.

#### Features

- Implementation of the MCTS algorithm to explore and optimize model parameters.
- Use of PyTorch and Darts libraries for constructing and training the N-BEATS model.
- Extensive use of logging and tqdm for tracking the progress of computations.

#### Requirements

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

#### Usage

To execute the MCTS for N-BEATS model optimization, run the script from the command line:

```bash
python mcts_nbeats_optimization.py
```

This will initiate a series of MCTS iterations to find optimal model settings, saving the results and iteration times to a CSV file.

#### Structure, Code Explanation and Detailed Functionality

##### 1. **Node Class**

The `Node` class represents each hyperparameter combination as a node in the search tree. Here's a more detailed look at its methods and attributes:

- **Attributes**:
  - `parent`: Reference to the parent node, used to navigate back up the tree.
  - `children`: List of child nodes, representing subsequent hyperparameter choices.
  - `value`: The specific value of the hyperparameter this node represents.
  - `variable`: The name of the hyperparameter.
  - `visits`: Counts how many times this node has been visited during the search.
  - `total_value`: Accumulates the total reward value obtained from model evaluations involving this node's hyperparameter settings.

- **Methods**:
  - `update(reward)`: Updates the node's statistics by incrementing the visit count and adding the reward obtained from a model evaluation. The reward typically depends on the model's performance metric, inversely related to the error rate.
  - `calculate_ucb1(max_reward_seen)`: Calculates the Upper Confidence Bound (UCB) value for the node, which is used to balance exploration and exploitation during the search. The formula considers the number of visits to this node, the number of visits to the parent node, the node's average reward, and an exploration factor. It aims to prioritize nodes that have either been promising or not sufficiently explored.

##### 2. **MCTree Class**

This class manages the overall tree structure and execution of the MCTS algorithm:

- **Attributes**:
  - `root`: The initial node of the tree, typically representing a baseline or null hyperparameter set.
  - `fixed_values`: A dictionary containing the mapping of each hyperparameter to its possible values.
  - `max_reward_seen`: Tracks the highest reward encountered during the search, aiding in the normalization of rewards.
  - `node_registry`: Helps in tracking all nodes and ensuring that each combination of hyperparameters is uniquely represented.
  - `results`: List to store the results of each complete path exploration, including hyperparameters and their evaluation metrics.
  - `iteration_times`: Records the time taken for each iteration to aid in performance analysis.

- **Methods**:
  - `run_iteration()`: This is a critical function where the MCTS algorithm is executed:
    1. **Traversal**: Starting from the root, the tree is traversed down to a leaf node based on the UCB values.
    2. **Expansion**: If the current node can be expanded (i.e., there are more hyperparameter values to explore), children are created and added.
    3. **Simulation and Backpropagation**: Once a leaf node is reached, the model is evaluated, and the resulting reward is backpropagated up the tree to update the nodes' statistics.
  - `save_results()`: After all iterations are complete, this method saves the results to a CSV file, providing a persistent record of the search outcomes.

##### 3. **Model Evaluation**

- **evaluate_model(params)**: This function encapsulates the evaluation of the N-BEATS model given a set of hyperparameters:
  - **Model Setup**: The model is configured with parameters like the number of blocks, layers, learning rate, etc., and trained on the training dataset.
  - **Prediction and Scoring**: After training, the model makes predictions, which are compared to the actual test data to compute the Mean Absolute Percentage Error (MAPE). The reward is then calculated as the inverse of MAPE to frame it as a maximization problem.

##### 4. **Execution Loop**

The main execution starts with defining the hyperparameter space, initializing the tree, and running a fixed number of iterations to explore the space effectively:

- **Initialization**: The search space for each hyperparameter is defined and converted into a structured format that the `MCTree` can use.
- **MCTS Execution**: The tree performs a predefined number of iterations, with each iteration exploring new paths or further exploring promising paths, driven by the UCB strategy.

This breakdown explains how the different parts of the code work together to implement an effective MCTS algorithm for hyperparameter tuning in the context of time series forecasting with the N-BEATS model.

##### Detailed Code Breakdown

###### Key Imports and Configuration

- **Basic Libraries**: The code uses Python libraries like `math`, `random`, `time`, `pandas` for data handling, and `tqdm` for progress tracking.
- **PyTorch and Darts**: `torch` and specific classes from `darts` are imported to build and evaluate the forecasting model. This includes the `NBEATSModel`, `Scaler` for data scaling, and `mape` for evaluating model accuracy.
- **Logging**: Logging for PyTorch is set to `ERROR` to reduce console clutter during model training.

```python
import math
import random
import time
import pandas as pd
import torch
from torch.optim import Adam, SGD, RMSprop
from darts.models.forecasting.nbeats import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import logging
from tqdm import tqdm

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
```

##### Node Class

- **Initialization**: Each node in the MCTS represents a set of hyperparameters with properties to track the node's visits and total value from evaluations.
- **Update and UCB Calculation**: The `update` method increments the visit count and accumulates value, while `calculate_ucb1` computes the UCB value used for navigating the search space, balancing exploration and exploitation.

```python
class Node:
    def __init__(self, parent=None, value=None, variable=None, fixed_values=None):
        self.parent = parent
        self.children = []
        self.value = value
        self.variable = variable
        self.visits = 0
        self.total_value = 0
        if fixed_values and variable is not None:
            self.fixed_node_value = fixed_values[variable][value]
        else:
            self.fixed_node_value = 0  

    def update(self, reward):
        self.visits += 1
        self.total_value += reward

    def calculate_ucb1(self, max_reward_seen):
        ...
```

##### MCTree Class

- **Tree Initialization**: Sets up the root node and manages the registry of nodes.
- **Running Iterations**: Conducts MCTS iterations by navigating from the root to suitable leaf nodes, expanding the tree, and updating nodes based on model evaluation outcomes.
- **Saving Results**: Results of the search, including the hyperparameter configurations and corresponding performance metrics, are saved to a CSV file.

```python
class MCTree:
    def __init__(self, fixed_values):
        self.root = Node(fixed_values=fixed_values)
        self.fixed_values = fixed_values
        self.max_reward_seen = 0
        ...

    def run_iteration(self):
        ...
        
    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_df['Iteration Time'] = pd.Series(self.iteration_times)
        ...
```

##### Model Evaluation

- **Model Setup and Training**: Configures the N-BEATS model with selected hyperparameters and trains it.
- **Performance Evaluation**: The model's performance is measured using the MAPE, which influences the reward for the MCTS.

```python
def evaluate_model(self, params):
    model = NBEATSModel(...)
    model.fit(train, verbose=False)
    prediction = model.predict(len(test))
    prediction_rescaled = scaler.inverse_transform(prediction)
    test_rescaled = scaler.inverse_transform(test)

    model_mape = mape(test_rescaled, prediction_rescaled)

    return 1 / model_mape if model_mape != 0 else float('inf')
```

##### Execution and Result Storage

- **Initialization**: Defines the search space for the hyperparameters and prepares the tree for iterations.
- **Loop Execution**: Runs a predetermined number of MCTS iterations to explore the hyperparameter space thoroughly.
- **Results**: Outputs are saved and potentially displayed or analyzed further.

```python
search_space = { ... }

fixed_values = {}
for i, (key, values) in enumerate(search_space.items()):
    fixed_values[key] = {i: value for i, value in enumerate(values)}

tree = MCTree(fixed_values=fixed_values)
for _ in range(100):  
    tree.run_iteration()

tree.save_results()
```

This detailed breakdown guides the user through each part of the script, clarifying how the MCTS is applied for optimizing hyperparameters in the N-BEATS model. This approach not only elucidates the operational aspects but also links them to the theoretical concepts outlined earlier in the README.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
```
