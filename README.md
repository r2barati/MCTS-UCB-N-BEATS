# MCTS-UCB-N-BEATS
Using MCTS UCB for Hyperparameter Tuning of N-BEATS

```markdown
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
