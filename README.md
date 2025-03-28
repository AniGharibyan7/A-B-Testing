# A/B Testing: Multi-Armed Bandit Algorithms

This repository contains implementations of different bandit algorithms to solve the exploration-exploitation dilemma in multi-armed bandit problems.

## Algorithms Implemented

1. **Epsilon Greedy**: Uses an epsilon parameter to balance between exploration (random arm selection) and exploitation (selecting the best arm). The implementation includes epsilon decay which reduces the exploration rate over time.

2. **Thompson Sampling**: A Bayesian approach that models uncertainty about each arm's reward probability using Beta distributions and makes decisions by sampling from these distributions.

3. **Bayesian UCB (Bonus)**: Combines Bayesian approach with Upper Confidence Bound principle, using quantiles from posterior distributions to make decisions, with adaptive quantile adjustment over time.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Pandas
- SciPy
- Loguru

You can install the required packages using:

```
pip install -r requirements.txt
```

## Usage

Run the main script to execute experiments with all three algorithms:

```
python Bandit.py
```

This will:
1. Run experiments for all three algorithms
2. Save data to CSV files
3. Generate performance comparison visualizations
4. Print summary statistics

## Visualization

The code produces several visualizations:
- Average reward over time (linear and log scale)
- Cumulative regret over time (linear and log scale)
- Comparison plots of all three algorithms

## Results

The algorithms are evaluated based on:
- Average reward
- Cumulative reward
- Cumulative regret

Results will vary due to the stochastic nature of the bandit problem, but generally:
- Thompson Sampling and Bayesian UCB tend to outperform Epsilon Greedy
- The best algorithm depends on the specific problem parameters and random initialization

## Implementation Details

- Each algorithm is implemented as a class inheriting from an abstract `Bandit` base class
- The `Visualization` class provides methods for plotting results
- Results are stored in CSV files for further analysis
- Comprehensive logging is included for tracking experiment progress

## License

This project is licensed under the MIT License - see the LICENSE file for details.
