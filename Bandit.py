"""
Multi-Armed Bandit Problem Implementation
- Epsilon Greedy Algorithm with decay
- Thompson Sampling Algorithm
- Bayesian UCB Algorithm (Bonus)
- Visualization and reporting tools

This module implements different bandit algorithms to solve the exploration-exploitation
dilemma in a multi-armed bandit setting, as well as tools for visualization and comparison.
"""
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import sys
import scipy.stats
from typing import List, Tuple, Dict


class Bandit(ABC):
    """
    Abstract Bandit class defining the interface for bandit algorithms.
    All bandit algorithm implementations should inherit from this class.
    """
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def experiment(self, n_trials):
        pass

    @abstractmethod
    def report(self):
        pass


class Visualization:
    """
    Class for visualizing the results of the bandit experiments.
    Provides methods for creating plots of rewards and regrets.
    """
    def plot1(self, eg_results, ts_results):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(eg_results['avg_rewards'], label='Epsilon Greedy')
        axes[0, 0].plot(ts_results['avg_rewards'], label='Thompson Sampling')
        axes[0, 0].set_title('Average Reward vs. Trials')
        axes[0, 0].set_xlabel('Trials')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(eg_results['avg_rewards'], label='Epsilon Greedy')
        axes[0, 1].plot(ts_results['avg_rewards'], label='Thompson Sampling')
        axes[0, 1].set_title('Average Reward vs. Trials (Log Scale)')
        axes[0, 1].set_xlabel('Trials')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(eg_results['cum_regrets'], label='Epsilon Greedy')
        axes[1, 0].plot(ts_results['cum_regrets'], label='Thompson Sampling')
        axes[1, 0].set_title('Cumulative Regret vs. Trials')
        axes[1, 0].set_xlabel('Trials')
        axes[1, 0].set_ylabel('Cumulative Regret')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(eg_results['cum_regrets'], label='Epsilon Greedy')
        axes[1, 1].plot(ts_results['cum_regrets'], label='Thompson Sampling')
        axes[1, 1].set_title('Cumulative Regret vs. Trials (Log Scale)')
        axes[1, 1].set_xlabel('Trials')
        axes[1, 1].set_ylabel('Cumulative Regret')
        axes[1, 1].set_xscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('learning_process.png')
        logger.info("Learning process visualization saved as 'learning_process.png'")
        plt.show()

    def plot2(self, eg_results, ts_results):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(eg_results['cum_rewards'], label='Epsilon Greedy')
        axes[0].plot(ts_results['cum_rewards'], label='Thompson Sampling')
        axes[0].set_title('Cumulative Reward vs. Trials')
        axes[0].set_xlabel('Trials')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(eg_results['cum_regrets'], label='Epsilon Greedy')
        axes[1].plot(ts_results['cum_regrets'], label='Thompson Sampling')
        axes[1].set_title('Cumulative Regret vs. Trials')
        axes[1].set_xlabel('Trials')
        axes[1].set_ylabel('Cumulative Regret')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('cumulative_comparison.png')
        logger.info("Cumulative comparison visualization saved as 'cumulative_comparison.png'")
        plt.show()


class EpsilonGreedy(Bandit):
    """
    Epsilon Greedy algorithm implementation for the multi-armed bandit problem.
    Balances exploration and exploitation using an epsilon parameter that
    controls the probability of random exploration vs. greedy exploitation.
    """
    def __init__(self, p, epsilon=1.0):
        """
        Initialize the Epsilon Greedy bandit.

        Args:
            p (List[float]): List of true probabilities for each arm
            epsilon (float): Initial exploration rate (probability of random arm selection)
        """
        self.p = p
        self.n_arms = len(p)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.epsilon = epsilon

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []

        logger.info(f"EpsilonGreedy initialized with {self.n_arms} arms and epsilon={self.epsilon}")

    def __repr__(self):
        return f"EpsilonGreedy(arms={self.n_arms}, epsilon={self.epsilon})"

    def pull(self, t=None):
        current_epsilon = self.epsilon
        if t is not None:
            current_epsilon = self.epsilon / (t + 1)  # Decay by 1/t

        if np.random.random() < current_epsilon:
            arm = np.random.randint(self.n_arms)
            logger.debug(f"Exploring with arm {arm}")

        else:
            arm = np.argmax(self.values)
            logger.debug(f"Exploiting with arm {arm}")

        reward = self.p[arm] if np.random.random() < self.p[arm] else 0

        return arm, reward

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        logger.debug(f"Updated arm {arm} with reward {reward}, new value: {self.values[arm]}")

    def experiment(self, n_trials):
        logger.info(f"Starting EpsilonGreedy experiment with {n_trials} trials")

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []

        total_reward = 0
        total_regret = 0
        optimal_arm = np.argmax(self.p)

        for t in range(n_trials):
            arm, reward = self.pull(t)
            regret = self.p[optimal_arm] - reward
            self.update(arm, reward)

            self.rewards.append(reward)
            self.arms_pulled.append(arm)
            self.regrets.append(regret)

            total_reward += reward
            total_regret += regret
            self.cum_rewards.append(total_reward)
            self.cum_regrets.append(total_regret)
            self.avg_rewards.append(total_reward / (t + 1))

            if (t + 1) % 1000 == 0:
                logger.info(f"EpsilonGreedy trial {t + 1}/{n_trials}: Avg Reward = {self.avg_rewards[-1]:.4f}")

        logger.info(f"EpsilonGreedy experiment completed: Final Avg Reward = {self.avg_rewards[-1]:.4f}")

        return {
            'rewards': self.rewards,
            'arms_pulled': self.arms_pulled,
            'regrets': self.regrets,
            'cum_rewards': self.cum_rewards,
            'cum_regrets': self.cum_regrets,
            'avg_rewards': self.avg_rewards,
            'total_reward': total_reward,
            'total_regret': total_regret,
            'avg_reward': total_reward / n_trials,
        }

    def report(self):
        if not self.rewards:
            logger.error("No experiment data available to report")
            return

        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)

        with open('epsilon_greedy_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Trial', 'Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, (arm, reward) in enumerate(zip(self.arms_pulled, self.rewards)):
                writer.writerow({
                    'Trial': i + 1,
                    'Bandit': arm + 1,
                    'Reward': reward,
                    'Algorithm': 'EpsilonGreedy'
                })

        logger.info(f"EpsilonGreedy results saved to 'epsilon_greedy_results.csv'")

        print(f"Epsilon Greedy - Average Reward: {avg_reward:.4f}")
        print(f"Epsilon Greedy - Average Regret: {avg_regret:.4f}")
        print(f"Epsilon Greedy - Cumulative Reward: {self.cum_rewards[-1]:.4f}")
        print(f"Epsilon Greedy - Cumulative Regret: {self.cum_regrets[-1]:.4f}")

        return {
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'cum_reward': self.cum_rewards[-1],
            'cum_regret': self.cum_regrets[-1]
        }


class ThompsonSampling(Bandit):

    """
    Thompson Sampling algorithm implementation for the multi-armed bandit problem.
    Uses Bayesian approach with Beta distribution to model uncertainty about arm rewards.
    """

    def __init__(self, p, alpha=1.0, beta=1.0):
        """
        Initialize the Thompson Sampling bandit.

        Args:
            p (List[float]): List of true probabilities for each arm
            alpha (float): Initial alpha parameter for Beta distribution (prior successes)
            beta (float): Initial beta parameter for Beta distribution (prior failures)
        """
        self.p = p
        self.n_arms = len(p)
        self.alpha = np.ones(self.n_arms) * alpha
        self.beta = np.ones(self.n_arms) * beta

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []

        logger.info(f"ThompsonSampling initialized with {self.n_arms} arms, alpha={alpha}, beta={beta}")

    def __repr__(self):
        return f"ThompsonSampling(arms={self.n_arms}, alpha={self.alpha[0]}, beta={self.beta[0]})"

    def pull(self):

        samples = np.random.beta(self.alpha, self.beta)
        arm = np.argmax(samples)
        logger.debug(f"Selected arm {arm} with sample value {samples[arm]:.4f}")

        reward = self.p[arm] if np.random.random() < self.p[arm] else 0
        return arm, reward

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        logger.debug(f"Updated arm {arm} with reward {reward}, new alpha={self.alpha[arm]}, beta={self.beta[arm]}")

    def experiment(self, n_trials):
        logger.info(f"Starting ThompsonSampling experiment with {n_trials} trials")

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []

        total_reward = 0
        total_regret = 0
        optimal_arm = np.argmax(self.p)

        for t in range(n_trials):
            arm, reward = self.pull()
            regret = self.p[optimal_arm] - reward
            self.update(arm, reward)

            self.rewards.append(reward)
            self.arms_pulled.append(arm)
            self.regrets.append(regret)

            total_reward += reward
            total_regret += regret
            self.cum_rewards.append(total_reward)
            self.cum_regrets.append(total_regret)
            self.avg_rewards.append(total_reward / (t + 1))

            if (t + 1) % 1000 == 0:
                logger.info(f"ThompsonSampling trial {t + 1}/{n_trials}: Avg Reward = {self.avg_rewards[-1]:.4f}")

        logger.info(f"ThompsonSampling experiment completed: Final Avg Reward = {self.avg_rewards[-1]:.4f}")

        return {
            'rewards': self.rewards,
            'arms_pulled': self.arms_pulled,
            'regrets': self.regrets,
            'cum_rewards': self.cum_rewards,
            'cum_regrets': self.cum_regrets,
            'avg_rewards': self.avg_rewards,
            'total_reward': total_reward,
            'total_regret': total_regret,
            'avg_reward': total_reward / n_trials,
        }

    def report(self):
        if not self.rewards:
            logger.error("No experiment data available to report")
            return

        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)

        with open('thompson_sampling_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Trial', 'Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, (arm, reward) in enumerate(zip(self.arms_pulled, self.rewards)):
                writer.writerow({
                    'Trial': i + 1,
                    'Bandit': arm + 1,
                    'Reward': reward,
                    'Algorithm': 'ThompsonSampling'
                })

        logger.info(f"ThompsonSampling results saved to 'thompson_sampling_results.csv'")

        print(f"Thompson Sampling - Average Reward: {avg_reward:.4f}")
        print(f"Thompson Sampling - Average Regret: {avg_regret:.4f}")
        print(f"Thompson Sampling - Cumulative Reward: {self.cum_rewards[-1]:.4f}")
        print(f"Thompson Sampling - Cumulative Regret: {self.cum_regrets[-1]:.4f}")

        return {
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'cum_reward': self.cum_rewards[-1],
            'cum_regret': self.cum_regrets[-1]
        }


def comparison(eg_results, ts_results):
    with open('comparison_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Trial', 'Bandit', 'Reward', 'Algorithm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, (arm, reward) in enumerate(zip(eg_results['arms_pulled'], eg_results['rewards'])):
            writer.writerow({
                'Trial': i + 1,
                'Bandit': arm + 1,
                'Reward': reward,
                'Algorithm': 'EpsilonGreedy'
            })

        for i, (arm, reward) in enumerate(zip(ts_results['arms_pulled'], ts_results['rewards'])):
            writer.writerow({
                'Trial': i + 1,
                'Bandit': arm + 1,
                'Reward': reward,
                'Algorithm': 'ThompsonSampling'
            })

    logger.info(f"Combined results saved to 'comparison_results.csv'")

    viz = Visualization()
    viz.plot1(eg_results, ts_results)
    viz.plot2(eg_results, ts_results)

    print("\nALGORITHM COMPARISON:")
    print(f"{'Algorithm':<20} {'Avg Reward':<15} {'Avg Regret':<15} {'Cum Reward':<15} {'Cum Regret':<15}")
    print("-" * 80)
    print(
        f"{'Epsilon Greedy':<20} {eg_results['avg_reward']:<15.4f} {eg_results['total_regret'] / len(eg_results['rewards']):<15.4f} {eg_results['cum_rewards'][-1]:<15.4f} {eg_results['cum_regrets'][-1]:<15.4f}")
    print(
        f"{'Thompson Sampling':<20} {ts_results['avg_reward']:<15.4f} {ts_results['total_regret'] / len(ts_results['rewards']):<15.4f} {ts_results['cum_rewards'][-1]:<15.4f} {ts_results['cum_regrets'][-1]:<15.4f}")

    if eg_results['cum_rewards'][-1] > ts_results['cum_rewards'][-1]:
        print("\nEpsilon Greedy achieved higher cumulative reward.")
    elif eg_results['cum_rewards'][-1] < ts_results['cum_rewards'][-1]:
        print("\nThompson Sampling achieved higher cumulative reward.")
    else:
        print("\nBoth algorithms achieved the same cumulative reward.")


# BONUS IMPLEMENTATION: Bayesian UCB
class BayesianUCB(Bandit):
    """
    Bayesian Upper Confidence Bound algorithm - combines Bayesian approach
    with UCB principle for improved exploration-exploitation balance.

    This algorithm uses quantiles from the posterior distribution to make decisions,
    allowing for more precise uncertainty quantification than standard UCB.
    """
    def __init__(self, p, alpha=1.0, beta=1.0, quantile=0.95, quantile_increase=0.01):
        """
        Initialize the Bayesian UCB bandit.

        Args:
            p (List[float]): List of true probabilities for each arm
            alpha (float): Initial alpha parameter for Beta distribution (prior successes)
            beta (float): Initial beta parameter for Beta distribution (prior failures)
            quantile (float): Initial quantile for UCB (higher means more exploration)
            quantile_increase (float): Rate at which quantile adapts over time
        """
        self.p = p
        self.n_arms = len(p)
        self.alpha = np.ones(self.n_arms) * alpha  # Prior alpha parameters
        self.beta = np.ones(self.n_arms) * beta  # Prior beta parameters
        self.quantile = quantile  # Initial quantile for UCB
        self.quantile_increase = quantile_increase  # Rate at which quantile adapts
        self.t = 0  # Time step counter

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []

        logger.info(f"BayesianUCB initialized with {self.n_arms} arms, alpha={alpha}, beta={beta}, quantile={quantile}")

    def __repr__(self):
        return f"BayesianUCB(arms={self.n_arms}, alpha={self.alpha[0]}, beta={self.beta[0]}, quantile={self.quantile})"

    def pull(self):
        self.t += 1

        current_quantile = min(1.0 - 1.0 / self.t, self.quantile + self.quantile_increase * np.log(self.t))

        upper_bounds = np.array([
            scipy.stats.beta.ppf(current_quantile, a, b)
            for a, b in zip(self.alpha, self.beta)
        ])

        arm = np.argmax(upper_bounds)
        logger.debug(f"Selected arm {arm} with upper bound {upper_bounds[arm]:.4f} at quantile {current_quantile:.4f}")

        reward = self.p[arm] if np.random.random() < self.p[arm] else 0
        return arm, reward

    def update(self, arm, reward):
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

        logger.debug(f"Updated arm {arm} with reward {reward}, new alpha={self.alpha[arm]}, beta={self.beta[arm]}")

    def experiment(self, n_trials):
        logger.info(f"Starting BayesianUCB experiment with {n_trials} trials")

        self.rewards = []
        self.arms_pulled = []
        self.regrets = []
        self.cum_rewards = []
        self.cum_regrets = []
        self.avg_rewards = []
        self.t = 0

        total_reward = 0
        total_regret = 0
        optimal_arm = np.argmax(self.p)

        for t in range(n_trials):
            arm, reward = self.pull()
            regret = self.p[optimal_arm] - reward
            self.update(arm, reward)

            self.rewards.append(reward)
            self.arms_pulled.append(arm)
            self.regrets.append(regret)

            total_reward += reward
            total_regret += regret
            self.cum_rewards.append(total_reward)
            self.cum_regrets.append(total_regret)
            self.avg_rewards.append(total_reward / (t + 1))

            if (t + 1) % 1000 == 0:
                logger.info(f"BayesianUCB trial {t + 1}/{n_trials}: Avg Reward = {self.avg_rewards[-1]:.4f}")

        logger.info(f"BayesianUCB experiment completed: Final Avg Reward = {self.avg_rewards[-1]:.4f}")

        return {
            'rewards': self.rewards,
            'arms_pulled': self.arms_pulled,
            'regrets': self.regrets,
            'cum_rewards': self.cum_rewards,
            'cum_regrets': self.cum_regrets,
            'avg_rewards': self.avg_rewards,
            'total_reward': total_reward,
            'total_regret': total_regret,
            'avg_reward': total_reward / n_trials,
        }

    def report(self):
        if not self.rewards:
            logger.error("No experiment data available to report")
            return

        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)

        with open('bayesian_ucb_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Trial', 'Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, (arm, reward) in enumerate(zip(self.arms_pulled, self.rewards)):
                writer.writerow({
                    'Trial': i + 1,
                    'Bandit': arm + 1,
                    'Reward': reward,
                    'Algorithm': 'BayesianUCB'
                })

        logger.info(f"BayesianUCB results saved to 'bayesian_ucb_results.csv'")

        print(f"BayesianUCB - Average Reward: {avg_reward:.4f}")
        print(f"BayesianUCB - Average Regret: {avg_regret:.4f}")
        print(f"BayesianUCB - Cumulative Reward: {self.cum_rewards[-1]:.4f}")
        print(f"BayesianUCB - Cumulative Regret: {self.cum_regrets[-1]:.4f}")

        return {
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'cum_reward': self.cum_rewards[-1],
            'cum_regret': self.cum_regrets[-1]
        }


def compare_all_algorithms(eg_results, ts_results, bayesian_ucb_results):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(eg_results['avg_rewards'], label='Epsilon Greedy')
    axes[0].plot(ts_results['avg_rewards'], label='Thompson Sampling')
    axes[0].plot(bayesian_ucb_results['avg_rewards'], label='Bayesian UCB (Bonus)')
    axes[0].set_title('Average Reward vs. Trials')
    axes[0].set_xlabel('Trials')
    axes[0].set_ylabel('Average Reward')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(eg_results['cum_regrets'], label='Epsilon Greedy')
    axes[1].plot(ts_results['cum_regrets'], label='Thompson Sampling')
    axes[1].plot(bayesian_ucb_results['cum_regrets'], label='Bayesian UCB (Bonus)')
    axes[1].set_title('Cumulative Regret vs. Trials')
    axes[1].set_xlabel('Trials')
    axes[1].set_ylabel('Cumulative Regret')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('all_algorithms_comparison.png')
    logger.info("All algorithms comparison visualization saved as 'all_algorithms_comparison.png'")
    plt.show()

    print("\nALL ALGORITHMS COMPARISON:")
    print(f"{'Algorithm':<20} {'Avg Reward':<15} {'Cum Reward':<15} {'Cum Regret':<15}")
    print("-" * 70)
    print(
        f"{'Epsilon Greedy':<20} {eg_results['avg_reward']:<15.4f} {eg_results['cum_rewards'][-1]:<15.4f} {eg_results['cum_regrets'][-1]:<15.4f}")
    print(
        f"{'Thompson Sampling':<20} {ts_results['avg_reward']:<15.4f} {ts_results['cum_rewards'][-1]:<15.4f} {ts_results['cum_regrets'][-1]:<15.4f}")
    print(
        f"{'Bayesian UCB (Bonus)':<20} {bayesian_ucb_results['avg_reward']:<15.4f} {bayesian_ucb_results['cum_rewards'][-1]:<15.4f} {bayesian_ucb_results['cum_regrets'][-1]:<15.4f}")

    rewards = [
        (eg_results['cum_rewards'][-1], "Epsilon Greedy"),
        (ts_results['cum_rewards'][-1], "Thompson Sampling"),
        (bayesian_ucb_results['cum_rewards'][-1], "Bayesian UCB (Bonus)")
    ]
    best_algorithm = max(rewards, key=lambda x: x[0])
    print(f"\nBEST ALGORITHM: {best_algorithm[1]} with cumulative reward of {best_algorithm[0]:.4f}")


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("bandit_experiment.log", level="DEBUG", rotation="5 MB")

    bandit_rewards = [1, 2, 3, 4]
    n_trials = 20000

    total_reward = sum(bandit_rewards)
    probs = [r / total_reward for r in bandit_rewards]

    logger.info(f"Starting bandit experiment with rewards {bandit_rewards} and {n_trials} trials")

    eg_bandit = EpsilonGreedy(probs, epsilon=0.1)
    eg_results = eg_bandit.experiment(n_trials)
    eg_stats = eg_bandit.report()

    ts_bandit = ThompsonSampling(probs, alpha=1.0, beta=1.0)
    ts_results = ts_bandit.experiment(n_trials)
    ts_stats = ts_bandit.report()

    comparison(eg_results, ts_results)

    logger.info("Starting bonus implementation (Bayesian UCB algorithm)")

    bayesian_ucb_bandit = BayesianUCB(probs, alpha=2.0, beta=3.0, quantile=0.98, quantile_increase=0.005)
    bayesian_ucb_results = bayesian_ucb_bandit.experiment(n_trials)
    bayesian_ucb_stats = bayesian_ucb_bandit.report()

    compare_all_algorithms(eg_results, ts_results, bayesian_ucb_results)

    logger.info("All experiments completed successfully")