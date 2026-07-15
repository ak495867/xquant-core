import pandas as pd
import numpy as np
from typing import List, Dict, Any
from xquant.risk.metrics import calculate_sharpe_ratio


class MonteCarloSimulator:
    """
    Monte Carlo simulation tools for strategy robustness testing.
    """

    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def permutation_test(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Randomly shuffle returns to create a distribution of performance metrics.
        Helps determine if strategy performance is due to luck.
        """
        observed_sharpe = calculate_sharpe_ratio(returns)
        shuffled_sharpes = []

        returns_val = returns.values
        for _ in range(self.n_simulations):
            np.random.shuffle(returns_val)
            shuffled_sharpes.append(calculate_sharpe_ratio(pd.Series(returns_val)))

        p_value = np.mean(np.array(shuffled_sharpes) >= observed_sharpe)

        return {
            "observed_sharpe": observed_sharpe,
            "mean_shuffled_sharpe": np.mean(shuffled_sharpes),
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }

    def bootstrap_test(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Resample returns with replacement to generate a distribution of equity curves.
        Provides confidence intervals for performance metrics.
        """
        bootstrapped_metrics = []

        for _ in range(self.n_simulations):
            resampled_returns = np.random.choice(
                returns.values, size=len(returns), replace=True
            )
            resampled_series = pd.Series(resampled_returns)
            bootstrapped_metrics.append(calculate_sharpe_ratio(resampled_series))

        return {
            "mean_sharpe": np.mean(bootstrapped_metrics),
            "std_sharpe": np.std(bootstrapped_metrics),
            "ci_95": (
                np.percentile(bootstrapped_metrics, 2.5),
                np.percentile(bootstrapped_metrics, 97.5),
            ),
            "worst_case_sharpe": np.min(bootstrapped_metrics),
        }
