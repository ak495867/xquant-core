import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Optional


class OverfittingValidator:
    """
    Tools to detect and adjust for backtest overfitting.
    Implements Deflated Sharpe Ratio (DSR) and related metrics.
    """

    @staticmethod
    def calculate_dsr(
        sharpe: float,
        trials_sharpes: List[float],
        n_observations: int,
        is_annualized: bool = True,
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio (DSR).

        Args:
            sharpe: The observed Sharpe ratio of the selected strategy.
            trials_sharpes: List of Sharpe ratios from all backtests performed.
            n_observations: Number of data points in the backtest.
            is_annualized: Whether the input sharpe ratios are annualized.

        Returns:
            The probability that the strategy is NOT overfit (DSR).
        """
        if not trials_sharpes:
            return 1.0

        n_trials = len(trials_sharpes)
        mean_sharpe = np.mean(trials_sharpes)
        std_sharpe = np.std(trials_sharpes)

        if std_sharpe == 0:
            return 0.0

        # Expected maximum Sharpe ratio under the null hypothesis (Bailey and Lopez de Prado)
        emc = 0.5772156649  # Euler-Mascheroni constant
        max_z = (1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(
            1 - 1 / (n_trials * np.e)
        )
        expected_max_sharpe = mean_sharpe + std_sharpe * max_z

        # DSR calculation
        # Note: This is a simplified version of the DSR formula
        variance_sharpe = (1 - sharpe**2) / (n_observations - 1)
        if variance_sharpe <= 0:
            return 0.0

        z_stat = (sharpe - expected_max_sharpe) / np.sqrt(variance_sharpe)
        return norm.cdf(z_stat)

    @staticmethod
    def bonferroni_adjustment(p_value: float, n_trials: int) -> float:
        """
        Apply Bonferroni correction for multiple testing.
        """
        return min(p_value * n_trials, 1.0)
