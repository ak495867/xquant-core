import pandas as pd
import numpy as np
from typing import List, Optional, Union
from scipy.optimize import minimize


class SignalCombiner:
    """
    Combines multiple alpha signals into a single composite signal.
    """

    def equal_weight(self, signals: pd.DataFrame) -> pd.Series:
        """
        Combine signals using equal weights.
        """
        return signals.mean(axis=1)

    def volatility_weighted(self, signals: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Combine signals using inverse volatility weighting.
        """
        vols = signals.rolling(window=window).std()
        inv_vols = 1.0 / vols
        weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)
        return (signals * weights).sum(axis=1)

    def custom_weighted(
        self, signals: pd.DataFrame, weights: Union[List[float], np.ndarray]
    ) -> pd.Series:
        """
        Combine signals using user-provided weights.
        """
        if len(weights) != signals.shape[1]:
            raise ValueError("Length of weights must match number of signals.")
        return signals.dot(weights)

    def optimize_ic(self, signals: pd.DataFrame, returns: pd.Series) -> pd.Series:
        """
        Combine signals by optimizing for maximum Information Coefficient.
        """
        n = signals.shape[1]

        def objective(w):
            combined = signals.dot(w)
            return -combined.corr(returns)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0, 1) for _ in range(n)]
        init_guess = np.array([1 / n] * n)

        res = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not res.success:
            print("Optimization failed, returning equal weights.")
            return self.equal_weight(signals)

        return signals.dot(res.x)
