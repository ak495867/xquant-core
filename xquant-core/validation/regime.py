import pandas as pd
import numpy as np
from typing import Dict, Any
from xquant.risk.metrics import get_performance_metrics


class RegimeValidator:
    """
    Analyzes strategy performance conditioned on market regimes.
    """

    def analyze(
        self, results: pd.DataFrame, regimes: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics for each identified market regime.

        Args:
            results: Backtest results DataFrame.
            regimes: Series containing regime labels, indexed the same as results.

        Returns:
            Dict mapping regime names to their performance metrics.
        """
        combined = results.join(regimes.rename("regime"))
        regime_performance = {}

        for regime_name, group in combined.groupby("regime"):
            if len(group) < 2:
                continue
            metrics = get_performance_metrics(group)
            regime_performance[str(regime_name)] = metrics

        return regime_performance

    @staticmethod
    def detect_simple_regimes(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Helper to detect simple trend-based regimes.
        """
        sma = data["close"].rolling(window).mean()
        regime = np.where(data["close"] > sma, "Bull", "Bear")
        return pd.Series(regime, index=data.index)
