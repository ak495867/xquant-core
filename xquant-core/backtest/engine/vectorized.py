import pandas as pd
import numpy as np
from typing import Any, Dict, Optional


class VectorizedEngine:
    """
    High-speed vectorized backtesting engine.
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        commission: float = 0.001,
        slippage: float = 0.0,
    ) -> pd.DataFrame:
        """
        Run backtest on a single asset.

        Args:
            data: Standardized OHLCV DataFrame.
            signals: Series of target positions [-1, 0, 1].
            commission: Fixed percentage commission per trade.
            slippage: Fixed percentage slippage per trade.

        Returns:
            pd.DataFrame: Backtest results (returns, equity curve, drawdowns).
        """
        results = data.copy()
        results["signal"] = signals.shift(1).fillna(0)  # Execute on next open

        # Calculate log returns
        results["market_returns"] = np.log(results["close"] / results["close"].shift(1))

        # Strategy returns
        results["strategy_returns"] = (
            results["market_returns"] * results["signal"]
        ).fillna(0)

        # Modeling costs (simplified vectorized approach)
        trades = results["signal"].diff().abs().fillna(0)
        costs = trades * (commission + slippage)
        results["strategy_returns"] -= costs

        # Equity curve
        results["cumulative_returns"] = (
            results["strategy_returns"].cumsum().apply(np.exp)
        )
        results["equity"] = results["cumulative_returns"] * self.initial_capital

        # Drawdown
        results["max_equity"] = results["equity"].cummax()
        results["drawdown"] = (results["equity"] - results["max_equity"]) / results[
            "max_equity"
        ]

        return results
