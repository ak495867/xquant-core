import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """
    Calculate the annualized Sharpe Ratio.
    """
    excess_returns = returns - (risk_free_rate / periods)
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods) * (excess_returns.mean() / excess_returns.std())


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """
    Calculate the annualized Sortino Ratio.
    """
    excess_returns = returns - (risk_free_rate / periods)
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.std() == 0:
        return 0.0
    return np.sqrt(periods) * (excess_returns.mean() / downside_returns.std())


def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    Calculate the maximum peak-to-trough drawdown.
    """
    max_equity = equity.cummax()
    drawdown = (equity - max_equity) / max_equity
    return drawdown.min()


def get_performance_metrics(results: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a dictionary of institutional performance metrics.
    """
    returns = results["strategy_returns"]
    equity = results["equity"]

    metrics = {
        "Total Return": (equity.iloc[-1] / equity.iloc[0]) - 1,
        "Annualized Return": (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity))
        - 1,
        "Annualized Volatility": returns.std() * np.sqrt(252),
        "Sharpe Ratio": calculate_sharpe_ratio(returns),
        "Sortino Ratio": calculate_sortino_ratio(returns),
        "Max Drawdown": calculate_max_drawdown(equity),
        "Win Rate": (
            len(returns[returns > 0]) / len(returns[returns != 0])
            if len(returns[returns != 0]) > 0
            else 0
        ),
    }
    return metrics
