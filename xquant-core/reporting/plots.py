import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve"):
    """
    Plot the equity curve over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label="Portfolio Equity", color="blue")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_drawdown(drawdown: pd.Series, title: str = "Drawdown Over Time"):
    """
    Plot the drawdown series.
    """
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    plt.plot(drawdown, color="red", linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.show()


def plot_rolling_sharpe(returns: pd.Series, window: int = 126):
    """
    Plot the rolling Sharpe ratio.
    """
    rolling_returns = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = (rolling_returns / rolling_std) * np.sqrt(252)

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe, label=f"Rolling Sharpe ({window}d)", color="green")
    plt.title("Rolling Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()
