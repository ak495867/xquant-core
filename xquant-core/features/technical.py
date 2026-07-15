import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from .base import Factor
import ta


class RSI(Factor):
    """
    Relative Strength Index (RSI).
    """

    def __init__(self, window: int = 14):
        super().__init__(name=f"RSI_{window}", params={"window": window})

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ta.momentum.rsi(data["close"], window=self.params["window"])


class MACD(Factor):
    """
    Moving Average Convergence Divergence (MACD).
    """

    def __init__(
        self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9
    ):
        super().__init__(
            name="MACD",
            params={"slow": window_slow, "fast": window_fast, "sign": window_sign},
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ta.trend.macd(
            data["close"],
            window_slow=self.params["slow"],
            window_fast=self.params["fast"],
        )


class BollingerBands(Factor):
    """
    Bollinger Bands (returns the width/percentage).
    """

    def __init__(self, window: int = 20, window_dev: int = 2):
        super().__init__(name="BB_Width", params={"window": window, "dev": window_dev})

    def compute(self, data: pd.DataFrame) -> pd.Series:
        indicator = ta.volatility.BollingerBands(
            data["close"], window=self.params["window"], window_dev=self.params["dev"]
        )
        return indicator.bollinger_wband()


class ATR(Factor):
    """
    Average True Range (ATR).
    """

    def __init__(self, window: int = 14):
        super().__init__(name=f"ATR_{window}", params={"window": window})

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ta.volatility.average_true_range(
            data["high"], data["low"], data["close"], window=self.params["window"]
        )


class MovingAverage(Factor):
    """
    Simple Moving Average (SMA).
    """

    def __init__(self, window: int = 50):
        super().__init__(name=f"SMA_{window}", params={"window": window})

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["close"].rolling(window=self.params["window"]).mean()
