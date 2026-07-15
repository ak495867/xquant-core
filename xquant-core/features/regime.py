import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from .base import Factor

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class VolatilityRegime(Factor):
    """
    Detects volatility regimes using rolling standard deviation or GARCH.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        window = self.params.get("window", 20)
        returns = data["close"].pct_change().dropna()

        method = self.params.get("method", "rolling_std")

        if method == "garch" and arch_model:
            model = arch_model(returns, vol="Garch", p=1, q=1)
            res = model.fit(disp="off")
            return res.conditional_volatility
        else:
            # Default to rolling std normalized (Z-score of vol)
            vol = returns.rolling(window=window).std()
            vol_mean = vol.rolling(window=window * 5).mean()
            vol_std = vol.rolling(window=window * 5).std()
            return (vol - vol_mean) / vol_std


class TrendRegime(Factor):
    """
    Detects trend regimes using moving average crossovers or ADX proxy.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        fast_window = self.params.get("fast_window", 20)
        slow_window = self.params.get("slow_window", 50)

        fast_ma = data["close"].rolling(window=fast_window).mean()
        slow_ma = data["close"].rolling(window=slow_window).mean()

        # 1 for uptrend, -1 for downtrend, 0 for neutral
        regime = np.where(fast_ma > slow_ma, 1, -1)

        # Optional: Add a 'neutral' zone if the difference is small
        threshold = self.params.get("threshold", 0.001)  # 0.1%
        diff = (fast_ma - slow_ma) / slow_ma
        regime = np.where(abs(diff) < threshold, 0, regime)

        return pd.Series(regime, index=data.index)


class HMMRegime(Factor):
    """
    Placeholder for Hidden Markov Model based regime detection.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Requires hmmlearn package
        return pd.Series(index=data.index, data=np.nan)
