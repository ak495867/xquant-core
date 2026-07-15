import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from .base import Factor


class AssetCorrelation(Factor):
    """
    Rolling correlation between the current asset and a reference asset.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        window = self.params.get("window", 20)
        ref_data = self.params.get("reference_data")  # Should be a Series of returns

        if ref_data is None:
            return pd.Series(index=data.index, data=np.nan)

        returns = data["close"].pct_change()
        return returns.rolling(window=window).corr(ref_data)


class AssetBeta(Factor):
    """
    Rolling Beta (CAPM) of the current asset relative to a benchmark.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        window = self.params.get("window", 60)
        benchmark_returns = self.params.get("benchmark_returns")

        if benchmark_returns is None:
            return pd.Series(index=data.index, data=np.nan)

        returns = data["close"].pct_change()

        # Beta = Cov(rs, rb) / Var(rb)
        covariance = returns.rolling(window=window).cov(benchmark_returns)
        variance = benchmark_returns.rolling(window=window).var()

        return covariance / variance


class LeadLagFactor(Factor):
    """
    Computes lead-lag relationship using cross-correlation at different lags.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        max_lag = self.params.get("max_lag", 5)
        ref_returns = self.params.get("reference_returns")

        if ref_returns is None:
            return pd.Series(index=data.index, data=np.nan)

        returns = data["close"].pct_change()

        # Example: Correlation of current asset with lagged reference asset
        # Positive lag means ref_returns leads 'returns'
        results = {}
        for lag in range(1, max_lag + 1):
            results[f"lag_{lag}"] = returns.corr(ref_returns.shift(lag))

        # Return the lag with highest correlation as a factor
        return pd.Series(results).idxmax()  # This is a simplified example
