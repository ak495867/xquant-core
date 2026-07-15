import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from .base import Factor


class BidAskSpread(Factor):
    """
    Computes the bid-ask spread. Requires 'bid' and 'ask' columns.
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        if "bid" in data.columns and "ask" in data.columns:
            return (data["ask"] - data["bid"]) / ((data["ask"] + data["bid"]) / 2)
        return pd.Series(index=data.index, data=np.nan)


class OrderFlowImbalance(Factor):
    """
    Simplified Order Flow Imbalance (OFI) based on price changes and volume.
    Ref: Cont et al. (2014)
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # Simplified OFI using Close price changes as a proxy for flow direction
        delta_p = data["close"].diff()
        ofi = np.sign(delta_p) * data["volume"]
        return ofi.rolling(window=self.params.get("window", 20)).sum()


class VWAPDeviation(Factor):
    """
    Deviation of the current price from the Volume Weighted Average Price (VWAP).
    """

    def compute(self, data: pd.DataFrame) -> pd.Series:
        window = self.params.get("window", 20)

        # Calculate VWAP
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        tp_v = typical_price * data["volume"]

        rolling_tp_v = tp_v.rolling(window=window).sum()
        rolling_v = data["volume"].rolling(window=window).sum()

        vwap = rolling_tp_v / rolling_v

        # Deviation in percentage or Z-score
        deviation = (data["close"] - vwap) / vwap

        if self.params.get("z_score", False):
            return (
                deviation - deviation.rolling(window=window).mean()
            ) / deviation.rolling(window=window).std()

        return deviation
