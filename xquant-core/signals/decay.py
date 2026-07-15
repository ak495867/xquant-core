import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from xquant.signals.scorer import SignalScorer


class SignalDecay:
    """
    Analyzes how signal efficacy (IC) decays over time.
    """

    def __init__(self, scorer: Optional[SignalScorer] = None):
        self.scorer = scorer or SignalScorer()

    def analyze_decay(
        self,
        signal: pd.Series,
        data: pd.DataFrame,
        horizons: List[int],
        return_prefix: str = "returns_",
    ) -> pd.Series:
        """
        Calculate IC across different forward horizons.

        Args:
            signal: The signal values.
            data: DataFrame containing forward returns for different horizons.
            horizons: List of horizons (e.g., [1, 2, 5, 10, 20]).
            return_prefix: Prefix for return columns in data (e.g., 'returns_5d').

        Returns:
            pd.Series: IC values indexed by horizon.
        """
        ic_decay = {}
        for h in horizons:
            ret_col = f"{return_prefix}{h}"
            if ret_col in data.columns:
                ic = self.scorer.calculate_rank_ic(signal, data[ret_col])
                ic_decay[h] = ic

        return pd.Series(ic_decay, name="IC_Decay")

    def calculate_half_life(self, ic_series: pd.Series) -> float:
        """
        Estimate signal half-life using exponential decay fit.
        IC(t) = IC(0) * exp(-lambda * t)
        Half-life = ln(2) / lambda
        """
        if len(ic_series) < 2:
            return 0.0

        # Linear regression on log(IC)
        # log(IC(t)) = log(IC(0)) - lambda * t
        y = np.log(ic_series.abs())
        x = ic_series.index.values

        # Remove NaNs and Infs
        mask = np.isfinite(y)
        if not any(mask):
            return 0.0

        x, y = x[mask], y[mask]

        # Simple linear fit
        coeffs = np.polyfit(x, y, 1)
        lam = -coeffs[0]

        if lam <= 0:
            return float("inf")

        return np.log(2) / lam
