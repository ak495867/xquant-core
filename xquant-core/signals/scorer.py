import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Any, Optional


class SignalScorer:
    """
    Evaluates signal performance using Information Coefficient (IC) metrics.
    """

    @staticmethod
    def calculate_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
        """
        Calculate Pearson Information Coefficient.
        """
        return signal.corr(forward_returns)

    @staticmethod
    def calculate_rank_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
        """
        Calculate Spearman Rank Information Coefficient.
        """
        # Spearman correlation is Pearson correlation between ranks
        return signal.corr(forward_returns, method="spearman")

    @staticmethod
    def calculate_ic_series(
        signals: pd.DataFrame, returns: pd.DataFrame, method: str = "spearman"
    ) -> pd.Series:
        """
        Calculate IC for each time period (cross-sectional).

        Args:
            signals: DataFrame with symbols as columns and timestamps as index.
            returns: DataFrame with symbols as columns and forward returns as index.
            method: 'pearson' or 'spearman'.
        """
        ic_series = signals.corrwith(returns, axis=1, method=method)
        return ic_series

    @classmethod
    def calculate_icir(
        cls, signals: pd.DataFrame, returns: pd.DataFrame, method: str = "spearman"
    ) -> float:
        """
        Calculate Information Coefficient Information Ratio (ICIR).
        """
        ic_series = cls.calculate_ic_series(signals, returns, method)
        if ic_series.std() == 0:
            return 0.0
        return ic_series.mean() / ic_series.std()

    @classmethod
    def get_summary_stats(
        cls, signal: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate summary of IC metrics.
        """
        return {
            "IC": cls.calculate_ic(signal, forward_returns),
            "Rank IC": cls.calculate_rank_ic(signal, forward_returns),
        }
