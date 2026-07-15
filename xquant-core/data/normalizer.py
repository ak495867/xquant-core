import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class DataNormalizer:
    """
    Normalizes and aligns multi-asset time series data.
    """

    @staticmethod
    def align_assets(
        data: Dict[str, pd.DataFrame], method: str = "outer"
    ) -> pd.DataFrame:
        """
        Align multiple asset DataFrames into a single multi-index DataFrame or a wide DataFrame.

        Args:
            data: Dictionary mapping symbol to its OHLCV DataFrame.
            method: Join method ('inner', 'outer', 'left', 'right').

        Returns:
            pd.DataFrame: Aligned DataFrame with MultiIndex columns (Asset, Metric).
        """
        if not data:
            return pd.DataFrame()

        # Ensure all timestamps are set as index
        processed_data = {}
        for symbol, df in data.items():
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            processed_data[symbol] = df

        # Combine using concat
        combined = pd.concat(
            processed_data.values(), axis=1, keys=processed_data.keys(), join=method
        )
        combined.index.name = "timestamp"
        return combined

    @staticmethod
    def handle_timezones(df: pd.DataFrame, target_tz: str = "UTC") -> pd.DataFrame:
        """
        Ensure all timestamps are in the same timezone.
        """
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(target_tz)
        else:
            df.index = df.index.tz_convert(target_tz)
        return df

    @staticmethod
    def adjust_for_splits_dividends(
        df: pd.DataFrame, adj_factor_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Adjust OHLC data for corporate actions if adjustment factors are provided.
        """
        if adj_factor_col and adj_factor_col in df.columns:
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col] * df[adj_factor_col]
        return df

    @staticmethod
    def resample_data(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a new frequency.
        """
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Only aggregate columns that exist
        available_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

        return df.resample(rule).agg(available_agg).dropna()
