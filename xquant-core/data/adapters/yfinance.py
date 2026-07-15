import yfinance as yf
import pandas as pd
from typing import Any, Dict, Optional
from .base import BaseAdapter


class YFinanceAdapter(BaseAdapter):
    """
    Adapter for fetching equity data from Yahoo Finance.
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using yfinance.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval, **kwargs)

        if df.empty:
            return pd.DataFrame()

        # Standardize columns
        df = df.reset_index()
        column_map = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=column_map)

        # Ensure only standard columns are returned
        standard_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return df[standard_cols]

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch symbol info using yfinance.
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
