from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseAdapter(ABC):
    """
    Abstract Base Class for data adapters.

    Adapters are responsible for fetching OHLCV data from specific sources
    and returning it in a standardized format.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and time range.

        Args:
            symbol: Asset symbol (e.g., "AAPL", "BTC/USDT").
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD). If None, defaults to current date.
            interval: Time interval (e.g., "1m", "5m", "1h", "1d").
            **kwargs: Additional provider-specific arguments.

        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        pass

    @abstractmethod
    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch metadata for a given symbol.
        """
        pass
