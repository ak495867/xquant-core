import ccxt
import pandas as pd
from typing import Any, Dict, Optional, List
from .base import BaseAdapter
import datetime


class CCXTAdapter(BaseAdapter):
    """
    Adapter for fetching crypto data from various exchanges using CCXT.
    """

    def __init__(
        self, exchange_id: str = "binance", config: Optional[Dict[str, Any]] = None
    ):
        self.exchange_id = exchange_id
        self.config = config or {}
        self.exchange = getattr(ccxt, exchange_id)(self.config)

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data using CCXT.

        Note: CCXT uses milliseconds for timestamps and specific interval strings.
        """
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")

        all_ohlcv = []
        limit = 1000

        target_timestamp = None
        if end_date:
            target_timestamp = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        else:
            target_timestamp = self.exchange.milliseconds()

        while since < target_timestamp:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, timeframe=interval, since=since, limit=limit, params=kwargs
            )
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1

            # Prevent infinite loops and respect rate limits
            if len(ohlcv) < limit:
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        if end_date:
            df = df[
                df["timestamp"]
                <= pd.to_datetime(end_date).tz_localize(df["timestamp"].dt.tz)
            ]

        return df

    def get_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch market metadata using CCXT.
        """
        markets = self.exchange.load_markets()
        return markets.get(symbol, {})
