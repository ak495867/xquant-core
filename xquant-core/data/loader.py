import pandas as pd
from typing import Any, Dict, List, Optional, Union
from .adapters.base import BaseAdapter
from .adapters.yfinance import YFinanceAdapter


class DataLoader:
    """
    Main entry point for loading multi-asset OHLCV data.
    """

    def __init__(self, adapter: Optional[BaseAdapter] = None):
        self.adapter = adapter or YFinanceAdapter()

    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data for one or more symbols.
        """
        if isinstance(symbols, str):
            return self.adapter.fetch_ohlcv(
                symbols, start_date, end_date, interval, **kwargs
            )

        data = {}
        for symbol in symbols:
            df = self.adapter.fetch_ohlcv(
                symbol, start_date, end_date, interval, **kwargs
            )
            if not df.empty:
                data[symbol] = df
        return data

    def set_adapter(self, adapter: BaseAdapter):
        """
        Switch the data source adapter.
        """
        self.adapter = adapter
