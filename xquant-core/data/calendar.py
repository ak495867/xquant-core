import pandas as pd
import numpy as np
from typing import List, Optional, Union

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None


class TradingCalendar:
    """
    Handles trading days, market holidays, and resampling logic.
    """

    def __init__(self, exchange: str = "NYSE"):
        self.exchange_name = exchange
        if mcal:
            self.calendar = mcal.get_calendar(exchange)
        else:
            self.calendar = None

    def get_trading_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Get a list of valid trading days for the exchange.
        """
        if self.calendar:
            schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)
            return schedule.index
        else:
            # Fallback to simple business days
            return pd.bdate_range(start=start_date, end=end_date)

    def is_trading_day(self, date: Union[str, pd.Timestamp]) -> bool:
        """
        Check if a specific date is a trading day.
        """
        days = self.get_trading_days(date, date)
        return len(days) > 0

    def get_market_hours(self, date: Union[str, pd.Timestamp]) -> Optional[dict]:
        """
        Get market open and close times for a specific date.
        """
        if self.calendar:
            schedule = self.calendar.schedule(start_date=date, end_date=date)
            if not schedule.empty:
                return {
                    "market_open": schedule.iloc[0]["market_open"],
                    "market_close": schedule.iloc[0]["market_close"],
                }
        return None

    @staticmethod
    def resample_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to business days, handling missing values.
        """
        return df.resample("B").ffill()
