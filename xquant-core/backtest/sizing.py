import numpy as np
import pandas as pd
from typing import Optional


class PositionSizer:
    """
    Base class for position sizing logic.
    """

    def __init__(self, initial_capital: float):
        self.capital = initial_capital

    def fixed_fraction(self, price: float, fraction: float = 0.1) -> int:
        """
        Risk a fixed fraction of capital per trade.
        """
        return int((self.capital * fraction) / price)

    def kelly_criterion(
        self, win_rate: float, win_loss_ratio: float, price: float
    ) -> int:
        """
        Kelly Criterion: f* = p - (1-p)/b
        p: win rate, b: win/loss ratio
        """
        if win_loss_ratio <= 0:
            return 0
        kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
        kelly_f = max(0, kelly_f)  # No leverage or short for now
        return int((self.capital * kelly_f) / price)

    def volatility_targeting(
        self,
        price: float,
        target_vol: float,
        realized_vol: float,
        notional_cap: Optional[float] = None,
    ) -> int:
        """
        Size position to hit a specific volatility target.
        Weight = Target Vol / Realized Vol
        """
        if realized_vol <= 0:
            return 0

        weight = target_vol / realized_vol

        # Apply cap if provided
        if notional_cap:
            weight = min(weight, notional_cap / self.capital)

        return int((self.capital * weight) / price)

    def update_capital(self, current_capital: float):
        self.capital = current_capital
