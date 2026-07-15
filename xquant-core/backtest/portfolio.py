import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from xquant.backtest.engine.event import SignalEvent, OrderEvent, FillEvent


class PortfolioManager:
    """
    Manages positions, orders, and capital for a multi-asset portfolio.
    """

    def __init__(self, symbols: List[str], initial_capital: float = 100000.0):
        self.symbols = symbols
        self.initial_capital = initial_capital

        # Current state
        self.current_positions: Dict[str, int] = {s: 0 for s in symbols}
        self.current_holdings: Dict[str, float] = {
            s: 0.0 for s in symbols
        }  # Market value
        self.cash = initial_capital

        # History
        self.all_positions: List[Dict] = []
        self.all_holdings: List[Dict] = []

    def update_signal(self, event: SignalEvent) -> Optional[OrderEvent]:
        """
        Handle a signal event by generating an order.
        """
        # Simplistic implementation: translate signal to order
        # In a real system, this would use PositionSizer
        symbol = event.symbol
        direction = "BUY" if event.signal_type == "LONG" else "SELL"

        # Determine quantity (example: fixed 10 units for demo)
        quantity = 10

        if event.signal_type == "EXIT":
            cur_pos = self.current_positions[symbol]
            if cur_pos == 0:
                return None
            direction = "SELL" if cur_pos > 0 else "BUY"
            quantity = abs(cur_pos)

        return OrderEvent(symbol, "MKT", quantity, direction)

    def update_fill(self, event: FillEvent):
        """
        Update positions and cash based on a fill event.
        """
        symbol = event.symbol
        fill_dir = 1 if event.direction == "BUY" else -1

        # Update positions
        self.current_positions[symbol] += fill_dir * event.quantity

        # Update cash
        cost = fill_dir * event.fill_cost * event.quantity
        commission = event.commission or 0.0
        self.cash -= cost + commission

    def update_market(self, symbol: str, price: float):
        """
        Update market value of holdings based on new price.
        """
        self.current_holdings[symbol] = self.current_positions[symbol] * price

    def get_total_equity(self) -> float:
        """
        Calculate total portfolio equity (cash + holdings).
        """
        return self.cash + sum(self.current_holdings.values())

    def create_equity_curve(self) -> pd.DataFrame:
        """
        Generate equity curve DataFrame from history.
        """
        # This requires historical snapshots to be recorded during backtest
        return pd.DataFrame(self.all_holdings)
