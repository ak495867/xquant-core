import pandas as pd
import time
from typing import Dict, Any, Optional, List
from xquant.execution.live import LiveBrokerAdapter


class PaperTradingEngine(LiveBrokerAdapter):
    """
    Simulates live execution (Paper Trading) against real-time data feeds.
    """

    def __init__(self, config: Dict[str, Any], initial_balance: float = 100000.0):
        super().__init__(config)
        self.balance = {"USD": initial_balance}
        self.positions = {}
        self.orders = []
        self.order_id_counter = 0
        self.authenticated = True  # Always True for paper trading

    def authenticate(self) -> bool:
        return True

    def get_balance(self) -> Dict[str, float]:
        return self.balance

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simulate order placement and immediate fill if possible.
        """
        self.order_id_counter += 1
        order_id = str(self.order_id_counter)

        # In a real paper engine, we'd wait for the next price update to fill
        # Here we simulate an immediate fill at the 'price' provided or a hypothetical one
        fill_price = price if price else 0.0  # Placeholder

        order = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": fill_price,
            "status": "filled",
            "timestamp": time.time(),
        }

        # Update positions and balance (simplified)
        cost = amount * fill_price
        if side == "buy":
            self.balance["USD"] -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + amount
        else:
            self.balance["USD"] += cost
            self.positions[symbol] = self.positions.get(symbol, 0) - amount

        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        for order in self.orders:
            if order["id"] == order_id and order["status"] == "open":
                order["status"] = "cancelled"
                return True
        return False

    def get_open_orders(self, symbol: Optional[str] = None) -> pd.DataFrame:
        open_orders = [o for o in self.orders if o["status"] == "open"]
        if symbol:
            open_orders = [o for o in open_orders if o["symbol"] == symbol]
        return pd.DataFrame(open_orders)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "amount": self.positions.get(symbol, 0.0)}
