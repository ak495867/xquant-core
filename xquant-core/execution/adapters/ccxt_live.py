import ccxt
import pandas as pd
from typing import Dict, Any, Optional
from xquant.execution.live import LiveBrokerAdapter


class CCXTLiveAdapter(LiveBrokerAdapter):
    """
    Live execution adapter using the CCXT library for cryptocurrency exchanges.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange_id = config.get("exchange_id")
        self.api_key = config.get("api_key")
        self.secret = config.get("secret")
        self.params = config.get("params", {})

        exchange_class = getattr(ccxt, self.exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": self.api_key,
                "secret": self.secret,
                "enableRateLimit": True,
                **self.params,
            }
        )

    def authenticate(self) -> bool:
        try:
            self.exchange.check_required_credentials()
            self.authenticated = True
            return True
        except Exception as e:
            print(f"Authentication failed for {self.exchange_id}: {e}")
            return False

    def get_balance(self) -> Dict[str, float]:
        balance = self.exchange.fetch_balance()
        return {
            asset: data["free"]
            for asset, data in balance["total"].items()
            if data["free"] > 0
        }

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        if order_type.lower() == "market":
            return self.exchange.create_order(symbol, "market", side, amount)
        else:
            return self.exchange.create_order(symbol, "limit", side, amount, price)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id)
            return True
        except Exception:
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> pd.DataFrame:
        orders = self.exchange.fetch_open_orders(symbol)
        return pd.DataFrame(orders)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        # Most crypto spot exchanges don't have 'positions' in the traditional sense,
        # but futures/perpetuals do.
        if hasattr(self.exchange, "fetch_position"):
            return self.exchange.fetch_position(symbol)
        else:
            # Fallback for spot: get balance of the base currency
            base = symbol.split("/")[0]
            balance = self.exchange.fetch_balance()
            return {"symbol": symbol, "amount": balance["total"].get(base, 0.0)}
