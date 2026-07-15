from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class LiveBrokerAdapter(ABC):
    """
    Abstract base class for live broker adapters.
    Defines the interface for authentication, order management, and data fetching.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authenticated = False

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the broker API."""
        pass

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balances."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a new order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        pass

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Fetch all currently open orders."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a specific symbol."""
        pass
