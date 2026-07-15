import queue
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Event:
    """
    Base class for all events in the event-driven backtester.
    """

    pass


class MarketEvent(Event):
    """
    Triggered when the DataHandler receives a new market update.
    """

    def __init__(self):
        self.type = "MARKET"


class SignalEvent(Event):
    """
    Triggered when a Strategy produces a signal.
    """

    def __init__(
        self, symbol: str, datetime: Any, signal_type: str, strength: float = 1.0
    ):
        self.type = "SIGNAL"
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type  # 'LONG', 'SHORT', 'EXIT'
        self.strength = strength


class OrderEvent(Event):
    """
    Triggered when the Portfolio sends an order to the ExecutionHandler.
    """

    def __init__(self, symbol: str, order_type: str, quantity: int, direction: str):
        self.type = "ORDER"
        self.symbol = symbol
        self.order_type = order_type  # 'MKT', 'LMT'
        self.quantity = quantity
        self.direction = direction  # 'BUY', 'SELL'


class FillEvent(Event):
    """
    Triggered when the ExecutionHandler fills an order.
    """

    def __init__(
        self,
        symbol: str,
        datetime: Any,
        quantity: int,
        direction: str,
        fill_cost: float,
        commission: float = None,
    ):
        self.type = "FILL"
        self.symbol = symbol
        self.datetime = datetime
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission


class DataHandler(ABC):
    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1):
        pass

    @abstractmethod
    def update_bars(self):
        pass


class Strategy(ABC):
    @abstractmethod
    def calculate_signals(self, event):
        pass


class Portfolio(ABC):
    @abstractmethod
    def update_signal(self, event):
        pass

    @abstractmethod
    def update_fill(self, event):
        pass

    @abstractmethod
    def update_market(self, event):
        pass


class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, event):
        pass


class EventDrivenEngine:
    """
    Main loop for the event-driven backtester.
    """

    def __init__(
        self,
        events: queue.Queue,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
    ):
        self.events = events
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler

    def run(self):
        """
        Executes the backtest.
        """
        while True:
            # Update bars
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break

            # Handle events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break

                if event is not None:
                    if event.type == "MARKET":
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_market(event)
                    elif event.type == "SIGNAL":
                        self.portfolio.update_signal(event)
                    elif event.type == "ORDER":
                        self.execution_handler.execute_order(event)
                    elif event.type == "FILL":
                        self.portfolio.update_fill(event)

            # Optional: Sleep for real-time simulation
            # time.sleep(self.heartbeat)
