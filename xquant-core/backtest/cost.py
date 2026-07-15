import numpy as np
from abc import ABC, abstractmethod


class CostModel(ABC):
    """
    Abstract base class for transaction cost modeling.
    """

    @abstractmethod
    def calculate_cost(self, quantity: float, price: float, **kwargs) -> float:
        pass


class CommissionModel(CostModel):
    """
    Standard commission models.
    """

    def __init__(self, pct: float = 0.001, flat: float = 0.0):
        self.pct = pct
        self.flat = flat

    def calculate_cost(self, quantity: float, price: float, **kwargs) -> float:
        return (abs(quantity) * price * self.pct) + self.flat


class SlippageModel(CostModel):
    """
    Standard slippage models.
    """

    def __init__(self, fixed_pct: float = 0.0001):
        self.fixed_pct = fixed_pct

    def calculate_cost(self, quantity: float, price: float, **kwargs) -> float:
        return abs(quantity) * price * self.fixed_pct


class MarketImpactSlippage(SlippageModel):
    """
    Slippage based on trade size relative to volume (Market Impact).
    Cost = price * sigma * (quantity / volume)^0.5
    """

    def __init__(self, sigma: float = 0.02, alpha: float = 0.1):
        self.sigma = sigma  # Daily volatility
        self.alpha = alpha  # Impact coefficient

    def calculate_cost(self, quantity: float, price: float, **kwargs) -> float:
        volume = kwargs.get("volume", 1.0)
        if volume <= 0:
            return 0.0

        impact = self.alpha * self.sigma * np.sqrt(abs(quantity) / volume)
        return abs(quantity) * price * impact


class TransactionCostEngine:
    """
    Combines commission and slippage models.
    """

    def __init__(self, commission: CostModel, slippage: CostModel):
        self.commission = commission
        self.slippage = slippage

    def get_total_cost(self, quantity: float, price: float, **kwargs) -> float:
        comm = self.commission.calculate_cost(quantity, price, **kwargs)
        slip = self.slippage.calculate_cost(quantity, price, **kwargs)
        return comm + slip
