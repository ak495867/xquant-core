from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional


class Factor(ABC):
    """
    Abstract Base Class for all features/factors.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the factor given a standardized OHLCV DataFrame.
        """
        pass

    def __repr__(self):
        return f"Factor(name={self.name}, params={self.params})"
