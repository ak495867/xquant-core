import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Any, Optional


class WalkForwardOptimization:
    """
    Implements Walk-Forward Optimization (WFO) for strategy validation.
    Supports both Rolling and Expanding window approaches.
    """

    def __init__(self, engine, strategy_func: Callable):
        """
        Initialize WFO.

        Args:
            engine: The backtesting engine (e.g., VectorizedEngine).
            strategy_func: A function that takes data and returns signals.
        """
        self.engine = engine
        self.strategy_func = strategy_func

    def run(
        self,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        anchor: str = "rolling",
    ) -> List[Dict[str, Any]]:
        """
        Execute the Walk-Forward Optimization.

        Args:
            data: Input DataFrame with price data.
            train_size: Number of periods for training.
            test_size: Number of periods for out-of-sample testing.
            anchor: 'rolling' for fixed window or 'expanding' for expanding training set.

        Returns:
            List[Dict]: Results for each walk-forward fold.
        """
        results = []
        n_periods = len(data)

        start_idx = 0
        while start_idx + train_size + test_size <= n_periods:
            train_start = 0 if anchor == "expanding" else start_idx
            train_end = start_idx + train_size
            test_start = train_end
            test_end = test_start + test_size

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            # Optimization/Training step (In practice, strategy_func might optimize params)
            # For this implementation, we assume strategy_func handles internal logic

            # Out-of-Sample test
            signals = self.strategy_func(test_data)
            fold_results = self.engine.run(test_data, signals)

            results.append(
                {
                    "fold": len(results) + 1,
                    "train_range": (data.index[train_start], data.index[train_end - 1]),
                    "test_range": (data.index[test_start], data.index[test_end - 1]),
                    "performance": fold_results,
                }
            )

            start_idx += test_size

        return results
