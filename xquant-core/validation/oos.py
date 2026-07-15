import pandas as pd
from typing import Tuple, Dict, Any
from xquant.risk.metrics import get_performance_metrics


class OutOfSampleValidator:
    """
    Standard Out-of-Sample (OOS) validation tool.
    Splits data into In-Sample (IS) and Out-of-Sample (OOS) sets.
    """

    @staticmethod
    def split_data(
        data: pd.DataFrame, split_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        """
        split_idx = int(len(data) * split_ratio)
        return data.iloc[:split_idx], data.iloc[split_idx:]

    def validate(
        self, engine, strategy_func, data: pd.DataFrame, split_ratio: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform OOS validation and return IS and OOS performance metrics.
        """
        train_data, test_data = self.split_data(data, split_ratio)

        # In-Sample results
        is_signals = strategy_func(train_data)
        is_results = engine.run(train_data, is_signals)
        is_metrics = get_performance_metrics(is_results)

        # Out-of-Sample results
        oos_signals = strategy_func(test_data)
        oos_results = engine.run(test_data, oos_signals)
        oos_metrics = get_performance_metrics(oos_results)

        return {
            "in_sample_metrics": is_metrics,
            "out_of_sample_metrics": oos_metrics,
            "is_oos_ratio": (
                oos_metrics["Sharpe Ratio"] / is_metrics["Sharpe Ratio"]
                if is_metrics["Sharpe Ratio"] != 0
                else 0
            ),
        }
