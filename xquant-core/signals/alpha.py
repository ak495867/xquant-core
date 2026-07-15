import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional
from xquant.signals.scorer import SignalScorer


class AlphaEngine:
    """
    Alpha Discovery engine for backtesting and evaluating multiple factors.
    """

    def __init__(self):
        self.factors: Dict[str, Callable] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.scorer = SignalScorer()

    def register_factor(self, name: str, func: Callable):
        """
        Register a new alpha factor function.
        """
        self.factors[name] = func

    def run_discovery(
        self,
        data: pd.DataFrame,
        forward_return_col: str = "forward_returns",
        method: str = "spearman",
    ) -> pd.DataFrame:
        """
        Execute all registered factors and evaluate their performance.

        Args:
            data: Input DataFrame containing features and returns.
            forward_return_col: Column name for forward returns.
            method: IC calculation method.

        Returns:
            pd.DataFrame: Performance summary for all factors.
        """
        summary_list = []

        for name, factor_func in self.factors.items():
            try:
                # Compute factor
                signal = factor_func(data)

                # Align signal and forward returns
                # Assuming data is a flat DataFrame (Tidy format)
                # If it's multi-index (date, symbol), we need to handle it.

                ic = self.scorer.calculate_ic(signal, data[forward_return_col])
                rank_ic = self.scorer.calculate_rank_ic(
                    signal, data[forward_return_col]
                )

                # Store results
                metrics = {
                    "factor": name,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "ic_std": 0.0,  # Placeholder if not using time-series IC here
                    "icir": 0.0,  # Placeholder
                }

                # If we can pivot to time-series
                if isinstance(data.index, pd.MultiIndex):
                    # Pivot for ICIR calculation
                    sig_pivot = signal.unstack()
                    ret_pivot = data[forward_return_col].unstack()
                    metrics["icir"] = self.scorer.calculate_icir(
                        sig_pivot, ret_pivot, method=method
                    )

                self.results[name] = metrics
                summary_list.append(metrics)

            except Exception as e:
                print(f"Error evaluating factor {name}: {e}")

        return pd.DataFrame(summary_list).sort_values(by="rank_ic", ascending=False)

    def get_factor_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for all registered factors.
        """
        signals = pd.DataFrame(index=data.index)
        for name, factor_func in self.factors.items():
            signals[name] = factor_func(data)
        return signals
