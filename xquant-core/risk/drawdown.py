import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DrawdownAnalyzer:
    """
    Advanced drawdown analysis for risk management.
    """

    def calculate_drawdown_details(self, equity: pd.Series) -> pd.DataFrame:
        """
        Calculate detailed drawdown periods including start, peak, end, and duration.
        """
        max_equity = equity.cummax()
        drawdowns = (equity - max_equity) / max_equity

        is_in_drawdown = drawdowns < 0
        drawdown_periods = []

        if not is_in_drawdown.any():
            return pd.DataFrame()

        # Group consecutive drawdown periods
        df = pd.DataFrame({"drawdown": drawdowns, "is_in_dd": is_in_drawdown})
        df["group"] = (df["is_in_dd"] != df["is_in_dd"].shift()).cumsum()

        for name, group in df[df["is_in_dd"]].groupby("group"):
            start = group.index[0]
            end = group.index[-1]
            peak_idx = group["drawdown"].idxmin()

            drawdown_periods.append(
                {
                    "Start": start,
                    "Peak": peak_idx,
                    "End": end,
                    "Depth": group["drawdown"].min(),
                    "Duration": len(group),
                }
            )

        return pd.DataFrame(drawdown_periods).sort_values(by="Depth")

    def get_underwater_data(self, equity: pd.Series) -> pd.Series:
        """
        Return the underwater curve (percentage drawdown over time).
        """
        max_equity = equity.cummax()
        return (equity - max_equity) / max_equity

    def stress_test_drawdowns(
        self, returns: pd.Series, scenarios: List[float]
    ) -> Dict[str, float]:
        """
        Apply stress scenarios to returns and calculate hypothetical drawdowns.
        """
        results = {}
        for shock in scenarios:
            shocked_returns = returns.copy()
            # Apply shock to the 5 most volatile days
            top_vol_idx = returns.abs().nlargest(5).index
            shocked_returns.loc[top_vol_idx] *= 1 + shock

            equity = (1 + shocked_returns).cumprod()
            max_dd = (equity - equity.cummax()).min()
            results[f"Shock {shock*100}%"] = max_dd

        return results
