import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression


class ExposureAnalyzer:
    """
    Analyzes strategy exposure to risk factors and asset concentration.
    """

    def calculate_factor_exposure(
        self, strategy_returns: pd.Series, factor_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate Beta exposure to provided factors using linear regression.

        Args:
            strategy_returns: Series of strategy returns.
            factor_returns: DataFrame of factor returns (e.g., Mkt-RF, SMB, HML).

        Returns:
            Dict of factor names and their corresponding Beta coefficients.
        """
        # Align data
        combined = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
        if combined.empty:
            return {}

        y = combined.iloc[:, 0].values
        X = combined.iloc[:, 1:].values

        model = LinearRegression()
        model.fit(X, y)

        exposures = {
            factor: coef for factor, coef in zip(factor_returns.columns, model.coef_)
        }
        exposures["Alpha"] = model.intercept_
        exposures["R-Squared"] = model.score(X, y)

        return exposures

    def calculate_concentration(self, weights: pd.Series) -> Dict[str, float]:
        """
        Calculate portfolio concentration metrics.
        """
        active_weights = weights[weights != 0]
        if active_weights.empty:
            return {"Herfindahl-Hirschman Index": 0.0, "Effective N": 0.0}

        hhi = (active_weights**2).sum()
        effective_n = 1 / hhi if hhi > 0 else 0

        return {
            "Herfindahl-Hirschman Index": hhi,
            "Effective N": effective_n,
            "Max Weight": active_weights.abs().max(),
            "Top 5 Concentration": active_weights.abs().nlargest(5).sum(),
        }
