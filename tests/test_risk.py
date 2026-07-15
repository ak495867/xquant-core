import pytest
import pandas as pd
import numpy as np
from xquant.risk.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)


def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252)
    assert isinstance(sharpe, float)
    assert sharpe > 0


def test_sortino_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0, periods=252)
    assert isinstance(sortino, float)


def test_max_drawdown():
    equity = pd.Series([100, 110, 105, 95, 100, 120])
    mdd = calculate_max_drawdown(equity)
    # Peak was 110, trough was 95. (95-110)/110 = -0.13636
    assert pytest.approx(mdd, 0.01) == -0.13636
