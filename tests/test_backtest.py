import pytest
import pandas as pd
import numpy as np
from xquant.backtest.engine.vectorized import VectorizedEngine


@pytest.fixture
def backtest_data():
    dates = pd.date_range(start="2023-01-01", periods=10)
    data = pd.DataFrame(
        {"close": [100, 101, 102, 101, 100, 102, 104, 106, 105, 107]}, index=dates
    )
    signals = pd.Series([0, 1, 1, 0, -1, -1, 0, 1, 1, 0], index=dates)
    return data, signals


def test_vectorized_engine(backtest_data):
    data, signals = backtest_data
    engine = VectorizedEngine(initial_capital=100000)
    results = engine.run(data, signals)

    assert "equity" in results
    assert "strategy_returns" in results
    assert results["equity"].iloc[0] == 100000
    assert not results["drawdown"].empty
    assert not results["equity"].isna().any()
