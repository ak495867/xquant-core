import pytest
import pandas as pd
import numpy as np
from xquant.features.technical import RSI, MovingAverage, MACD


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "close": [10, 11, 12, 11, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10],
            "high": [11, 12, 13, 12, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11],
            "low": [9, 10, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10, 9],
        }
    )


def test_moving_average(sample_data):
    sma = MovingAverage(window=3)
    result = sma.compute(sample_data)
    expected = sample_data["close"].rolling(window=3).mean()
    pd.testing.assert_series_equal(result, expected)


def test_rsi(sample_data):
    rsi = RSI(window=2)
    result = rsi.compute(sample_data)
    assert len(result) == len(sample_data)
    assert not result.dropna().empty


def test_macd(sample_data):
    macd = MACD(window_slow=5, window_fast=2, window_sign=2)
    result = macd.compute(sample_data)
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
