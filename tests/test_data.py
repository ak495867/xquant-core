import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from xquant.data.loader import DataLoader
from xquant.data.adapters.yfinance import YFinanceAdapter


@pytest.fixture
def mock_ohlcv_data():
    return pd.DataFrame(
        {
            "Date": pd.date_range(start="2023-01-01", periods=5),
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }
    ).set_index("Date")


def test_yfinance_adapter_fetch(mock_ohlcv_data):
    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = mock_ohlcv_data
        mock_instance.info = {"symbol": "AAPL"}

        adapter = YFinanceAdapter()
        df = adapter.fetch_ohlcv("AAPL", start_date="2023-01-01")

        assert not df.empty
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert len(df) == 5


def test_data_loader_single_symbol():
    mock_adapter = MagicMock()
    mock_adapter.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [1, 2, 3],
        }
    )

    loader = DataLoader(adapter=mock_adapter)
    data = loader.load("AAPL", start_date="2023-01-01")

    assert isinstance(data, pd.DataFrame)
    mock_adapter.fetch_ohlcv.assert_called_once()


def test_data_loader_multi_symbol():
    mock_adapter = MagicMock()
    mock_adapter.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [1, 2, 3],
        }
    )

    loader = DataLoader(adapter=mock_adapter)
    data = loader.load(["AAPL", "MSFT"], start_date="2023-01-01")

    assert isinstance(data, dict)
    assert "AAPL" in data
    assert "MSFT" in data
