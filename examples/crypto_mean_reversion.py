from xquant.data.adapters.ccxt import CCXTAdapter
from xquant.data.loader import DataLoader
from xquant.features.technical import RSI, BollingerBands
from xquant.backtest.engine.vectorized import VectorizedEngine
from xquant.reporting.tearsheet import Tearsheet
import pandas as pd


def run_crypto_strategy():
    print("Initializing Crypto Mean Reversion Strategy...")

    # 1. Load Data (Simulated with yfinance for example convenience)
    loader = DataLoader()  # Defaults to yfinance
    data = loader.load("BTC-USD", start_date="2022-01-01", end_date="2023-12-31")

    if data.empty:
        print("Data load failed.")
        return

    # 2. Features
    bb = BollingerBands(window=20, window_dev=2)
    rsi = RSI(window=14)

    data["bb_width"] = bb.compute(data)
    data["rsi"] = rsi.compute(data)

    # 3. Signals: Mean Reversion
    # Buy if RSI < 30 (Oversold), Sell if RSI > 70 (Overbought)
    signals = pd.Series(0, index=data.index)
    signals[data["rsi"] < 30] = 1
    signals[data["rsi"] > 70] = -1

    # 4. Backtest
    engine = VectorizedEngine(initial_capital=10000)
    results = engine.run(data, signals, commission=0.001)

    # 5. Reporting
    ts = Tearsheet(results)
    ts.summary()


if __name__ == "__main__":
    run_crypto_strategy()
