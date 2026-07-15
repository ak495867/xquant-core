from xquant.data.loader import DataLoader
from xquant.features.technical import MovingAverage
from xquant.backtest.engine.vectorized import VectorizedEngine
from xquant.reporting.tearsheet import Tearsheet
import pandas as pd


def run_forex_carry():
    print("Initializing Forex Carry Strategy...")

    # 1. Load Data (USD/JPY as proxy for carry)
    loader = DataLoader()
    data = loader.load("JPY=X", start_date="2021-01-01", end_date="2023-12-31")

    if data.empty:
        print("Data load failed.")
        return

    # 2. Features: 200 SMA for trend filter
    sma = MovingAverage(window=200)
    data["sma_200"] = sma.compute(data)

    # 3. Signals: Simple trend following carry
    signals = pd.Series(0, index=data.index)
    signals[data["close"] > data["sma_200"]] = 1  # Long if above 200 SMA

    # 4. Backtest
    engine = VectorizedEngine(initial_capital=50000)
    results = engine.run(data, signals, commission=0.0001)  # Low commission for FX

    # 5. Reporting
    ts = Tearsheet(results)
    ts.summary()


if __name__ == "__main__":
    run_forex_carry()
