from xquant.data.loader import DataLoader
from xquant.features.technical import RSI, MovingAverage
from xquant.backtest.engine.vectorized import VectorizedEngine
from xquant.risk.metrics import get_performance_metrics
import pandas as pd


def run_momentum_strategy():
    print("Initializing XQuant Momentum Strategy...")

    # 1. Load Data
    loader = DataLoader()
    data = loader.load("AAPL", start_date="2020-01-01", end_date="2023-01-01")

    if data.empty:
        print("Failed to load data.")
        return

    # 2. Engineering Features
    rsi_factor = RSI(window=14)
    sma_factor = MovingAverage(window=50)

    data["rsi"] = rsi_factor.compute(data)
    data["sma"] = sma_factor.compute(data)

    # 3. Generate Signals
    # Simple logic: Buy if RSI < 40 (relatively oversold)
    signals = pd.Series(0, index=data.index)
    signals[data["rsi"] < 40] = 1
    signals[data["rsi"] > 60] = 0  # Exit if overbought

    print(f"Total bars: {len(data)}")
    print(f"Signal count (Long): {len(signals[signals == 1])}")

    # 4. Run Backtest
    engine = VectorizedEngine(initial_capital=100000)
    results = engine.run(data, signals)

    # 5. Performance Metrics
    metrics = get_performance_metrics(results)

    print("\n--- Strategy Performance ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nStrategy execution complete.")


if __name__ == "__main__":
    run_momentum_strategy()
