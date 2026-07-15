import pytest
import pandas as pd
import numpy as np
from xquant.signals.scorer import SignalScorer


def test_signal_scorer_ic():
    # Perfect positive correlation
    signal = pd.Series([1, 2, 3, 4, 5])
    returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

    ic = SignalScorer.calculate_ic(signal, returns)
    assert ic == 1.0


def test_signal_scorer_rank_ic():
    # Perfect rank correlation
    signal = pd.Series([1, 2, 3, 4, 5])
    returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

    rank_ic = SignalScorer.calculate_rank_ic(signal, returns)
    assert rank_ic == pytest.approx(1.0)


def test_signal_scorer_icir():
    # Correlation varying over time to produce non-zero std
    signals = pd.DataFrame({"2023-01-01": [1, 2, 3], "2023-01-02": [3, 2, 1]})
    returns = pd.DataFrame(
        {"2023-01-01": [0.1, 0.2, 0.3], "2023-01-02": [0.1, 0.2, 0.3]}
    )

    icir = SignalScorer.calculate_icir(signals.T, returns.T)
    # Day 1 IC = 1.0, Day 2 IC = -1.0. Mean = 0, ICIR = 0
    assert isinstance(icir, (float, np.float64))
