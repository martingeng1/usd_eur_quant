import pandas as pd
from src.features.volatility import add_atr
from src.features.trend import add_trend_state
from src.research.statistics import benjamini_hochberg


def test_trend_is_causal_and_no_zero_atr_division() -> None:
    idx = pd.date_range("2024-01-01", periods=250, freq="h", tz="UTC")
    values = pd.Series(range(250), index=idx, dtype=float) + 2000
    frame = pd.DataFrame({"open": values, "high": values + 1, "low": values - 1, "close": values}, index=idx)
    result = add_trend_state(add_atr(frame))
    assert result.atr.iloc[-1] > 0
    assert result.trend_state.iloc[-1] == "up"


def test_benjamini_hochberg_is_monotone_in_rank_order() -> None:
    q = benjamini_hochberg(pd.Series([0.01, 0.04, 0.20]))
    assert q.iloc[0] <= q.iloc[1] <= q.iloc[2]
