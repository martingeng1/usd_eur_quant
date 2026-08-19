from __future__ import annotations
import numpy as np
import pandas as pd


def add_trend_state(frame: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Causal trend state; rolling means only contain completed historical bars."""
    result = frame.copy()
    result["ma_fast"] = result.close.rolling(fast, min_periods=fast).mean()
    result["ma_slow"] = result.close.rolling(slow, min_periods=slow).mean()
    result["trend_20"] = result.close.pct_change(20)
    result["trend_state"] = np.select(
        [(result.close > result.ma_slow) & (result.ma_fast > result.ma_slow),
         (result.close < result.ma_slow) & (result.ma_fast < result.ma_slow)],
        ["up", "down"], default="range")
    return result
