from __future__ import annotations
import numpy as np
import pandas as pd


def fake_breakout_events(frame: pd.DataFrame, atr_multiple: float = 0.05) -> pd.DataFrame:
    """A range break that closes back inside on the next completed bar."""
    result = frame.copy()
    threshold = result.atr.fillna(np.inf) * atr_multiple
    up = (result.high > result.asia_high + threshold) & (result.close <= result.asia_high)
    down = (result.low < result.asia_low - threshold) & (result.close >= result.asia_low)
    result["fake_break_direction"] = np.select([up, down], [-1, 1], default=0)
    result["fake_break_event"] = result.fake_break_direction != 0
    return result
