from __future__ import annotations
import numpy as np
import pandas as pd


def asian_breakout_events(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    above = result.close > result.asia_high
    below = result.close < result.asia_low
    direction = np.select([above, below], [1, -1], default=0)
    result["breakout_direction"] = direction
    # only first crossing of completed daily range, avoiding repeated same-day entries
    local_day = pd.Series(result.index.tz_convert("Asia/Tokyo").date, index=result.index)
    crossed = (pd.Series(direction, index=result.index) != 0) & (pd.Series(direction, index=result.index).shift(1).fillna(0) == 0)
    result["breakout_event"] = crossed & ~crossed.groupby(local_day).shift(fill_value=False).groupby(local_day).cummax()
    return result


def label_horizons(frame: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    result = frame.copy()
    for horizon in horizons:
        future = result.close.shift(-horizon) / result.close - 1
        result[f"forward_{horizon}h"] = future * result.breakout_direction
        high = result.high.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
        low = result.low.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
        result[f"mfe_{horizon}h"] = np.where(result.breakout_direction > 0, high / result.close - 1, result.close / low - 1)
        result[f"mae_{horizon}h"] = np.where(result.breakout_direction > 0, low / result.close - 1, result.close / high - 1)
    return result
