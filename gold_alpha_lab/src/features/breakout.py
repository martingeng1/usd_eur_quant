from __future__ import annotations
import pandas as pd
from src.data.timezone import in_local_window


def add_asian_range(frame: pd.DataFrame) -> pd.DataFrame:
    """Completed Tokyo 00:00-09:00 range, available only after the range has ended."""
    result = frame.copy()
    asia = in_local_window(result.index, "Asia/Tokyo", "00:00", "09:00").to_numpy()
    local = result.index.tz_convert("Asia/Tokyo")
    day = pd.Index(local.date, name="tokyo_date")
    temp = pd.DataFrame({"high": result.high, "low": result.low, "asia": asia}, index=result.index)
    highs = temp.high.where(temp.asia).groupby(day).max()
    lows = temp.low.where(temp.asia).groupby(day).min()
    mapping_high, mapping_low = pd.Series(day, index=result.index).map(highs), pd.Series(day, index=result.index).map(lows)
    completed = ~asia & (pd.Series(local.time, index=result.index) >= pd.Timestamp("09:00").time())
    result["asia_high"] = mapping_high.where(completed).to_numpy()
    result["asia_low"] = mapping_low.where(completed).to_numpy()
    result["asia_range"] = result.asia_high - result.asia_low
    return result
