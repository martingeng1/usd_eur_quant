from __future__ import annotations
from datetime import time
from zoneinfo import ZoneInfo
import pandas as pd


def in_local_window(index: pd.DatetimeIndex, timezone: str, start: str, end: str) -> pd.Series:
    """Timezone-aware [start, end) session membership; DST follows IANA rules."""
    local = index.tz_convert(ZoneInfo(timezone))
    start_t, end_t = time.fromisoformat(start), time.fromisoformat(end)
    clock = pd.Series(local.time, index=index)
    if start_t <= end_t:
        return (clock >= start_t) & (clock < end_t)
    return (clock >= start_t) | (clock < end_t)


def session_date(index: pd.DatetimeIndex, timezone: str, rollover: str = "17:00") -> pd.DatetimeIndex:
    local = index.tz_convert(ZoneInfo(timezone))
    roll = time.fromisoformat(rollover)
    dates = pd.DatetimeIndex(local.date)
    return dates.where(pd.Series(local.time, index=index).to_numpy() >= roll, dates - pd.Timedelta(days=1))
