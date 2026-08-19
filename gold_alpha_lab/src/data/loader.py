from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"open", "high", "low", "close"}


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load real OHLCV, normalize timestamps to UTC, and preserve no fabricated bars."""
    frame = pd.read_csv(path)
    timestamp = "datetime" if "datetime" in frame.columns else "timestamp"
    if timestamp not in frame.columns:
        raise ValueError("Data must contain datetime or timestamp")
    if timestamp == "timestamp" and pd.api.types.is_numeric_dtype(frame[timestamp]):
        index = pd.to_datetime(frame[timestamp], unit="ms", utc=True)
    else:
        index = pd.to_datetime(frame[timestamp], utc=True)
    frame.index = pd.DatetimeIndex(index, name="timestamp")
    frame = frame.rename(columns=str.lower)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]
    frame = frame[cols].apply(pd.to_numeric, errors="coerce").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    return frame
