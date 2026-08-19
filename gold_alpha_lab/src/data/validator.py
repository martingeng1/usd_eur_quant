from __future__ import annotations

from dataclasses import asdict, dataclass
import pandas as pd


@dataclass(frozen=True)
class DataQuality:
    rows: int
    start_utc: str
    end_utc: str
    inferred_bar_minutes: float
    duplicate_timestamps: int
    nonpositive_prices: int
    large_gaps: int


def validate_ohlcv(frame: pd.DataFrame) -> DataQuality:
    if frame.empty:
        raise ValueError("No rows after loading source data")
    diffs = frame.index.to_series().diff().dropna().dt.total_seconds() / 60
    bar = float(diffs.median()) if not diffs.empty else 0.0
    threshold = max(bar * 3, 180)
    return DataQuality(
        rows=len(frame), start_utc=frame.index.min().isoformat(), end_utc=frame.index.max().isoformat(),
        inferred_bar_minutes=bar, duplicate_timestamps=int(frame.index.duplicated().sum()),
        nonpositive_prices=int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        large_gaps=int((diffs > threshold).sum()),
    )


def quality_dict(frame: pd.DataFrame) -> dict[str, object]:
    return asdict(validate_ohlcv(frame))
