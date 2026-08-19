from __future__ import annotations
import pandas as pd


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in frame:
        agg["volume"] = "sum"
    return frame.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open", "high", "low", "close"])
