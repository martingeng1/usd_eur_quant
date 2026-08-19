from __future__ import annotations
import pandas as pd


def add_returns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["return_1"] = result.close.pct_change()
    return result
