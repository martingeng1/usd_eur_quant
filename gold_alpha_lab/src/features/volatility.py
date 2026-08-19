from __future__ import annotations
import pandas as pd


def add_atr(frame: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    result = frame.copy()
    previous = result.close.shift(1)
    tr = pd.concat([result.high - result.low, (result.high - previous).abs(), (result.low - previous).abs()], axis=1).max(axis=1)
    result["atr"] = tr.rolling(period, min_periods=period).mean()
    result["atr_pct"] = result.atr / result.close
    result["realized_vol"] = result.close.pct_change().rolling(period, min_periods=period).std()
    return result
