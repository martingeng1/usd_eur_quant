from __future__ import annotations
import numpy as np
import pandas as pd


def detect_regime(frame: pd.DataFrame) -> pd.Series:
    vol_cut = frame.realized_vol.rolling(252, min_periods=100).quantile(0.8)
    return pd.Series(np.select(
        [frame.realized_vol > vol_cut, frame.trend_state.eq("up"), frame.trend_state.eq("down")],
        ["high_vol", "trend_up", "trend_down"], default="range"), index=frame.index, name="regime")
