from __future__ import annotations
import pandas as pd

def primary_state(frame: pd.DataFrame) -> pd.Series:
    return frame.get("regime", pd.Series("unknown", index=frame.index)).rename("primary_state")
