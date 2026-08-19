from __future__ import annotations
from collections.abc import Mapping
import pandas as pd
from src.data.timezone import in_local_window


def add_session_flags(frame: pd.DataFrame, sessions: Mapping[str, Mapping[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    for name, spec in sessions.items():
        result[f"session_{name}"] = in_local_window(result.index, spec["timezone"], spec["start"], spec["end"]).to_numpy()
    return result
