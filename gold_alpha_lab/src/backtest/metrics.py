from __future__ import annotations
import numpy as np
import pandas as pd

def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
