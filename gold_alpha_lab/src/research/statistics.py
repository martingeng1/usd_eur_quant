from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
from src.backtest.metrics import max_drawdown


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted q-values; NaN hypotheses remain NaN."""
    valid = p_values.dropna()
    if valid.empty:
        return pd.Series(float("nan"), index=p_values.index)
    ordered = valid.sort_values()
    m = len(ordered)
    adjusted = (ordered * m / pd.Series(range(1, m + 1), index=ordered.index)).iloc[::-1].cummin().clip(upper=1)
    return adjusted.reindex(p_values.index)


def summarize_returns(values: pd.Series, annualization: int = 252) -> dict[str, float | int]:
    x = values.dropna().astype(float)
    if len(x) == 0:
        return {"n": 0}
    mean, std = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
    downside = float(x[x < 0].std(ddof=1)) if (x < 0).sum() > 1 else 0.0
    t_stat, p_value = stats.ttest_1samp(x, 0.0) if len(x) > 1 else (np.nan, np.nan)
    gross_profit, gross_loss = x[x > 0].sum(), -x[x < 0].sum()
    return {
        "n": int(len(x)), "win_rate": float((x > 0).mean()), "mean": mean, "median": float(x.median()),
        "std": std, "profit_factor": float(gross_profit / gross_loss) if gross_loss else np.nan,
        "sharpe": float(mean / std * math.sqrt(annualization)) if std else np.nan,
        "sortino": float(mean / downside * math.sqrt(annualization)) if downside else np.nan,
        "t_stat": float(t_stat), "p_value": float(p_value), "max_drawdown": max_drawdown(x),
    }
