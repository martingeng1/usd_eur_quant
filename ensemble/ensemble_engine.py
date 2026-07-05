"""
多策略集成引擎 v3 — 连续信号 + 波动率过滤 + 200EMA趋势过滤
"""
import pandas as pd
import numpy as np

from strategies.trend_following import compute_trend_signal, get_trend_confidence
from strategies.mean_reversion import compute_mean_reversion_signal, get_reversion_confidence
from strategies.momentum import compute_momentum_signal, get_momentum_confidence


def compute_ensemble_signal(df, weights=None, use_ml=False):
    """
    多策略加权评分 — 始终输出方向信号

    改进点：
    1. 加权评分 > 0 → 多头, < 0 → 空头（始终持仓）
    2. 波动率过低 → 平仓（只有这个条件会让信号归零）
    3. 多头必须 price > 200EMA，空头必须 price < 200EMA

    返回
    ----
    results : dict
    """
    from config import VOLATILITY_MIN_ATR_PCT, USE_200EMA_FILTER

    if weights is None:
        from config import ENSEMBLE_WEIGHTS as weights

    # 1. 计算子策略信号
    trend_sig = compute_trend_signal(df)
    reversion_sig = compute_mean_reversion_signal(df)
    momentum_sig = compute_momentum_signal(df)

    trend_conf = get_trend_confidence(df)
    reversion_conf = get_reversion_confidence(df)
    momentum_conf = get_momentum_confidence(df)

    sub_signals = {
        "trend_following": trend_sig,
        "mean_reversion": reversion_sig,
        "momentum": momentum_sig,
    }
    sub_confidences = {
        "trend_following": trend_conf,
        "mean_reversion": reversion_conf,
        "momentum": momentum_conf,
    }

    if use_ml:
        try:
            from strategies.ml_model import compute_ml_signal, get_ml_confidence
            sub_signals["ml_model"] = compute_ml_signal(df)
            sub_confidences["ml_model"] = get_ml_confidence(df)
        except Exception:
            pass

    # 加权得分
    weighted_score = pd.Series(0.0, index=df.index, dtype=float)
    for name, sig in sub_signals.items():
        w = weights.get(name, 0)
        weighted_score += sig * sub_confidences[name] * w

    # ---- 波动率过滤 ----
    close = df["close"]
    atr = compute_atr_series(df, 14)
    atr_pct = atr / close

    # ---- 200EMA 趋势过滤 ----
    ema200 = close.ewm(span=200, adjust=False).mean()
    above_200 = close > ema200

    # ---- 最终信号：始终持仓（动量方向），但低波动/逆趋势时平仓 ----
    signal = pd.Series(0.0, index=df.index, dtype=float)

    # 多头：得分 > 0 且 波动率够高
    long_ok = (weighted_score > 0) & (atr_pct > VOLATILITY_MIN_ATR_PCT)
    if USE_200EMA_FILTER:
        long_ok = long_ok & above_200
    signal[long_ok] = 1.0

    # 空头：得分 < 0 且 波动率够高
    short_ok = (weighted_score < 0) & (atr_pct > VOLATILITY_MIN_ATR_PCT)
    if USE_200EMA_FILTER:
        short_ok = short_ok & ~above_200
    signal[short_ok] = -1.0

    return {
        "ensemble_signal": signal,
        "ensemble_score": weighted_score,
        "sub_signals": sub_signals,
        "sub_confidences": sub_confidences,
        "atr_pct": atr_pct,
    }


def compute_atr_series(df, period=14):
    """ATR 序列"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()