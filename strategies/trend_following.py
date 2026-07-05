"""
趋势跟踪策略 — EMA 双均线交叉 + ADX 趋势过滤
"""
import pandas as pd
import numpy as np


def compute_trend_signal(df, params=None):
    """
    计算趋势跟踪信号

    逻辑：
    - EMA 快线 > EMA 慢线 → 看涨
    - ADX > 阈值 → 趋势有效，否则为震荡市
    - 返回: +1 (多头), -1 (空头), 0 (无信号)

    参数
    ----
    df : pd.DataFrame, 需包含 'close', 'high', 'low' 列
    params : dict, 趋势参数

    返回
    ----
    pd.Series : 信号序列
    """
    if params is None:
        from config import TREND_PARAMS as params

    ema_fast = params["ema_fast"]
    ema_slow = params["ema_slow"]
    adx_period = params["adx_period"]
    adx_threshold = params["adx_threshold"]

    close = df["close"]

    # EMA
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    # ADX 计算
    high = df["high"]
    low = df["low"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    # 平滑 TR 和 DM
    atr = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()

    # 信号生成
    signal = pd.Series(0, index=df.index, dtype=float)

    # 多头: EMA快 > EMA慢 且 ADX > 阈值
    long_cond = (ema_f > ema_s) & (adx > adx_threshold)
    signal[long_cond] = 1.0

    # 空头: EMA快 < EMA慢 且 ADX > 阈值
    short_cond = (ema_f < ema_s) & (adx > adx_threshold)
    signal[short_cond] = -1.0

    # 震荡过滤：ADX 低于阈值时信号置零
    weak_trend = adx < adx_threshold
    signal[weak_trend] = 0.0

    return signal


def get_trend_confidence(df, params=None):
    """
    计算趋势信号置信度 (0-1)，用于集成投票权重

    基于 ADX 强度：ADX 越高，趋势越可靠
    """
    if params is None:
        from config import TREND_PARAMS as params

    adx_period = params["adx_period"]

    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr = tr.ewm(alpha=1.0 / adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / adx_period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1.0 / adx_period, adjust=False).mean()

    # ADX 归一化到 [0, 1]，40 以上视为满分
    confidence = (adx / 40.0).clip(0.0, 1.0)
    return confidence.fillna(0)