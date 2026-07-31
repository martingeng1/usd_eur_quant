"""
动量策略 — MACD 信号 + 多时间框架确认
"""
import pandas as pd
import numpy as np


def compute_momentum_signal(df, params=None):
    """
    计算动量信号

    逻辑：
    - MACD 线上穿信号线 → 看涨动量
    - MACD 线下穿信号线 → 看跌动量
    - 返回: +1 (多头), -1 (空头), 0 (无信号)

    参数
    ----
    df : pd.DataFrame, 需包含 'close' 列
    params : dict

    返回
    ----
    pd.Series : 信号序列
    """
    if params is None:
        from config import MOMENTUM_PARAMS as params

    macd_fast = params["macd_fast"]
    macd_slow = params["macd_slow"]
    macd_signal = params["macd_signal"]

    close = df["close"]

    # MACD
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 信号
    final_signal = pd.Series(0, index=df.index, dtype=float)

    # 金叉：MACD线上穿信号线
    golden_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    final_signal[golden_cross] = 1.0

    # 死叉：MACD线下穿信号线
    death_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    final_signal[death_cross] = -1.0

    # v7改进：不再填充持续性信号，仅在金叉/死叉时输出信号
    # 这大幅减少了交易频率，避免在MACD横盘时反复翻转

    return final_signal


def get_momentum_confidence(df, params=None):
    """
    动量置信度 (0-1)

    基于 MACD 柱状图强度
    """
    if params is None:
        from config import MOMENTUM_PARAMS as params

    macd_fast = params["macd_fast"]
    macd_slow = params["macd_slow"]
    macd_signal_period = params["macd_signal"]

    close = df["close"]

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # 用 MACD 柱状图的 Z-score 做置信度
    hist_rolling_std = macd_hist.rolling(window=50).std()
    hist_zscore = abs(macd_hist) / (hist_rolling_std + 1e-10)

    confidence = (hist_zscore / 3.0).clip(0, 1)
    return confidence.fillna(0)