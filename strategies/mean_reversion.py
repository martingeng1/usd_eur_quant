"""
均值回归策略 — 布林带 + RSI 超买超卖检测
"""
import pandas as pd
import numpy as np


def compute_mean_reversion_signal(df, params=None):
    """
    计算均值回归信号

    逻辑：
    - 价格触及布林带下轨 + RSI 超卖 → 看涨反转
    - 价格触及布林带上轨 + RSI 超买 → 看跌反转
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
        from config import MEAN_REVERSION_PARAMS as params

    bb_period = params["bb_period"]
    bb_std = params["bb_std"]
    rsi_period = params["rsi_period"]
    rsi_oversold = params["rsi_oversold"]
    rsi_overbought = params["rsi_overbought"]

    close = df["close"]

    # 布林带
    bb_mid = close.rolling(window=bb_period).mean()
    bb_std_val = close.rolling(window=bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 信号
    signal = pd.Series(0, index=df.index, dtype=float)

    # 多头：价格 <= 下轨 且 RSI < 超卖阈值
    long_cond = (close <= bb_lower) & (rsi < rsi_oversold)
    signal[long_cond] = 1.0

    # 空头：价格 >= 上轨 且 RSI > 超买阈值
    short_cond = (close >= bb_upper) & (rsi > rsi_overbought)
    signal[short_cond] = -1.0

    return signal


def get_reversion_confidence(df, params=None):
    """
    均值回归置信度 (0-1)

    RSI 越极端，布林带突破越深，反转概率越高
    """
    if params is None:
        from config import MEAN_REVERSION_PARAMS as params

    bb_period = params["bb_period"]
    bb_std = params["bb_std"]
    rsi_period = params["rsi_period"]

    close = df["close"]

    bb_mid = close.rolling(window=bb_period).mean()
    bb_std_val = close.rolling(window=bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_val
    bb_lower = bb_mid - bb_std * bb_std_val

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # 布林带位置：0 = 中轨, 1 = 上轨, -1 = 下轨
    bb_position = ((close - bb_mid) / (bb_std_val * bb_std + 1e-10)).clip(-1, 1)

    # RSI 极端度：(0-100 → 转换到 0-1, 以 50 为中心)
    rsi_extreme = abs(rsi - 50) / 50.0

    # 综合置信度 = 布林带偏离度 * RSI 极端度
    confidence = (abs(bb_position) * rsi_extreme).clip(0, 1)
    return confidence.fillna(0)