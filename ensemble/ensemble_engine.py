"""
多策略集成引擎 v4 — 高收益低回撤版
核心改进：
  1. 多策略共识过滤：至少 MIN_CONSENSUS 个子策略同意才开仓
  2. 信号质量评分：加权得分 + 共识强度
  3. 趋势强度确认：ADX + EMA200 双重过滤
  4. 波动率自适应：高波动时降低信号敏感度
"""
import pandas as pd
import numpy as np

from strategies.trend_following import compute_trend_signal, get_trend_confidence
from strategies.mean_reversion import compute_mean_reversion_signal, get_reversion_confidence
from strategies.momentum import compute_momentum_signal, get_momentum_confidence


def apply_time_filter(signal, df):
    """
    v9 时间过滤器：清除低质量时间段内的信号

    过滤规则：
    1. 周一前 N 小时（流动性差，跳空风险）
    2. 周五后 N 小时（避免周末风险）
    3. 纽约收盘前后（21:00-23:00 UTC 低质量时段）
    """
    from config import (
        TIME_FILTER_ENABLED, AVOID_MONDAY_FIRST_HOURS,
        AVOID_FRIDAY_LAST_HOURS, AVOID_NY_CLOSE_HOUR
    )
    if not TIME_FILTER_ENABLED:
        return signal

    filtered = signal.copy()
    for i, idx in enumerate(signal.index):
        if pd.isna(idx):
            continue
        t = pd.Timestamp(idx)
        dow = t.dayofweek  # 0=Mon, 4=Fri
        hour = t.hour

        # 周一前几小时
        if dow == 0 and hour < AVOID_MONDAY_FIRST_HOURS:
            filtered.iloc[i] = 0

        # 周五后几小时
        if dow == 4 and hour >= (24 - AVOID_FRIDAY_LAST_HOURS):
            filtered.iloc[i] = 0

        # 纽约收盘前后 (21-23 UTC ≈ NY 16:00-18:00)
        if AVOID_NY_CLOSE_HOUR and 21 <= hour <= 23:
            filtered.iloc[i] = 0

    return filtered


def compute_ensemble_signal(df, weights=None, use_ml=False):
    """
    多策略加权评分 — v9 高质量信号过滤版

    改进点：
    1. 多策略共识：需要至少 MIN_CONSENSUS 个子策略同意同一方向
    2. 加权评分 > 阈值 → 多头, < -阈值 → 空头
    3. 波动率自适应过滤
    4. 200EMA 趋势确认
    5. 信号强度评分（用于仓位缩放）
    6. v9: 时间过滤器（避开低质量时段）
    7. v9: ML 模型默认启用（如果有训练好的模型）

    返回
    ----
    results : dict
    """
    from config import (
        VOLATILITY_MIN_ATR_PCT, USE_200EMA_FILTER, MIN_CONSENSUS
    )

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

    # v9: ML 模型默认尝试加载（如果有 saved model）
    ml_loaded = False
    if use_ml:
        try:
            from strategies.ml_model import compute_ml_signal, get_ml_confidence
            sub_signals["ml_model"] = compute_ml_signal(df)
            sub_confidences["ml_model"] = get_ml_confidence(df)
            ml_loaded = True
        except Exception:
            pass

    # ---- 多策略共识计数 ----
    n_strategies = len(sub_signals)
    long_consensus = pd.Series(0, index=df.index, dtype=int)
    short_consensus = pd.Series(0, index=df.index, dtype=int)
    consensus_score = pd.Series(0.0, index=df.index, dtype=float)

    for name, sig in sub_signals.items():
        w = weights.get(name, 0)
        conf = sub_confidences[name].fillna(0)
        long_consensus += (sig > 0).astype(int) * (conf > 0.3).astype(int)
        short_consensus += (sig < 0).astype(int) * (conf > 0.3).astype(int)
        consensus_score += sig * conf * w

    # ---- 共识过滤 ----
    long_consensus_ok = long_consensus >= MIN_CONSENSUS
    short_consensus_ok = short_consensus >= MIN_CONSENSUS

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

    # ---- 趋势强度过滤（ADX辅助）- 可通过config开关 ----
    adx = compute_adx_series(df, 14)
    from config import USE_STRONG_TREND_FILTER
    if USE_STRONG_TREND_FILTER:
        strong_trend = adx > 20  # 日线：要求明确趋势
    else:
        strong_trend = pd.Series(True, index=df.index)  # 1H: 关闭强趋势过滤


    # ---- 最终信号 ----
    signal = pd.Series(0.0, index=df.index, dtype=float)
    signal_strength = pd.Series(0.0, index=df.index, dtype=float)

    # 多头条件：
    #   1. 加权得分 > 0
    #   2. 至少 MIN_CONSENSUS 个策略同意
    #   3. 波动率够高
    #   4. 价格在200EMA之上
    #   5. v7新增：同时满足 strong_trend 或 above_200（强制至少一个趋势确认）
    long_ok = (
        (weighted_score > 0) &
        long_consensus_ok &
        (atr_pct > VOLATILITY_MIN_ATR_PCT) &
        (above_200 | strong_trend)  # 必须趋势确认
    )
    signal[long_ok] = 1.0
    signal_strength[long_ok] = (
        (weighted_score[long_ok].clip(0, 1) * 0.4 +
         (long_consensus[long_ok] / n_strategies) * 0.35 +
         strong_trend[long_ok].astype(float) * 0.25)
    ).clip(0.3, 1.0)

    # 空头条件：
    #   1. 加权得分 < 0
    #   2. 至少 MIN_CONSENSUS 个策略同意
    #   3. 波动率够高
    #   4. 价格在200EMA之下
    #   5. v7新增：同时满足 strong_trend 或 ~above_200（强制至少一个趋势确认）
    short_ok = (
        (weighted_score < 0) &
        short_consensus_ok &
        (atr_pct > VOLATILITY_MIN_ATR_PCT) &
        (~above_200 | strong_trend)  # 必须趋势确认
    )
    signal[short_ok] = -1.0
    signal_strength[short_ok] = (
        (abs(weighted_score[short_ok]).clip(0, 1) * 0.4 +
         (short_consensus[short_ok] / n_strategies) * 0.35 +
         strong_trend[short_ok].astype(float) * 0.25)
    ).clip(0.3, 1.0)

    # v9: 应用时间过滤器（在最终信号上清除低质量时段的信号）
    signal = apply_time_filter(signal, df)
    signal_strength[signal == 0] = 0.0

    return {
        "ensemble_signal": signal,
        "ensemble_score": weighted_score,
        "signal_strength": signal_strength,
        "sub_signals": sub_signals,
        "sub_confidences": sub_confidences,
        "atr_pct": atr_pct,
        "long_consensus": long_consensus,
        "short_consensus": short_consensus,
        "adx": adx,
        "ml_loaded": ml_loaded,
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


def compute_adx_series(df, period=14):
    """ADX 序列"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()
