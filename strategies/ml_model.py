"""
机器学习增强模型 — XGBoost 特征融合，预测价格方向
"""
import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "strategies", "xgboost_audusd.pkl")


def build_features(df, lookback=50):
    """
    构建技术指标特征矩阵

    生成 25+ 特征：
    - 价格动量（收益率 lag 1-5）
    - 波动率（ATR, 历史波动率）
    - 趋势（EMA斜率, ADX）
    - 动量（RSI, MACD）
    - 均值回归（布林带位置）
    - 成交量特征

    返回
    ----
    X : pd.DataFrame, 特征矩阵
    y : pd.Series, 下一根K线的方向 (1=涨, 0=跌)
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    features = pd.DataFrame(index=df.index)

    # --- 收益率 lag ---
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f"return_lag_{lag}"] = close.pct_change(lag)

    # --- 价格位置 ---
    features["hl_ratio"] = (close - low) / (high - low + 1e-10)
    features["close_position"] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min() + 1e-10)

    # --- 波动率 ---
    features["volatility_5"] = close.pct_change().rolling(5).std()
    features["volatility_20"] = close.pct_change().rolling(20).std()
    features["atr_ratio"] = compute_atr(df, 14) / close

    # --- 趋势 ---
    for span in [9, 20, 50, 200]:
        ema = close.ewm(span=span, adjust=False).mean()
        features[f"ema_{span}_slope"] = (ema - ema.shift(5)) / (ema.shift(5) + 1e-10)

    # ADX
    adx = compute_adx(df, 14)
    features["adx"] = adx

    # --- 动量 ---
    features["rsi_14"] = compute_rsi(close, 14)
    features["rsi_7"] = compute_rsi(close, 7)
    features["macd_hist"] = compute_macd_hist(close)
    features["macd_line"] = compute_macd_line(close)

    # --- 均值回归 ---
    bb_position = (close - close.rolling(20).mean()) / (close.rolling(20).std() * 2 + 1e-10)
    features["bb_position"] = bb_position

    # --- 成交量 ---
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    features["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, 1)

    # --- 目标：下根K线涨跌 ---
    y = (close.shift(-1) > close).astype(int)

    # 清理
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().fillna(0)
    y = y.fillna(0)

    return features, y


def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd_line(close, fast=12, slow=26):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_macd_hist(close, fast=12, slow=26, signal_period=9):
    macd_line = compute_macd_line(close, fast, slow)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line - signal_line


def train_ml_model(df, lookback=50, retrain=True):
    """
    训练 XGBoost 模型

    返回
    ----
    model : 训练好的 XGBoost 模型
    feature_cols : 特征列名列表
    """
    from config import ML_PARAMS

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[ML] XGBoost 未安装，请运行: pip install xgboost")
        return None, None

    X, y = build_features(df, lookback)

    if len(X) < 500:
        print("[ML] 数据量不足（< 500），跳过训练")
        return None, None

    train_split = ML_PARAMS["train_split"]
    split_idx = int(len(X) * train_split)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[ML] 训练集: {len(X_train)}, 测试集: {len(X_test)}, 特征数: {X.shape[1]}")

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # 评估
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"[ML] 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")

    # 保存模型
    if retrain:
        joblib.dump({"model": model, "feature_cols": X.columns.tolist()}, MODEL_PATH)
        print(f"[ML] 模型已保存到 {MODEL_PATH}")

    return model, X.columns.tolist()


def compute_ml_signal(df, model=None, feature_cols=None, lookback=50):
    """
    使用训练好的 ML 模型预测信号

    返回
    ----
    pd.Series : +1 (预测涨), -1 (预测跌), 0 (无信号/无模型)
    """
    from config import ML_PARAMS

    threshold_long = ML_PARAMS.get("signal_threshold_long", 0.60)
    threshold_short = ML_PARAMS.get("signal_threshold_short", 0.40)

    # 尝试加载模型
    if model is None:
        if os.path.exists(MODEL_PATH):
            saved = joblib.load(MODEL_PATH)
            model = saved["model"]
            feature_cols = saved["feature_cols"]
        else:
            print("[ML] 未找到已训练模型，返回中性信号")
            return pd.Series(0, index=df.index, dtype=float)

    X, _ = build_features(df, lookback)

    if feature_cols:
        X = X[feature_cols]

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # 预测概率
    proba = model.predict_proba(X)
    prob_up = proba[:, 1]  # 涨的概率

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[prob_up > threshold_long] = 1.0
    signal[prob_up < threshold_short] = -1.0

    return signal


def get_ml_confidence(df, model=None, feature_cols=None, lookback=50):
    """ML 预测置信度 (0-1)，基于预测概率偏离50%的程度"""
    if model is None:
        if os.path.exists(MODEL_PATH):
            saved = joblib.load(MODEL_PATH)
            model = saved["model"]
            feature_cols = saved["feature_cols"]
        else:
            return pd.Series(0, index=df.index, dtype=float)

    X, _ = build_features(df, lookback)
    if feature_cols:
        X = X[feature_cols]
    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    proba = model.predict_proba(X)
    prob_up = proba[:, 1]

    confidence = (abs(prob_up - 0.5) * 2).clip(0, 1)
    return pd.Series(confidence, index=df.index).fillna(0)