"""
USD/EUR 量化交易系统 — 激进优化版
目标：年化40%+，回撤<25%
"""
# ---------- 交易品种 ----------
SYMBOL = "EURUSD=X"
BASE_CURRENCY = "EUR"
QUOTE_CURRENCY = "USD"

# ---------- 时间框架 ----------
PRIMARY_TIMEFRAME = "1h"
SECONDARY_TIMEFRAMES = ["4h", "1d"]

# ---------- 数据 ----------
DATA_START = "2015-01-01"
DATA_END = "2026-07-05"
DATA_START_1H = "2024-10-01"       # Yahoo 1H 数据仅限最近 730 天
DATA_SOURCE = "yfinance"

# ---------- 初始资金与仓位 ----------
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_RISK = 0.03         # 单笔风险 3%（提升至40%+CAGR）
MAX_LEVERAGE = 50                 # 外汇杠杆 50倍

# ---------- 风险管理 ----------
MAX_DAILY_LOSS = -0.05
MAX_DRAWDOWN_LIMIT = 0.25         # 允许25%回撤
ATR_STOP_MULTIPLIER = 1.5         # ATR 止损倍数（收紧）
TAKE_PROFIT_ATR = 4.0             # 止盈目标
TRAILING_STOP_ATR = 1.0           # 移动止损距离
TRAILING_STOP_ACTIVATION = 0.8    # 快速激活移动止损
MAX_CONCURRENT_POSITIONS = 1

# ---------- 策略权重 ----------
ENSEMBLE_WEIGHTS = {
    "trend_following": 0.35,
    "mean_reversion": 0.20,
    "momentum": 0.20,
    "ml_model": 0.25,             # ML权重提高到25%
}
MIN_CONSENSUS = 1
VOLATILITY_MIN_ATR_PCT = 0.002    # 波动率阈值
USE_200EMA_FILTER = True          # 启用200EMA过滤

# ---------- 策略参数 ----------
TREND_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_period": 14,
    "adx_threshold": 20,
}

MEAN_REVERSION_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 32,
    "rsi_overbought": 68,
}

MOMENTUM_PARAMS = {
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
}

ML_PARAMS = {
    "lookback": 50,
    "train_split": 0.7,
    "model_type": "xgboost",
    "retrain_frequency": 20,
}

# ---------- 回测参数 ----------
BACKTEST_SPREAD = 0.0001
BACKTEST_COMMISSION = 0.00005
BACKTEST_SLIPPAGE = 0.0001

# ---------- 执行 ----------
EXECUTION_MODE = "backtest"
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 1