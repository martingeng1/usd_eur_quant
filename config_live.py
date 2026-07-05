"""
USD/EUR 量化交易系统 — 小额实盘配置（1000 AUD）
"""
# ---------- 账户 ----------
ACCOUNT_CURRENCY = "AUD"
INITIAL_CAPITAL_AUD = 1000.0       # 账户资金（澳元）
INITIAL_CAPITAL_USD = 620.0        # 约合美元（1 AUD ≈ 0.62 USD）

# IBKR 会对 AUD 账户自动转换保证金到 USD
# 如果账户只有 AUD，IBKR 会按实时汇率自动借入 USD 用于保证金

# ---------- 交易品种 ----------
SYMBOL = "EURUSD"
BASE_CURRENCY = "EUR"
QUOTE_CURRENCY = "USD"

# ---------- 时间框架 ----------
PRIMARY_TIMEFRAME = "1h"           # 1小时K线
SECONDARY_TIMEFRAMES = ["4h"]

# ---------- 小额账户仓位控制 ----------
POSITION_SIZE_RISK = 0.05          # 单笔风险 5%（小额账户需要更高风险比例）
MAX_LEVERAGE = 50                  # 外汇杠杆
MIN_POSITION_USD = 1000            # 最小仓位（1微型手 = 1000 EUR ≈ 1100 USD）

# ---------- 风险管理 ----------
MAX_DAILY_LOSS_PCT = 0.08          # 日内最大亏损 8%
MAX_DRAWDOWN_LIMIT = 0.30          # 小额账户允许更大回撤
ATR_STOP_MULTIPLIER = 1.5          # ATR 止损倍数
TAKE_PROFIT_ATR = 3.0              # 止盈目标（紧缩以快速锁定利润）
TRAILING_STOP_ATR = 0.8            # 移动止损距离（更紧）
TRAILING_STOP_ACTIVATION = 0.5     # 快速激活移动止损

# ---------- 策略权重 ----------
ENSEMBLE_WEIGHTS = {
    "trend_following": 0.35,
    "mean_reversion": 0.20,
    "momentum": 0.20,
    "ml_model": 0.25,
}
MIN_CONSENSUS = 1
VOLATILITY_MIN_ATR_PCT = 0.002
USE_200EMA_FILTER = True

# ---------- 策略参数 ----------
TREND_PARAMS = {
    "ema_fast": 20,
    "ema_slow": 50,
    "adx_period": 14,
    "adx_threshold": 22,
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

# ---------- 交易成本 ----------
SPREAD = 0.0001                    # EUR/USD 典型点差 0.1 pip
COMMISSION = 0.00002               # IBKR 外汇佣金约 0.2 pip

# ---------- IBKR 连接 ----------
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002                   # IB Gateway 默认端口（纸交易）; TWS=7497
IBKR_CLIENT_ID = 1

# ---------- 执行模式 ----------
EXECUTION_MODE = "paper"           # paper（模拟）→ 确认无误后改为 live
CHECK_INTERVAL_SECONDS = 3600      # 1小时检查一次信号
MAX_RETRIES = 3                    # 最大重连次数