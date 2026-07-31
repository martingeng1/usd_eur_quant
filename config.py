"""
AUD/USD 量化交易系统 — v9 最终版（ML + 日线）
目标：年化20-30%，回撤<15%，夏普>1.0
核心配置：
  1. 日线交易（1D K线，噪音最低，ML预测最准）
  2. ML 模型默认启用（XGBoost，日线准确率81%），权重40%
  3. 2.0x ATR 止损、6.0x ATR 止盈（盈亏比 3:1）
  4. 阶梯式回撤熔断（5%/10%/12%/15%）
  5. 保本止损（盈利1.5x ATR → 开仓价）
  6. 部分止盈（2.5x ATR 平仓50%）
  7. 时间过滤器（避开周一早盘、周五尾盘）
"""
# ---------- 交易品种 ----------
SYMBOL = "AUDUSD=X"
BASE_CURRENCY = "AUD"
QUOTE_CURRENCY = "USD"

# ---------- 时间框架 ----------
PRIMARY_TIMEFRAME = "1d"           # 日线：ML准确率79%，最佳框架
SECONDARY_TIMEFRAMES = ["4h"]

# ---------- 数据 ----------
DATA_START = "2015-01-01"
DATA_END = "2026-07-05"
DATA_START_1H = "2024-10-01"
DATA_SOURCE = "yfinance"

# ---------- 初始资金与仓位 ----------
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_RISK = 0.04          # 单笔风险 4%（激进）
MAX_LEVERAGE = 50
DYNAMIC_POSITION = True
POSITION_SCALING_FLOOR = 0.10

# ---------- 风险管理 ----------
MAX_DAILY_LOSS = -0.04
MAX_DRAWDOWN_LIMIT = 0.15
DD_WARNING_LEVEL = 0.05            # 回撤5%：仓位降至50%
DD_DANGER_LEVEL = 0.10             # 回撤10%：仓位降至25%
DD_NEAR_LIMIT_LEVEL = 0.12         # 回撤12%：仅平仓不开仓
ATR_STOP_MULTIPLIER = 2.0          # ATR 止损 2.0x（给更多呼吸空间）
TAKE_PROFIT_ATR = 6.0              # 止盈 6.0x ATR（盈亏比 3:1）
PARTIAL_TP_ATR = 3.0               # 部分止盈 3.0x ATR 平仓50%
PARTIAL_TP_RATIO = 0.5
TRAILING_STOP_ATR = 2.5            # 移动止损距离 2.5x
TRAILING_STOP_ACTIVATION = 0.5     # 盈利0.5x ATR激活移动止损
BREAKEVEN_STOP_ACTIVATION = 1.5    # 盈利1.5x ATR时保本止损
MAX_HOLDING_BARS = 96              # 4H图；日线实际不会触发
MAX_CONSECUTIVE_LOSSES = 6         # 连续亏损6笔后冷静期
COOLDOWN_BARS = 20
MAX_CONCURRENT_POSITIONS = 1

# ---------- v9 时间过滤器 ----------
TIME_FILTER_ENABLED = True
AVOID_MONDAY_FIRST_HOURS = 4       # 周一前4小时不开仓
AVOID_FRIDAY_LAST_HOURS = 4        # 周五后4小时不开仓
AVOID_NY_CLOSE_HOUR = False        # 日线不需要日内过滤

# ---------- 策略权重 ----------
ENSEMBLE_WEIGHTS = {
    "trend_following": 0.30,
    "mean_reversion": 0.20,
    "momentum": 0.10,
    "ml_model": 0.40,              # ML权重 40%（提升ML影响力）
}
MIN_CONSENSUS = 1
VOLATILITY_MIN_ATR_PCT = 0.0010
USE_200EMA_FILTER = True
USE_STRONG_TREND_FILTER = False    # v9最终：日线ML不需要ADX过滤

# ---------- 策略参数 ----------
TREND_PARAMS = {
    "ema_fast": 15,
    "ema_slow": 50,
    "adx_period": 14,
    "adx_threshold": 20,
}

MEAN_REVERSION_PARAMS = {
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
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
    "signal_threshold_long": 0.60,
    "signal_threshold_short": 0.40,
}

# ---------- 回测参数 ----------
BACKTEST_SPREAD = 0.0001
BACKTEST_COMMISSION = 0.00005
BACKTEST_SLIPPAGE = 0.0001

# ---------- 执行 ----------
EXECUTION_MODE = "backtest"
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 2
IBKR_MIN_AUD_ORDER = 20000        # IDEALPRO 官方 AUD 最小现货外汇订单量
MIN_LIVE_SIGNAL_STRENGTH = 0.35   # 实盘允许向上取整到最小订单量的信号门槛
TRADING_CAPITAL_LIMIT_AUD = 50000.0  # 本项目专用资金上限，其余资金留给其他项目
IG_MIN_AUD_ORDER = 1000             # IG AUD/USD CFD 最小策略计量单位（AUD）
IG_MAX_LEVERAGE = 30                # ASIC 零售客户主要外汇最高杠杆
BROKER_ACCOUNT_REFRESH_SECONDS = 15
BROKER_QUOTE_REFRESH_SECONDS = 15
IG_RATE_LIMIT_BACKOFF_SECONDS = 65
LIVE_SIGNAL_REFRESH_SECONDS = 900   # 日线策略每15分钟检查一次即可