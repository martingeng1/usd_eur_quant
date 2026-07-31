"""
AUD/USD 量化交易系统 — 实盘配置 v9 (ML + 日线)
与 config.py 回测参数保持一致
"""
# ---------- 账户 ----------
ACCOUNT_CURRENCY = "AUD"
INITIAL_CAPITAL_AUD = 2000.0        # 账户资金（澳元）
INITIAL_CAPITAL_USD = 1200.0        # 约合美元（1 AUD ≈ 0.60 USD）

# ---------- 交易品种 ----------
SYMBOL = "AUDUSD"
BASE_CURRENCY = "AUD"
QUOTE_CURRENCY = "USD"

# ---------- 时间框架 ----------
PRIMARY_TIMEFRAME = "1d"            # 日线：ML准确率79%

# ---------- 仓位控制 ----------
POSITION_SIZE_RISK = 0.02           # 单笔风险 2%
MAX_LEVERAGE = 50
MIN_POSITION_USD = 1000             # 最小仓位（1微型手）

# ---------- 风险管理 ----------
MAX_DAILY_LOSS_PCT = 0.04           # 日内最大亏损 4%
MAX_DRAWDOWN_LIMIT = 0.15           # 最大回撤 15%
DD_WARNING_LEVEL = 0.05             # 回撤5%：仓位降至50%
DD_DANGER_LEVEL = 0.10              # 回撤10%：仓位降至25%
DD_NEAR_LIMIT_LEVEL = 0.12          # 回撤12%：仅平仓不开仓
ATR_STOP_MULTIPLIER = 2.0           # ATR 止损 2.0x
TAKE_PROFIT_ATR = 5.0               # 止盈 5.0x ATR
PARTIAL_TP_ATR = 2.5                # 部分止盈 2.5x ATR
PARTIAL_TP_RATIO = 0.5              # 部分止盈比例 50%
TRAILING_STOP_ATR = 2.0             # 移动止损距离 2.0x
TRAILING_STOP_ACTIVATION = 0.5      # 移动止损激活阈值 0.5x ATR
BREAKEVEN_STOP_ACTIVATION = 1.5     # 保本止损激活 1.5x ATR
MAX_HOLDING_DAYS = 30               # 持仓时间上限（日线30天）
MAX_CONSECUTIVE_LOSSES = 6          # 连续亏损6笔冷静期
COOLDOWN_DAYS = 5                   # 冷静期5天（日线）

# ---------- 时间过滤器 ----------
TIME_FILTER_ENABLED = False         # 日线不需要日内过滤
AVOID_MONDAY = False                # 日线可以周一交易

# ---------- 策略权重 ----------
ENSEMBLE_WEIGHTS = {
    "trend_following": 0.35,
    "mean_reversion": 0.25,
    "momentum": 0.15,
    "ml_model": 0.25,
}
MIN_CONSENSUS = 1
VOLATILITY_MIN_ATR_PCT = 0.0010
USE_200EMA_FILTER = True
USE_STRONG_TREND_FILTER = False     # ML已经隐含趋势信息

# ---------- ML 参数 ----------
ML_PARAMS = {
    "lookback": 50,
    "train_split": 0.7,
    "model_type": "xgboost",
    "retrain_frequency": 20,        # 每20根K线重训练
    "signal_threshold_long": 0.55,
    "signal_threshold_short": 0.45,
}

# ---------- 交易成本 ----------
SPREAD = 0.0001
COMMISSION = 0.00002

# ---------- IBKR 连接 ----------
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002                    # IB Gateway 纸交易端口 4002；TWS 实盘端口 7497
IBKR_CLIENT_ID = 2                  # Client ID（同一时间每个ID只能有一个连接；如果冲突可改为 1/3/4）

# ---------- 执行模式 ----------
EXECUTION_MODE = "paper"            # paper → 确认无误后改为 live
CHECK_INTERVAL_SECONDS = 14400      # 4小时检查一次（日线不需要太频繁）
RETRAIN_INTERVAL_DAYS = 20          # 每20天重训练ML