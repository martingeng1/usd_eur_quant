"""
回测引擎 — 完整的向量化回测 + 事件驱动回测
"""
import pandas as pd
import numpy as np
from datetime import datetime


def vectorized_backtest(df, signal_series, initial_capital=100000.0,
                        spread=0.0001, commission=0.00005, slippage=0.0001):
    """
    向量化回测 — 快速计算策略收益曲线

    逻辑：
    - 信号 +1 → 持有多头
    - 信号 -1 → 持有空头
    - 信号 0 → 空仓
    - 交易时扣除点差、手续费、滑点

    参数
    ----
    df : pd.DataFrame, 含 'close', 'high', 'low' 列
    signal_series : pd.Series, 信号序列 (+1/-1/0)
    initial_capital : float, 初始资金
    spread : float, 点差
    commission : float, 手续费比例
    slippage : float, 滑点

    返回
    ----
    results : dict
        {
            "equity_curve": pd.Series,
            "returns": pd.Series,
            "trades": list,
            "stats": dict
        }
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 检测信号变化点
    signal_shift = signal_series.shift(1).fillna(0)
    signal_change = signal_series != signal_shift

    # 每日收益（无交易成本）
    daily_returns = close.pct_change().fillna(0)

    # 策略收益 = 信号 * 市场收益
    strategy_returns = signal_shift * daily_returns
    strategy_returns = strategy_returns.fillna(0)

    # 交易成本（仅在信号变化时扣除）
    trade_cost = pd.Series(0.0, index=df.index, dtype=float)
    trade_cost[signal_change] = spread + commission * 2  # 开仓+平仓各一次
    strategy_returns = strategy_returns - trade_cost

    # 计算净值曲线
    cumulative_returns = (1 + strategy_returns).cumprod()
    equity_curve = cumulative_returns * initial_capital

    # 收集交易记录
    trades = []
    trade_entry_idx = None
    trade_entry_price = None
    trade_direction = None

    for i in range(len(df)):
        sig = signal_series.iloc[i]
        prev_sig = signal_shift.iloc[i]

        # 开仓
        if sig != 0 and prev_sig != sig:
            trade_entry_idx = i
            trade_entry_price = close.iloc[i]
            trade_direction = sig
        # 平仓（反转或归零）
        elif sig != prev_sig and trade_entry_idx is not None:
            exit_price = close.iloc[i]
            entry_price = trade_entry_price
            direction = trade_direction

            pnl = direction * (exit_price - entry_price) - spread - commission
            trades.append({
                "entry_time": df.index[trade_entry_idx],
                "exit_time": df.index[i],
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "bars": i - trade_entry_idx,
            })
            trade_entry_idx = None
            trade_entry_price = None
            trade_direction = None

    # 统计
    stats = compute_statistics(strategy_returns, equity_curve, trades, initial_capital)

    return {
        "equity_curve": equity_curve,
        "returns": strategy_returns,
        "trades": trades,
        "stats": stats,
    }


def event_driven_backtest(df, signal_series, risk_manager=None, initial_capital=100000.0,
                          spread=0.0001, commission=0.00005, slippage=0.0001):
    """
    事件驱动回测 — 模拟真实交易流程，含风控检查

    与向量化版本不同，这个版本：
    1. 逐根 K 线遍历
    2. 检查止损/止盈
    3. 风控过滤
    4. 滑点模拟
    5. 只能同时持有一笔仓位
    """
    from risk.risk_manager import RiskManager

    if risk_manager is None:
        risk_manager = RiskManager(initial_capital)

    risk_manager.reset()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr_series = compute_atr_series(df)

    trades = []
    equity_curve = [initial_capital]
    in_position = False
    current_signal = 0

    for i in range(len(df)):
        sig = signal_series.iloc[i]
        price = close.iloc[i]
        hi = high.iloc[i]
        lo = low.iloc[i]
        atr_val = atr_series.iloc[i]

        # 已有持仓 → 更新移动止损 + 检查止损/止盈
        if risk_manager.trade_open and in_position:
            # 移动止损
            risk_manager.update_trailing_stop(hi, lo, atr_val)
            exit_signal, exit_reason, exit_price = risk_manager.check_exit(price, hi, lo)
            if exit_signal:
                result = risk_manager.close_trade(exit_price, exit_reason)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "direction": entry_direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": result["pnl"],
                    "pnl_pct": result["pnl_pct"],
                    "reason": exit_reason,
                    "bars": i - entry_bar,
                })
                in_position = False

        # 新信号 → 开仓或平仓
        if sig != 0 and sig != current_signal:
            if in_position:
                # 反转持仓：先平仓
                result = risk_manager.close_trade(price, "signal_reverse")
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "direction": entry_direction,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": result["pnl"],
                    "pnl_pct": result["pnl_pct"],
                    "reason": "signal_reverse",
                    "bars": i - entry_bar,
                })
                in_position = False

            # 开新仓（需要确保当前无持仓）
            if not in_position:
                entry_result = risk_manager.open_trade(
                    int(sig), price + slippage if sig == 1 else price - slippage, atr_val
                )
                if entry_result["status"] == "opened":
                    in_position = True
                    entry_time = df.index[i]
                    entry_price = entry_result["entry_price"]
                    entry_direction = entry_result["direction"]
                    entry_bar = i

        # 信号消失 → 平仓
        elif sig == 0 and in_position:
            result = risk_manager.close_trade(price, "signal_flat")
            trades.append({
                "entry_time": entry_time,
                "exit_time": df.index[i],
                "direction": entry_direction,
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": result["pnl"],
                "pnl_pct": result["pnl_pct"],
                "reason": "signal_flat",
                "bars": i - entry_bar,
            })
            in_position = False

        current_signal = sig
        equity_curve.append(risk_manager.current_capital)

    # 强制平仓（回测结束时仍有持仓）
    if in_position and risk_manager.trade_open:
        result = risk_manager.close_trade(close.iloc[-1], "end_of_backtest")
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "direction": entry_direction,
            "entry_price": entry_price,
            "exit_price": close.iloc[-1],
            "pnl": result["pnl"],
            "pnl_pct": result["pnl_pct"],
            "reason": "end_of_backtest",
        })
        equity_curve.append(risk_manager.current_capital)

    equity_curve = pd.Series(equity_curve[1:], index=df.index)
    returns = equity_curve.pct_change().fillna(0)

    stats = compute_statistics(returns, equity_curve, trades, initial_capital)

    return {
        "equity_curve": equity_curve,
        "returns": returns,
        "trades": trades,
        "stats": stats,
    }


def compute_atr_series(df, period=14):
    """计算 ATR 序列"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_statistics(returns, equity_curve, trades, initial_capital):
    """
    计算策略绩效指标
    """
    stats = {}

    # 总交易数
    stats["total_trades"] = len(trades)
    if len(trades) == 0:
        stats.update({
            "total_return": 0, "cagr": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "win_rate": 0, "avg_win": 0,
            "avg_loss": 0, "profit_factor": 0, "calmar_ratio": 0,
        })
        return stats

    # 总收益率
    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    stats["total_return"] = total_return
    stats["final_equity"] = final_equity

    # 年化收益率 (CAGR) - 用 trading days / 252 估算
    trading_days = len(equity_curve)
    years = max(trading_days / 252, 0.01)
    stats["cagr"] = (final_equity / initial_capital) ** (1 / years) - 1

    # 夏普比率 (日线数据用 sqrt(252)，小时级用 sqrt(365*24))
    excess_returns = returns - 0.0
    if returns.std() > 0:
        stats["sharpe_ratio"] = excess_returns.mean() / returns.std() * np.sqrt(252)
    else:
        stats["sharpe_ratio"] = 0

    # 最大回撤
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    stats["max_drawdown"] = drawdown.min()
    stats["max_drawdown_duration"] = compute_max_drawdown_duration(equity_curve)

    # 胜率
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]
    stats["win_rate"] = len(winning_trades) / len(trades) if trades else 0

    # 平均盈亏
    stats["avg_win"] = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
    stats["avg_loss"] = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0

    # 盈亏比
    if stats["avg_loss"] != 0:
        stats["profit_factor"] = stats["avg_win"] / abs(stats["avg_loss"])
    else:
        stats["profit_factor"] = float("inf") if stats["avg_win"] > 0 else 0

    # 胜率 * 盈亏比
    stats["expectancy"] = stats["win_rate"] * stats["avg_win"] + (1 - stats["win_rate"]) * stats["avg_loss"]

    # 卡玛比率
    if abs(stats["max_drawdown"]) > 0.001:
        stats["calmar_ratio"] = stats["cagr"] / abs(stats["max_drawdown"])
    else:
        stats["calmar_ratio"] = 0

    return stats


def compute_max_drawdown_duration(equity_curve):
    """计算最大回撤持续时长（K线根数）"""
    running_max = equity_curve.cummax()
    is_drawdown = equity_curve < running_max

    max_duration = 0
    current_duration = 0
    for val in is_drawdown:
        if val:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration