"""
风险管理模块 v2 — 仓位计算、ATR 止损止盈、移动止损、回撤控制
"""
import pandas as pd
import numpy as np


def compute_atr(df, period=14):
    """计算 ATR"""
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_position_size(capital, entry_price, stop_loss_price, max_risk_pct=0.025, max_leverage=20):
    """
    基于固定比例风险计算仓位大小

    参数
    ----
    capital : float
    entry_price : float
    stop_loss_price : float
    max_risk_pct : float, 最大风险比例
    max_leverage : int, 最大杠杆

    返回
    ----
    float : 仓位大小
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0

    max_risk_amount = capital * max_risk_pct
    position_size = max_risk_amount / risk_per_unit

    max_position = capital * max_leverage / entry_price
    position_size = min(position_size, max_position)

    return position_size


def compute_stop_loss(entry_price, atr_value, multiplier=1.8, direction=1):
    """基于 ATR 计算止损价"""
    if direction == 1:
        return entry_price - atr_value * multiplier
    else:
        return entry_price + atr_value * multiplier


def compute_take_profit(entry_price, atr_value, multiplier=4.0, direction=1):
    """基于 ATR 计算止盈价"""
    if direction == 1:
        return entry_price + atr_value * multiplier
    else:
        return entry_price - atr_value * multiplier


class RiskManager:
    """
    风险控制器 v2 — 含移动止损
    """

    def __init__(self, initial_capital=100000.0):
        from config import (
            POSITION_SIZE_RISK, MAX_LEVERAGE, ATR_STOP_MULTIPLIER,
            TAKE_PROFIT_ATR, TRAILING_STOP_ATR, TRAILING_STOP_ACTIVATION,
            MAX_DRAWDOWN_LIMIT, MAX_DAILY_LOSS
        )
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size_risk = POSITION_SIZE_RISK
        self.max_leverage = MAX_LEVERAGE
        self.atr_stop_mult = ATR_STOP_MULTIPLIER
        self.take_profit_atr = TAKE_PROFIT_ATR
        self.trailing_stop_atr = TRAILING_STOP_ATR
        self.trailing_activation = TRAILING_STOP_ACTIVATION
        self.max_drawdown = MAX_DRAWDOWN_LIMIT
        self.max_daily_loss = MAX_DAILY_LOSS

        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.best_price = 0.0           # 持仓期间最有利价格
        self.equity_history = []
        self.trade_open = False

    def reset(self):
        self.current_capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.best_price = 0.0
        self.equity_history = []
        self.trade_open = False

    def can_trade(self):
        if self.trade_open:
            return False, "已有持仓"
        if len(self.equity_history) > 1:
            running_max = pd.Series(self.equity_history).cummax().iloc[-1]
            dd = (self.current_capital - running_max) / running_max
            if dd < -self.max_drawdown:
                return False, f"触发最大回撤 ({abs(dd):.1%})"
        return True, "ok"

    def open_trade(self, direction, entry_price, atr_value):
        ok, reason = self.can_trade()
        if not ok:
            return {"status": "rejected", "reason": reason}

        stop_loss = compute_stop_loss(entry_price, atr_value, self.atr_stop_mult, direction)
        take_profit = compute_take_profit(entry_price, atr_value, self.take_profit_atr, direction)
        pos_size = compute_position_size(
            self.current_capital, entry_price, stop_loss,
            self.position_size_risk, self.max_leverage
        )

        self.position = pos_size * direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = stop_loss
        self.trailing_activated = False
        self.best_price = entry_price
        self.trade_open = True

        return {
            "status": "opened",
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": entry_price,
            "position_size": pos_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_amount": abs(entry_price - stop_loss) * pos_size,
        }

    def update_trailing_stop(self, current_high, current_low, current_atr):
        """
        更新移动止损

        多头：价格创新高 → 止损上移
        空头：价格创新低 → 止损下移

        触发条件：盈利 >= trailing_activation * ATR 后才启用
        """
        if not self.trade_open:
            return

        trailing_distance = self.trailing_stop_atr * current_atr

        if self.position > 0:  # 多头
            # 更新最佳价格
            if current_high > self.best_price:
                self.best_price = current_high

            # 盈利达到激活阈值
            profit_pips = self.best_price - self.entry_price
            if not self.trailing_activated and profit_pips >= self.trailing_activation * current_atr:
                self.trailing_activated = True

            if self.trailing_activated:
                new_stop = self.best_price - trailing_distance
                self.stop_loss = max(self.stop_loss, new_stop)

        else:  # 空头
            if current_low < self.best_price:
                self.best_price = current_low

            profit_pips = self.entry_price - self.best_price
            if not self.trailing_activated and profit_pips >= self.trailing_activation * current_atr:
                self.trailing_activated = True

            if self.trailing_activated:
                new_stop = self.best_price + trailing_distance
                self.stop_loss = min(self.stop_loss, new_stop)

    def check_exit(self, current_price, current_high=None, current_low=None):
        if not self.trade_open:
            return False, None, None

        # 价格触及止损/止盈
        if self.position > 0:
            if current_low and current_low <= self.stop_loss:
                return True, "stop_loss", self.stop_loss
            if current_high and current_high >= self.take_profit:
                return True, "take_profit", self.take_profit
        else:
            if current_high and current_high >= self.stop_loss:
                return True, "stop_loss", self.stop_loss
            if current_low and current_low <= self.take_profit:
                return True, "take_profit", self.take_profit

        return False, None, None

    def close_trade(self, exit_price, exit_reason="signal"):
        if not self.trade_open:
            return {"status": "no_position"}

        pnl = self.position * (exit_price - self.entry_price)
        self.current_capital += pnl
        self.equity_history.append(self.current_capital)

        result = {
            "status": "closed",
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl / self.current_capital if self.current_capital else 0,
            "current_capital": self.current_capital,
            "reason": exit_reason,
        }

        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.best_price = 0.0
        self.trade_open = False

        return result