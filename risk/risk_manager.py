"""
风险管理模块 v4 — 仓位计算、ATR 止损止盈、移动止损、阶梯式回撤控制、连续亏损熔断、日亏损限制、持仓时间上限、部分止盈
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


def compute_position_size(capital, entry_price, stop_loss_price, max_risk_pct=0.015,
                          max_leverage=25, dd_scale=1.0, signal_strength=1.0):
    """
    基于固定比例风险计算仓位大小（动态缩放版）

    参数
    ----
    capital : float
    entry_price : float
    stop_loss_price : float
    max_risk_pct : float, 最大风险比例
    max_leverage : int, 最大杠杆
    dd_scale : float, 回撤缩放因子 (0-1)
    signal_strength : float, 信号强度 (0-1)

    返回
    ----
    float : 仓位大小
    """
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0

    # 回撤越大仓位越小，信号越强仓位越大
    adjusted_risk_pct = max_risk_pct * dd_scale * signal_strength

    max_risk_amount = capital * adjusted_risk_pct
    position_size = max_risk_amount / risk_per_unit

    max_position = capital * max_leverage / entry_price
    position_size = min(position_size, max_position)

    return position_size


def compute_stop_loss(entry_price, atr_value, multiplier=1.0, direction=1):
    """基于 ATR 计算止损价"""
    if direction == 1:
        return entry_price - atr_value * multiplier
    else:
        return entry_price + atr_value * multiplier


def compute_take_profit(entry_price, atr_value, multiplier=2.5, direction=1):
    """基于 ATR 计算止盈价"""
    if direction == 1:
        return entry_price + atr_value * multiplier
    else:
        return entry_price - atr_value * multiplier


class RiskManager:
    """
    风险控制器 v4 — 含移动止损、保本止损、连续亏损熔断、日亏损限制、阶梯式回撤熔断、持仓时间上限、部分止盈
    """

    def __init__(self, initial_capital=100000.0):
        from config import (
            POSITION_SIZE_RISK, MAX_LEVERAGE, ATR_STOP_MULTIPLIER,
            TAKE_PROFIT_ATR, TRAILING_STOP_ATR, TRAILING_STOP_ACTIVATION,
            MAX_DRAWDOWN_LIMIT, MAX_DAILY_LOSS, MAX_CONSECUTIVE_LOSSES,
            COOLDOWN_BARS, DYNAMIC_POSITION, POSITION_SCALING_FLOOR,
            DD_WARNING_LEVEL, DD_DANGER_LEVEL, DD_NEAR_LIMIT_LEVEL,
            PARTIAL_TP_ATR, PARTIAL_TP_RATIO, MAX_HOLDING_BARS,
            BREAKEVEN_STOP_ACTIVATION
        )
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size_risk = POSITION_SIZE_RISK
        self.max_leverage = MAX_LEVERAGE
        self.atr_stop_mult = ATR_STOP_MULTIPLIER
        self.take_profit_atr = TAKE_PROFIT_ATR
        self.trailing_stop_atr = TRAILING_STOP_ATR
        self.trailing_activation = TRAILING_STOP_ACTIVATION
        self.breakeven_activation = BREAKEVEN_STOP_ACTIVATION  # v8新增
        self.max_drawdown = MAX_DRAWDOWN_LIMIT
        self.max_daily_loss = MAX_DAILY_LOSS
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.cooldown_bars = COOLDOWN_BARS
        self.dynamic_position = DYNAMIC_POSITION
        self.position_scaling_floor = POSITION_SCALING_FLOOR

        # 阶梯式回撤熔断参数
        self.dd_warning = DD_WARNING_LEVEL      # 5%
        self.dd_danger = DD_DANGER_LEVEL        # 10%
        self.dd_near_limit = DD_NEAR_LIMIT_LEVEL # 12%

        # 部分止盈参数
        self.partial_tp_atr = PARTIAL_TP_ATR
        self.partial_tp_ratio = PARTIAL_TP_RATIO

        # 持仓时间上限
        self.max_holding_bars = MAX_HOLDING_BARS

        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.breakeven_activated = False         # v8: 保本止损是否已激活
        self.best_price = 0.0           # 持仓期间最有利价格
        self.entry_bar = 0              # 开仓时K线索引
        self.equity_history = []
        self.trade_open = False
        self.partial_tp_triggered = False  # 是否已触发部分止盈

        # 连续亏损熔断
        self.consecutive_losses = 0
        self.cooldown_counter = 0       # 冷静期剩余K线数
        self.trade_history = []         # [(pnl, exit_bar), ...]

        # 日亏损追踪
        self.daily_pnl = 0.0
        self.last_day = None
        self.daily_loss_triggered = False

    def reset(self):
        self.current_capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.breakeven_activated = False
        self.best_price = 0.0
        self.entry_bar = 0
        self.equity_history = []
        self.trade_open = False
        self.partial_tp_triggered = False
        self.consecutive_losses = 0
        self.cooldown_counter = 0
        self.trade_history = []
        self.daily_pnl = 0.0
        self.last_day = None
        self.daily_loss_triggered = False

    def update_daily(self, current_time):
        """更新日统计，检测日亏损上限"""
        if hasattr(current_time, 'date'):
            current_day = current_time.date()
        else:
            # 如果传入的是字符串或datetime，尝试解析
            current_day = pd.Timestamp(current_time).date() if hasattr(current_time, 'date') else None

        if current_day is None:
            return False

        if self.last_day is not None and current_day != self.last_day:
            # 新的一天开始了
            if self.daily_pnl < self.max_daily_loss * self.current_capital:
                self.daily_loss_triggered = True
            else:
                self.daily_loss_triggered = False
            self.daily_pnl = 0.0

        self.last_day = current_day
        return self.daily_loss_triggered

    def get_drawdown(self):
        """获取当前回撤比例"""
        if len(self.equity_history) < 2:
            return 0.0
        all_equity = self.equity_history + [self.current_capital]
        running_max = max(all_equity)
        if running_max <= 0:
            return 0.0
        return (self.current_capital - running_max) / running_max

    def get_position_scaler(self):
        """
        v4 阶梯式回撤仓位缩放：
        回撤 0-5%   → 1.0（满仓）
        回撤 5-10%  → 0.50（半仓）
        回撤 10-12% → 0.25（轻仓）
        回撤 12-15% → 仅平仓不开仓
        回撤 >=15%  → 完全停止
        """
        if not self.dynamic_position:
            return 1.0

        dd = abs(self.get_drawdown())

        if dd < self.dd_warning:
            return 1.0
        elif dd < self.dd_danger:
            return 0.50
        elif dd < self.dd_near_limit:
            return 0.25
        else:
            return 0.0  # 不允许开新仓

    def can_trade(self):
        """检查是否可以交易（多重风控 v4 — 含阶梯式回撤熔断）"""
        if self.trade_open:
            return False, "已有持仓"

        # 1. 连续亏损熔断
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"连续亏损{self.consecutive_losses}次，已触发熔断"

        # 2. 冷静期
        if self.cooldown_counter > 0:
            return False, f"冷静期剩余{self.cooldown_counter}根K线"

        # 3. 日亏损限制
        if self.daily_loss_triggered:
            return False, f"当日亏损已达{abs(self.max_daily_loss)*100:.1f}%上限"

        # 4. 阶梯式回撤熔断
        dd = self.get_drawdown()
        if abs(dd) >= self.max_drawdown:
            return False, f"触发最大回撤 ({abs(dd):.1%})"
        if abs(dd) >= self.dd_near_limit:
            return False, f"回撤接近上限 ({abs(dd):.1%})，仅平仓不开仓"

        return True, "ok"

    def tick_cooldown(self):
        """每根K线调用，减少冷静期计数"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.consecutive_losses = 0  # 冷静期结束，重置连亏计数

    def open_trade(self, direction, entry_price, atr_value, signal_strength=1.0):
        """
        开仓（支持动态仓位缩放和信号强度加权）
        """
        ok, reason = self.can_trade()
        if not ok:
            return {"status": "rejected", "reason": reason}

        stop_loss = compute_stop_loss(entry_price, atr_value, self.atr_stop_mult, direction)
        take_profit = compute_take_profit(entry_price, atr_value, self.take_profit_atr, direction)
        dd_scale = self.get_position_scaler()
        pos_size = compute_position_size(
            self.current_capital, entry_price, stop_loss,
            self.position_size_risk, self.max_leverage,
            dd_scale, signal_strength
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
            "dd_scale": dd_scale,
            "signal_strength": signal_strength,
        }

    def update_trailing_stop(self, current_high, current_low, current_atr):
        """
        更新移动止损 v4 — 含保本止损

        多头：价格创新高 → 止损上移
        空头：价格创新低 → 止损下移

        触发条件：
        1. 盈利 >= breakeven_activation * ATR → 止损移到开仓价（保本）
        2. 盈利 >= trailing_activation * ATR → 启用移动止损
        """
        if not self.trade_open:
            return

        trailing_distance = self.trailing_stop_atr * current_atr

        if self.position > 0:  # 多头
            # 更新最佳价格
            if current_high > self.best_price:
                self.best_price = current_high

            profit_pips = self.best_price - self.entry_price

            # v8: 保本止损 — 盈利达标后止损移到开仓价
            if not self.breakeven_activated and profit_pips >= self.breakeven_activation * current_atr:
                self.breakeven_activated = True
                self.stop_loss = max(self.stop_loss, self.entry_price)

            # 移动止损激活
            if not self.trailing_activated and profit_pips >= self.trailing_activation * current_atr:
                self.trailing_activated = True

            if self.trailing_activated:
                new_stop = self.best_price - trailing_distance
                self.stop_loss = max(self.stop_loss, new_stop)

        else:  # 空头
            if current_low < self.best_price:
                self.best_price = current_low

            profit_pips = self.entry_price - self.best_price

            # v8: 保本止损 — 盈利达标后止损移到开仓价
            if not self.breakeven_activated and profit_pips >= self.breakeven_activation * current_atr:
                self.breakeven_activated = True
                self.stop_loss = min(self.stop_loss, self.entry_price)

            # 移动止损激活
            if not self.trailing_activated and profit_pips >= self.trailing_activation * current_atr:
                self.trailing_activated = True

            if self.trailing_activated:
                new_stop = self.best_price + trailing_distance
                self.stop_loss = min(self.stop_loss, new_stop)

    def check_exit(self, current_price, current_high=None, current_low=None, current_bar=None):
        if not self.trade_open:
            return False, None, None

        # 持仓时间上限检查
        if current_bar is not None and self.max_holding_bars > 0:
            holding_bars = current_bar - self.entry_bar
            if holding_bars >= self.max_holding_bars:
                return True, "max_holding_time", current_price

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

    def check_partial_tp(self, current_high, current_low, current_atr, entry_price, direction):
        """
        v4 部分止盈检查：当盈利达到 PARTIAL_TP_ATR 时，平仓 PARTIAL_TP_RATIO 比例
        返回 (should_partial_close, exit_price, ratio)
        """
        if self.partial_tp_triggered:
            return False, None, 0

        if direction == 1:  # 多头
            tp_price = entry_price + self.partial_tp_atr * current_atr
            if current_high >= tp_price:
                self.partial_tp_triggered = True
                return True, tp_price, self.partial_tp_ratio
        else:  # 空头
            tp_price = entry_price - self.partial_tp_atr * current_atr
            if current_low <= tp_price:
                self.partial_tp_triggered = True
                return True, tp_price, self.partial_tp_ratio

        return False, None, 0

    def close_trade(self, exit_price, exit_reason="signal", current_bar=None):
        if not self.trade_open:
            return {"status": "no_position"}

        pnl = self.position * (exit_price - self.entry_price)
        self.current_capital += pnl
        self.equity_history.append(self.current_capital)

        # 更新日亏损
        self.daily_pnl += pnl

        result = {
            "status": "closed",
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl / self.current_capital if self.current_capital else 0,
            "current_capital": self.current_capital,
            "reason": exit_reason,
        }

        # 记录交易历史（连续亏损熔断用）
        self.trade_history.append({
            "pnl": pnl,
            "bar": current_bar if current_bar is not None else len(self.trade_history),
        })

        # 连续亏损追踪
        if pnl <= 0:
            self.consecutive_losses += 1
            # 达到连续亏损上限，进入冷静期
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_counter = self.cooldown_bars
        else:
            # 盈利一笔就重置连亏计数
            self.consecutive_losses = 0
            self.cooldown_counter = 0

        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_activated = False
        self.breakeven_activated = False
        self.best_price = 0.0
        self.trade_open = False

        return result
