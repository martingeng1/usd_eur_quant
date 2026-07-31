"""
AUD/USD 量化交易系统 v9 — Web 管理面板
Flask 后端，提供回测、实盘管理、交易历史 API
"""
import sys
import os
import json
import threading
import time
from datetime import datetime, timedelta

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 终端日志捕获（重定向 print 到缓冲区，供 Web 面板显示）──
from collections import deque
_log_buffer = deque(maxlen=500)
_original_print = print

def _log_print(*args, **kwargs):
    ts = datetime.now().strftime("[%m-%d %H:%M:%S]")
    msg = f"{ts} " + " ".join(str(a) for a in args)
    _log_buffer.append(msg)
    try:
        _original_print(*args, **kwargs)
    except (UnicodeEncodeError, AttributeError):
        # pythonw may have no usable console, and legacy Windows consoles may
        # reject Chinese/Unicode output. Logging to the web buffer must never
        # be allowed to terminate the server.
        stream = kwargs.get("file", sys.stdout)
        if stream is not None and hasattr(stream, "buffer"):
            safe_text = " ".join(str(a) for a in args)
            stream.buffer.write((safe_text + kwargs.get("end", "\n")).encode("utf-8"))
            stream.flush()

# 重定向所有 print 调用
import builtins
builtins.print = _log_print

# ── ib_insync 兼容性：保证每个 Flask 请求线程都有 asyncio event loop ──
try:
    import asyncio
    import ib_insync.util
    ib_insync.util.startLoop()
except Exception:
    pass

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np

# 导入策略引擎
from data.fetch_data import fetch_audusd, load_data
from ensemble.ensemble_engine import compute_ensemble_signal
from backtest.backtest_engine import vectorized_backtest, event_driven_backtest

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.after_request
def disable_local_dashboard_cache(response):
    """Always serve the current dashboard code after a local app restart."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# 全局状态
live_engine = None
live_thread = None
backtest_cache = None
trade_history = []
ibkr_executor = None
ibkr_refresh_thread = None
active_broker = None
live_trading_active = False
system_status = {
    "live_running": False,
    "ibkr_connected": False,
    "broker": None,
    "broker_environment": None,
    "ibkr_host": "127.0.0.1",
    "ibkr_port": 4002,
    "last_signal": None,
    "current_position": None,
    "entry_price": None,
    "stop_loss": None,
    "take_profit": None,
    "equity": 10000.0,
    "initial_equity": 10000.0,
    "unrealized_pnl": 0.0,
    "realized_pnl": 0.0,
    "drawdown": 0.0,
    "consecutive_losses": 0,
    "positions": [],
    "last_refresh": None,
    "last_trade_time": None,
    "market_price": None,
    "bid": None,
    "ask": None,
    "spread": None,
    "auto_trading": False,
    "position_qty": 0,
    "last_order_status": None,
    "last_order_error": None,
    "account_currency": None,
    "equity_native": None,
    "initial_equity_native": None,
    "available_funds": None,
    "available_funds_native": None,
    "buying_power": None,
    "buying_power_native": None,
    "base_per_usd": None,
    "allocated_capital_native": None,
    "allocated_capital_usd": None,
}


def _update_allocated_capital():
    """Apply this strategy's AUD capital allocation and expose its USD value."""
    import config

    limit_aud = float(getattr(config, "TRADING_CAPITAL_LIMIT_AUD", 50000.0))
    rate = system_status.get("base_per_usd")
    currency = system_status.get("account_currency")
    if currency != "AUD" or not rate or rate <= 0:
        system_status["allocated_capital_native"] = None
        system_status["allocated_capital_usd"] = None
        return None

    limit_usd = limit_aud / float(rate)
    caps = [limit_usd]
    for key in ("equity", "available_funds"):
        value = system_status.get(key)
        if value is not None and float(value) >= 0:
            caps.append(float(value))
    usable_usd = min(caps)
    system_status["allocated_capital_native"] = usable_usd * float(rate)
    system_status["allocated_capital_usd"] = usable_usd
    return usable_usd


# ── 后台 IBKR 账户刷新 ──
def _ibkr_refresh_loop():
    global ibkr_executor, system_status, active_broker
    import config

    account_interval = float(
        getattr(config, "BROKER_ACCOUNT_REFRESH_SECONDS", 30)
    )
    quote_interval = float(
        getattr(config, "BROKER_QUOTE_REFRESH_SECONDS", 15)
    )
    rate_limit_backoff = float(
        getattr(config, "IG_RATE_LIMIT_BACKOFF_SECONDS", 65)
    )
    last_account_refresh = 0.0
    while system_status["ibkr_connected"] and ibkr_executor:
        try:
            now_monotonic = time.monotonic()
            if now_monotonic - last_account_refresh >= account_interval:
                account = ibkr_executor.get_account_snapshot()
                equity = account.get("equity") if account else None
                positions = ibkr_executor.get_positions()
                if active_broker == "ig":
                    system_status["positions"] = positions or []
                    audusd_positions = []
                    for item in positions or []:
                        market = item.get("market") or {}
                        position = item.get("position") or {}
                        if market.get("epic") == getattr(ibkr_executor, "epic", None):
                            audusd_positions.append(position)

                    directions = {
                        str(position.get("direction") or "").upper()
                        for position in audusd_positions
                        if float(position.get("size") or 0) > 0
                    }
                    directions.discard("")
                    if len(directions) == 1:
                        direction = next(iter(directions))
                        units_per_contract = float(
                            getattr(ibkr_executor, "units_per_contract", 10000)
                        )
                        total_units = sum(
                            float(position.get("size") or 0) * units_per_contract
                            for position in audusd_positions
                        )
                        weighted_entry = sum(
                            float(position.get("level") or 0)
                            * float(position.get("size") or 0)
                            * units_per_contract
                            for position in audusd_positions
                        )
                        system_status["current_position"] = (
                            "Long" if direction == "BUY" else "Short"
                        )
                        system_status["position_qty"] = int(round(total_units))
                        system_status["entry_price"] = round(
                            weighted_entry / total_units, 5
                        ) if total_units else None
                        stop_levels = [
                            float(position.get("stopLevel"))
                            for position in audusd_positions
                            if position.get("stopLevel") is not None
                        ]
                        limit_levels = [
                            float(position.get("limitLevel"))
                            for position in audusd_positions
                            if position.get("limitLevel") is not None
                        ]
                        system_status["stop_loss"] = (
                            round(stop_levels[0], 5) if stop_levels else None
                        )
                        system_status["take_profit"] = (
                            round(limit_levels[0], 5) if limit_levels else None
                        )
                    elif not directions:
                        system_status["current_position"] = None
                        system_status["position_qty"] = 0
                        system_status["entry_price"] = None
                        system_status["stop_loss"] = None
                        system_status["take_profit"] = None
                last_account_refresh = now_monotonic

                if equity is not None:
                    system_status["equity"] = equity
                    if system_status["initial_equity"] == 0 or system_status["initial_equity"] is None:
                        system_status["initial_equity"] = equity
                    peak = max(system_status["initial_equity"], equity)
                    system_status["drawdown"] = (peak - equity) / peak if peak > 0 else 0
                    for key in (
                        "account_currency",
                        "equity_native",
                        "available_funds",
                        "available_funds_native",
                        "buying_power",
                        "buying_power_native",
                        "base_per_usd",
                    ):
                        system_status[key] = account.get(key)
                    if system_status.get("initial_equity_native") is None:
                        system_status["initial_equity_native"] = account.get("equity_native")

                    if active_broker == "ig":
                        unrealized_usd = 0.0
                        for item in positions or []:
                            market = item.get("market") or {}
                            position = item.get("position") or {}
                            direction = str(position.get("direction") or "").upper()
                            entry = float(position.get("level") or 0)
                            size = float(position.get("size") or 0)
                            contract_size = float(position.get("contractSize") or 0)
                            close_price = float(
                                market.get("bid") if direction == "BUY"
                                else market.get("offer") or 0
                            )
                            if entry and close_price and size and contract_size:
                                sign = 1 if direction == "BUY" else -1
                                unrealized_usd += sign * (close_price - entry) * size * contract_size
                        system_status["unrealized_pnl"] = unrealized_usd
                    _update_allocated_capital()

            # Use the connected broker's quote first. Yahoo is only a fallback.
            try:
                price = ibkr_executor.get_market_price()
                if price and price > 0:
                    system_status["market_price"] = round(float(price), 5)
            except Exception:
                try:
                    import yfinance as yf
                    t = yf.Ticker("AUDUSD=X")
                    info = t.fast_info
                    price = info.last_price if info and hasattr(info, 'last_price') else None
                    if price and price > 0:
                        system_status["market_price"] = round(float(price), 5)
                except Exception:
                    pass

            system_status["last_refresh"] = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            print(f"[Broker Refresh] 错误: {e}")
            if "exceeded-" in str(e) or "allowance" in str(e):
                print(f"[Broker Refresh] 触发 IG 限流，退避 {int(rate_limit_backoff)} 秒")
                time.sleep(rate_limit_backoff)
                continue
        time.sleep(quote_interval)


def start_ibkr_refresh():
    global ibkr_refresh_thread, system_status
    if system_status["ibkr_connected"]:
        system_status["initial_equity"] = system_status["equity"]
        ibkr_refresh_thread = threading.Thread(target=_ibkr_refresh_loop, daemon=True)
        ibkr_refresh_thread.start()
        print("[Broker] 后台刷新已启动")
        return True
    return False


def stop_ibkr_refresh():
    global system_status
    system_status["ibkr_connected"] = False


# ── 自动交易引擎 ──
auto_trading_thread = None
auto_trading_stop = False


def _auto_trading_loop():
    global ibkr_executor, system_status, trade_history, auto_trading_stop, active_broker
    import config

    last_signal_check = None
    in_position = False
    position_qty = 0
    entry_price = None
    entry_direction = None
    stop_loss_price = None
    take_profit_price = None
    consecutive_losses = 0
    allocated_capital = _update_allocated_capital() or 0.0
    highest_equity = allocated_capital
    last_entry_attempt_bar = None
    signal_refresh_seconds = float(
        getattr(config, "LIVE_SIGNAL_REFRESH_SECONDS", 900)
    )

    print("[AutoTrade] 自动交易引擎已启动")
    system_status["live_running"] = True
    system_status["auto_trading"] = True

    # 从 system_status 恢复持仓（跨 Stop/Start 周期有效）
    if system_status.get("current_position") in ("Long", "Short"):
        entry_direction = 1 if system_status["current_position"] == "Long" else -1
        entry_price = system_status.get("entry_price")
        stop_loss_price = system_status.get("stop_loss")
        take_profit_price = system_status.get("take_profit")
        position_qty = int(system_status.get("position_qty") or 0)
        if position_qty <= 0:
            print("[AutoTrade] 本地持仓缺少有效数量，停止自动交易以避免错误平仓")
            system_status["last_order_error"] = "本地持仓缺少有效数量，请先与 IBKR 持仓核对"
            system_status["auto_trading"] = False
            system_status["live_running"] = False
            return
        in_position = True
        print(f"[AutoTrade] 恢复持仓: {system_status['current_position']} {position_qty} @ {entry_price}")

    while not auto_trading_stop and system_status["ibkr_connected"] and ibkr_executor:
        try:
            now = datetime.now()

            # 日线策略无需每分钟重复计算同一根K线。
            if last_signal_check is None or (now - last_signal_check).total_seconds() > signal_refresh_seconds:
                last_signal_check = now

                try:
                    df = load_data(interval="1d", max_age_hours=12)
                except Exception:
                    # Do not silently fall back to the configured historical
                    # end date in live trading.
                    df = fetch_audusd(start=config.DATA_START, end=None, interval="1d")

                if df is None or len(df) < 50:
                    time.sleep(30)
                    continue

                # A live system must never trade from an old cache. Four
                # calendar days allows normal weekend/holiday gaps.
                latest_bar = pd.Timestamp(df.index[-1])
                if latest_bar.tzinfo is not None:
                    latest_bar = latest_bar.tz_localize(None)
                data_age = pd.Timestamp.now() - latest_bar
                if data_age > pd.Timedelta(days=4):
                    message = f"行情数据过期（最后日线: {latest_bar.date()}），停止生成订单"
                    system_status["last_order_status"] = "blocked_stale_data"
                    system_status["last_order_error"] = message
                    print(f"[AutoTrade] {message}")
                    time.sleep(60)
                    continue

                ensemble = compute_ensemble_signal(df, use_ml=True)
                signal = int(ensemble["ensemble_signal"].iloc[-1])
                strength = ensemble.get("signal_strength", None)
                strength_val = float(strength.iloc[-1]) if strength is not None and len(strength) > 0 else 1.0
                strength_val = max(0.0, min(strength_val, 1.0))
                bar_key = str(df.index[-1])

                system_status["last_signal"] = signal
                print(f"[AutoTrade] 信号: {'多头' if signal > 0 else '空头' if signal < 0 else '空仓'} (强度: {strength_val:.2f})")

                current_direction = entry_direction if in_position else 0

                # 风控
                strategy_equity = (
                    (_update_allocated_capital() or 0.0)
                    + float(system_status.get("realized_pnl") or 0.0)
                    + float(system_status.get("unrealized_pnl") or 0.0)
                )
                if strategy_equity > highest_equity:
                    highest_equity = strategy_equity
                dd = (
                    (highest_equity - strategy_equity) / highest_equity
                    if highest_equity > 0
                    else 0
                )
                system_status["drawdown"] = dd
                dd_abs = abs(dd)

                can_trade = True
                reason = "ok"
                if dd_abs >= config.MAX_DRAWDOWN_LIMIT:
                    can_trade = False
                    reason = f"最大回撤熔断 ({dd_abs:.1%})"
                elif dd_abs >= config.DD_NEAR_LIMIT_LEVEL and not in_position:
                    can_trade = False
                    reason = f"回撤接近上限 ({dd_abs:.1%})"
                if consecutive_losses >= config.MAX_CONSECUTIVE_LOSSES:
                    can_trade = False
                    reason = f"连续亏损 {consecutive_losses} 次"

                # 执行交易
                if can_trade and signal != 0:
                    if signal != current_direction:
                        # 平仓
                        if in_position:
                            print(f"[AutoTrade] 信号反转，平现有 {'多头' if current_direction > 0 else '空头'}")
                            close_action = "SELL" if current_direction > 0 else "BUY"
                            close_ok, close_result = ibkr_executor.place_market_order(close_action, position_qty)
                            if not close_ok:
                                system_status["last_order_status"] = "close_failed"
                                system_status["last_order_error"] = str(close_result)
                                print(f"[AutoTrade] 平仓失败，保留本地持仓并停止本轮反转: {close_result}")
                                continue
                            system_status["current_position"] = None
                            system_status["position_qty"] = 0
                            system_status["stop_loss"] = None
                            system_status["take_profit"] = None
                            system_status["entry_price"] = None
                            system_status["last_order_status"] = "closed"
                            system_status["last_order_error"] = None
                            in_position = False
                            entry_price = None
                            entry_direction = None
                            position_qty = 0
                            time.sleep(2)

                        # 开新仓
                        if not in_position:
                            if last_entry_attempt_bar == bar_key:
                                print(f"[AutoTrade] 本根日线 {bar_key} 已尝试过开仓，跳过重复下单")
                                continue
                            last_entry_attempt_bar = bar_key
                            action = "BUY" if signal > 0 else "SELL"
                            atr_val = 0.005
                            try:
                                high = df["high"]; low = df["low"]; close = df["close"]
                                tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
                                atr_val = float(tr.ewm(alpha=1.0/14, adjust=False).mean().iloc[-1])
                            except Exception:
                                pass
                            current_price = df["close"].iloc[-1]
                            sl = current_price - atr_val * config.ATR_STOP_MULTIPLIER if signal > 0 else current_price + atr_val * config.ATR_STOP_MULTIPLIER
                            tp = current_price + atr_val * config.TAKE_PROFIT_ATR if signal > 0 else current_price - atr_val * config.TAKE_PROFIT_ATR

                            capital = _update_allocated_capital()
                            if not capital or capital <= 0:
                                message = "无法计算本项目的 50,000 AUD 资金配额，禁止下单"
                                system_status["last_order_status"] = "blocked_capital_allocation"
                                system_status["last_order_error"] = message
                                print(f"[AutoTrade] {message}")
                                continue
                            risk_per_unit = atr_val * config.ATR_STOP_MULTIPLIER
                            position_size = 0
                            dd_scale = 1.0
                            max_size = 0
                            if risk_per_unit > 0:
                                if dd_abs >= config.DD_DANGER_LEVEL:
                                    dd_scale = 0.25
                                elif dd_abs >= config.DD_WARNING_LEVEL:
                                    dd_scale = 0.50
                                max_risk = capital * config.POSITION_SIZE_RISK * dd_scale * strength_val
                                position_size = int(max_risk / risk_per_unit)
                                leverage_limit = (
                                    float(getattr(config, "IG_MAX_LEVERAGE", 30))
                                    if active_broker == "ig"
                                    else float(config.MAX_LEVERAGE)
                                )
                                max_size = int(capital * leverage_limit / current_price)
                                buying_power = system_status.get("buying_power")
                                if buying_power and buying_power > 0:
                                    max_size = min(
                                        max_size,
                                        int(float(buying_power) / current_price),
                                    )
                                position_size = min(position_size, max_size)
                            qty = max(position_size, 0)

                            is_ibkr = active_broker == "ibkr"
                            minimum_qty = int(
                                getattr(
                                    config,
                                    "IBKR_MIN_AUD_ORDER" if is_ibkr else "IG_MIN_AUD_ORDER",
                                    20000 if is_ibkr else 1000,
                                )
                            )
                            if qty < minimum_qty:
                                unscaled_risk_limit = (
                                    capital * config.POSITION_SIZE_RISK * dd_scale
                                )
                                minimum_order_risk = minimum_qty * risk_per_unit
                                strength_floor = float(
                                    getattr(config, "MIN_LIVE_SIGNAL_STRENGTH", 0.35)
                                )
                                can_round_up = (
                                    minimum_qty <= max_size
                                    and minimum_order_risk <= unscaled_risk_limit
                                    and strength_val >= strength_floor
                                )
                                if can_round_up:
                                    print(
                                        f"[AutoTrade] 风险仓位 {qty} AUD 向上取整至 "
                                        f"{str(active_broker).upper()} 最小量 {minimum_qty} AUD；"
                                        f"止损风险 ${minimum_order_risk:.2f}，"
                                        f"上限 ${unscaled_risk_limit:.2f}"
                                    )
                                    qty = minimum_qty
                                else:
                                    message = (
                                        f"风险模型仓位 {qty} AUD 低于 {str(active_broker).upper()} 最小量 "
                                        f"{minimum_qty} AUD，且向上取整不满足风险、"
                                        f"购买力或信号强度限制"
                                    )
                                    system_status["last_order_status"] = "blocked_min_size"
                                    system_status["last_order_error"] = message
                                    print(f"[AutoTrade] {message}")
                                    continue

                            print(f"[AutoTrade] 开{'多' if signal > 0 else '空'}仓 qty={qty} price={current_price:.5f} sl={sl:.5f} tp={tp:.5f}")
                            if active_broker == "ig":
                                success, trade_or_msg = ibkr_executor.place_market_order(
                                    action, qty, stop_loss=sl, take_profit=tp
                                )
                            else:
                                success, trade_or_msg = ibkr_executor.place_market_order(action, qty)
                            if success:
                                print(f"[AutoTrade] 下单成功, Order ID: {trade_or_msg.order.orderId}")
                                position_qty = int(
                                    getattr(trade_or_msg.order, "totalQuantity", qty)
                                    or qty
                                )
                                fill_price = float(trade_or_msg.orderStatus.avgFillPrice or current_price)
                                entry_price = fill_price
                                entry_direction = signal
                                entry_time = now
                                calculated_sl = fill_price - atr_val * config.ATR_STOP_MULTIPLIER if signal > 0 else fill_price + atr_val * config.ATR_STOP_MULTIPLIER
                                calculated_tp = fill_price + atr_val * config.TAKE_PROFIT_ATR if signal > 0 else fill_price - atr_val * config.TAKE_PROFIT_ATR
                                stop_loss_price = getattr(trade_or_msg, "stopLevel", None) or calculated_sl
                                take_profit_price = getattr(trade_or_msg, "limitLevel", None) or calculated_tp
                                system_status["stop_loss"] = round(stop_loss_price, 5)
                                system_status["take_profit"] = round(take_profit_price, 5)
                                system_status["entry_price"] = round(fill_price, 5)
                                system_status["current_position"] = "Long" if signal > 0 else "Short"
                                system_status["position_qty"] = position_qty
                                system_status["last_order_status"] = "filled"
                                system_status["last_order_error"] = None
                                in_position = True
                            else:
                                system_status["last_order_status"] = "entry_failed"
                                system_status["last_order_error"] = str(trade_or_msg)
                                print(f"[AutoTrade] 下单失败，本根日线不再重试: {trade_or_msg}")

                elif signal == 0 and in_position:
                    print("[AutoTrade] 信号归零，平仓")
                    close_action = "SELL" if current_direction > 0 else "BUY"
                    close_ok, close_result = ibkr_executor.place_market_order(close_action, position_qty)
                    if close_ok:
                        in_position = False
                        entry_price = None
                        entry_direction = None
                        position_qty = 0
                        stop_loss_price = None
                        take_profit_price = None
                        system_status["current_position"] = None
                        system_status["position_qty"] = 0
                        system_status["entry_price"] = None
                        system_status["stop_loss"] = None
                        system_status["take_profit"] = None
                        system_status["last_order_status"] = "closed"
                        system_status["last_order_error"] = None
                    else:
                        system_status["last_order_status"] = "close_failed"
                        system_status["last_order_error"] = str(close_result)
                        print(f"[AutoTrade] 平仓失败，继续保留持仓: {close_result}")
                    time.sleep(2)

                if not can_trade and not in_position:
                    print(f"[AutoTrade] 风控阻止开仓: {reason}")

            # 止盈止损监控
            if in_position and entry_price and stop_loss_price and take_profit_price:
                try:
                    current_market = system_status.get("market_price", entry_price) or entry_price
                except Exception:
                    current_market = entry_price

                triggered = False
                close_result = None
                if entry_direction == 1 and current_market <= stop_loss_price:
                    print(f"[AutoTrade] 止损触发: {current_market:.5f} <= {stop_loss_price:.5f}")
                    triggered, close_result = ibkr_executor.place_market_order("SELL", position_qty)
                elif entry_direction == -1 and current_market >= stop_loss_price:
                    print(f"[AutoTrade] 止损触发: {current_market:.5f} >= {stop_loss_price:.5f}")
                    triggered, close_result = ibkr_executor.place_market_order("BUY", position_qty)
                elif entry_direction == 1 and current_market >= take_profit_price:
                    print(f"[AutoTrade] 止盈触发: {current_market:.5f} >= {take_profit_price:.5f}")
                    triggered, close_result = ibkr_executor.place_market_order("SELL", position_qty)
                elif entry_direction == -1 and current_market <= take_profit_price:
                    print(f"[AutoTrade] 止盈触发: {current_market:.5f} <= {take_profit_price:.5f}")
                    triggered, close_result = ibkr_executor.place_market_order("BUY", position_qty)

                if triggered:
                    consecutive_losses += 1
                    in_position = False
                    entry_price = None
                    entry_direction = None
                    position_qty = 0
                    stop_loss_price = None
                    take_profit_price = None
                    system_status["current_position"] = None
                    system_status["position_qty"] = 0
                    system_status["entry_price"] = None
                    system_status["stop_loss"] = None
                    system_status["take_profit"] = None
                    system_status["last_order_status"] = "closed"
                    system_status["last_order_error"] = None
                elif close_result is not None:
                    system_status["last_order_status"] = "close_failed"
                    system_status["last_order_error"] = str(close_result)
                    print(f"[AutoTrade] 止盈/止损平仓失败，继续保留持仓: {close_result}")

            time.sleep(30)

        except Exception as e:
            print(f"[AutoTrade] 循环错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

    system_status["auto_trading"] = False
    system_status["live_running"] = False
    print("[AutoTrade] 自动交易引擎已停止")


def start_auto_trading():
    global auto_trading_thread, auto_trading_stop, system_status, active_broker
    if not system_status["ibkr_connected"]:
        return False
    if auto_trading_thread and auto_trading_thread.is_alive():
        return False

    # A web-process restart clears the in-memory position state.  Always rebuild
    # it from IG before starting the strategy, otherwise an existing position can
    # be mistaken for a flat account and the same daily signal opens again.
    if active_broker == "ig":
        try:
            positions = ibkr_executor.get_positions()
            matching = []
            for item in positions or []:
                market = item.get("market") or {}
                position = item.get("position") or {}
                if market.get("epic") == getattr(ibkr_executor, "epic", None):
                    matching.append(position)

            directions = {
                str(position.get("direction") or "").upper()
                for position in matching
                if float(position.get("size") or 0) > 0
            }
            directions.discard("")
            if len(directions) > 1:
                raise RuntimeError("IG AUD/USD 同时存在多仓和空仓，请先手动整理持仓")

            if matching and directions:
                direction = next(iter(directions))
                units_per_contract = float(
                    getattr(ibkr_executor, "units_per_contract", 10000)
                )
                total_units = 0.0
                weighted_level = 0.0
                for position in matching:
                    size = float(position.get("size") or 0)
                    if size <= 0:
                        continue
                    units = size * units_per_contract
                    total_units += units
                    weighted_level += float(position.get("level") or 0) * units

                system_status["current_position"] = (
                    "Long" if direction == "BUY" else "Short"
                )
                system_status["position_qty"] = int(round(total_units))
                system_status["entry_price"] = round(
                    weighted_level / total_units, 5
                ) if total_units else None
                print(
                    f"[AutoTrade] 从 IG 恢复持仓: "
                    f"{system_status['current_position']} "
                    f"{system_status['position_qty']} AUD @ "
                    f"{system_status['entry_price']}"
                )
            else:
                system_status["current_position"] = None
                system_status["position_qty"] = 0
                system_status["entry_price"] = None
        except Exception as exc:
            system_status["last_order_status"] = "position_sync_failed"
            system_status["last_order_error"] = f"启动前同步 IG 持仓失败: {exc}"
            print(f"[AutoTrade] {system_status['last_order_error']}")
            return False

    auto_trading_stop = False
    auto_trading_thread = threading.Thread(target=_auto_trading_loop, daemon=True)
    auto_trading_thread.start()
    print("[AutoTrade] 启动自动交易线程")
    return True


def stop_auto_trading():
    global auto_trading_stop, system_status
    auto_trading_stop = True
    system_status["auto_trading"] = False
    system_status["live_running"] = False
    print("[AutoTrade] 发送停止信号")


# ======================== 帮助函数 ========================
def get_config():
    import config
    return {
        "primary_timeframe": config.PRIMARY_TIMEFRAME,
        "initial_capital": config.INITIAL_CAPITAL,
        "position_size_risk": config.POSITION_SIZE_RISK,
        "atr_stop": config.ATR_STOP_MULTIPLIER,
        "take_profit_atr": config.TAKE_PROFIT_ATR,
        "trailing_stop_atr": config.TRAILING_STOP_ATR,
        "dd_warning": config.DD_WARNING_LEVEL,
        "dd_danger": config.DD_DANGER_LEVEL,
        "dd_near_limit": config.DD_NEAR_LIMIT_LEVEL,
        "max_drawdown": config.MAX_DRAWDOWN_LIMIT,
        "min_consensus": config.MIN_CONSENSUS,
        "weights": config.ENSEMBLE_WEIGHTS,
        "use_200ema": config.USE_200EMA_FILTER,
        "use_strong_trend": config.USE_STRONG_TREND_FILTER,
        "trading_capital_limit_aud": config.TRADING_CAPITAL_LIMIT_AUD,
    }


def run_backtest(interval="1d", train_ml=True):
    import config
    try:
        df = load_data(interval=interval)
    except FileNotFoundError:
        df = fetch_audusd(start=config.DATA_START, end=config.DATA_END, interval=interval)

    use_ml = False
    ml_accuracy = None
    if train_ml:
        try:
            from strategies.ml_model import train_ml_model, build_features
            X, y = build_features(df)
            if len(X) > 500:
                split = int(len(X) * 0.7)
                model, _ = train_ml_model(df)
                if model is not None:
                    use_ml = True
                    from sklearn.metrics import accuracy_score
                    ml_accuracy = float(accuracy_score(y.iloc[split:], model.predict(X.iloc[split:])))
        except Exception:
            pass

    ensemble = compute_ensemble_signal(df, use_ml=use_ml)
    signal = ensemble["ensemble_signal"]
    signal_strength = ensemble.get("signal_strength", None)

    evt = event_driven_backtest(df, signal, initial_capital=config.INITIAL_CAPITAL,
                                  spread=config.BACKTEST_SPREAD, commission=config.BACKTEST_COMMISSION,
                                  slippage=config.BACKTEST_SLIPPAGE, signal_strength=signal_strength)

    bh_ret = df["close"].pct_change().fillna(0)
    bh_eq = (1 + bh_ret).cumprod() * config.INITIAL_CAPITAL
    bh_dd = float((bh_eq - bh_eq.cummax()).min() / bh_eq.cummax().max())

    def make_curve(equity_series):
        dates = [str(d)[:19] for d in equity_series.index[-500:]]
        values = [float(v) for v in equity_series.values[-500:]]
        return {"dates": dates, "values": values}

    def make_trades(trades_list, max_n=20000):
        result = []
        for t in trades_list[-max_n:]:
            result.append({
                "entry_time": str(t.get("entry_time", ""))[:19],
                "exit_time": str(t.get("exit_time", ""))[:19],
                "direction": t.get("direction", "?"),
                "entry_price": round(float(t.get("entry_price", 0)), 5),
                "exit_price": round(float(t.get("exit_price", 0)), 5),
                "pnl": round(float(t.get("pnl", 0)), 2),
                "reason": t.get("reason", "signal"),
                "bars": t.get("bars", 0),
            })
        return result

    return {
        "success": True,
        "data_range": f"{df.index[0]} → {df.index[-1]}",
        "data_count": len(df),
        "ml_accuracy": ml_accuracy,
        "signal_stats": {
            "long_pct": round(float((signal == 1).sum() / len(signal) * 100), 1),
            "short_pct": round(float((signal == -1).sum() / len(signal) * 100), 1),
            "flat_pct": round(float((signal == 0).sum() / len(signal) * 100), 1),
        },
        "vectorized": {
            "stats": {
                "total_return": round(float(evt["stats"].get("total_return", 0)) * 100, 2),
                "cagr": round(float(evt["stats"].get("cagr", 0)) * 100, 2),
                "sharpe": round(float(evt["stats"].get("sharpe_ratio", 0)), 2),
                "max_dd": round(float(evt["stats"].get("max_drawdown", 0)) * 100, 2),
                "win_rate": round(float(evt["stats"].get("win_rate", 0)) * 100, 2),
                "total_trades": evt["stats"].get("total_trades", 0),
                "final_equity": round(float(evt["stats"].get("final_equity", 0)), 2),
                "avg_win": round(float(evt["stats"].get("avg_win", 0)), 2),
                "avg_loss": round(float(evt["stats"].get("avg_loss", 0)), 2),
                "profit_factor": round(float(evt["stats"].get("profit_factor", 0)), 2),
                "calmar": round(float(evt["stats"].get("calmar_ratio", 0)), 2),
            },
            "equity_curve": make_curve(evt["equity_curve"]),
            "trades": make_trades(evt["trades"]),
        },
        "benchmark": {
            "total_return": round(float((bh_eq.iloc[-1] / config.INITIAL_CAPITAL - 1) * 100), 2),
            "max_dd": round(float(bh_dd * 100), 2),
        },
    }


# ======================== API 路由 ========================
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/config")
def api_config():
    return jsonify(get_config())

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    global backtest_cache
    data = request.get_json() or {}
    interval = data.get("interval", "1d")
    train_ml = data.get("train_ml", True)
    try:
        backtest_cache = run_backtest(interval=interval, train_ml=train_ml)
        return jsonify(backtest_cache)
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})

@app.route("/api/backtest", methods=["GET"])
def api_backtest_get():
    global backtest_cache
    if backtest_cache is None:
        return jsonify({"success": False, "error": "请先执行回测"})
    return jsonify(backtest_cache)

@app.route("/api/status")
def api_status():
    global system_status
    return jsonify(system_status)

@app.route("/api/trade-history")
def api_trade_history():
    global trade_history, ibkr_executor, active_broker
    if active_broker == "ig" and ibkr_executor and system_status.get("ibkr_connected"):
        try:
            return jsonify(ibkr_executor.get_trade_history(days=90, limit=100))
        except Exception as exc:
            return jsonify({"error": f"读取 IG 交易历史失败: {exc}"}), 502
    return jsonify(trade_history[-50:])

@app.route("/api/ibkr/connect", methods=["POST"])
def api_ibkr_connect():
    global ibkr_executor, system_status, active_broker
    try:
        from execution.ibkr_executor import IBKRExecutor
        data = request.get_json() or {}
        host = data.get("host", system_status["ibkr_host"])
        port = int(data.get("port", system_status["ibkr_port"]))
        client_id = int(data.get("client_id", 2))
        system_status["ibkr_host"] = host
        system_status["ibkr_port"] = port
        ibkr_executor = IBKRExecutor(host=host, port=port, client_id=client_id)
        success, msg, info = ibkr_executor.connect()
        if success:
            active_broker = "ibkr"
            system_status["ibkr_connected"] = True
            system_status["broker"] = "IBKR"
            system_status["broker_environment"] = (
                "paper" if port in (7497, 4002) else "live"
            )
            if info.get("equity"):
                system_status["equity"] = info["equity"]
                system_status["initial_equity"] = info["equity"]
            for key in (
                "account_currency",
                "equity_native",
                "available_funds",
                "available_funds_native",
                "buying_power",
                "buying_power_native",
                "base_per_usd",
            ):
                system_status[key] = info.get(key)
            system_status["initial_equity_native"] = info.get("equity_native")
            _update_allocated_capital()
            start_ibkr_refresh()
            return jsonify({"success": True, "message": msg, "account_id": info.get("account_id"), "equity": system_status["equity"]})
        else:
            return jsonify({"success": False, "error": msg})
    except ImportError:
        return jsonify({"success": False, "error": "ib_insync 未安装"})
    except Exception as e:
        import traceback
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()})


@app.route("/api/ig/connect", methods=["POST"])
def api_ig_connect():
    global ibkr_executor, system_status, active_broker
    data = request.get_json() or {}
    environment = str(data.get("environment", "demo")).lower()
    if environment not in ("demo", "live"):
        return jsonify({"success": False, "error": "environment must be demo or live"})
    try:
        from execution.ig_executor import IGExecutor

        if system_status.get("auto_trading"):
            return jsonify({
                "success": False,
                "error": "请先停止自动交易，再切换经纪商",
            })
        if ibkr_executor:
            try:
                ibkr_executor.disconnect()
            except Exception:
                pass
        system_status["ibkr_connected"] = False
        executor = IGExecutor(environment=environment)
        success, msg, info = executor.connect()
        if not success:
            return jsonify({"success": False, "error": msg})

        ibkr_executor = executor
        active_broker = "ig"
        system_status["ibkr_connected"] = True
        system_status["broker"] = "IG"
        system_status["broker_environment"] = environment
        if info.get("equity") is not None:
            system_status["equity"] = info["equity"]
            system_status["initial_equity"] = info["equity"]
        for key in (
            "account_currency",
            "equity_native",
            "available_funds",
            "available_funds_native",
            "buying_power",
            "buying_power_native",
            "base_per_usd",
        ):
            system_status[key] = info.get(key)
        system_status["initial_equity_native"] = info.get("equity_native")
        _update_allocated_capital()
        start_ibkr_refresh()
        print(f"[IG] 已连接 {environment.upper()}，账户 {info.get('account_id')}")
        return jsonify({
            "success": True,
            "message": msg,
            "account_id": info.get("account_id"),
            "equity": system_status["equity"],
            "broker": "IG",
            "environment": environment,
        })
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        })

@app.route("/api/logs")
def api_logs():
    return jsonify(list(_log_buffer)[-200:])

@app.route("/api/ibkr/disconnect", methods=["POST"])
def api_ibkr_disconnect():
    global ibkr_executor, system_status, active_broker
    try:
        if ibkr_executor:
            ibkr_executor.disconnect()
            ibkr_executor = None
        system_status["ibkr_connected"] = False
        system_status["broker"] = None
        system_status["broker_environment"] = None
        active_broker = None
        return jsonify({"success": True, "message": "已断开"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/ibkr/auto-trading", methods=["POST"])
def api_toggle_auto_trading():
    global system_status
    data = request.get_json() or {}
    enabled = data.get("enabled", False)
    if enabled:
        if start_auto_trading():
            return jsonify({"success": True, "auto_trading": True, "message": "自动交易已开启"})
        else:
            return jsonify({"success": False, "error": "无法启动，请先连接交易账户"})
    else:
        stop_auto_trading()
        return jsonify({"success": True, "auto_trading": False, "message": "自动交易已关闭"})

@app.route("/api/config/update", methods=["POST"])
def api_config_update():
    import config
    data = request.get_json() or {}
    allowed = ["ATR_STOP_MULTIPLIER", "TAKE_PROFIT_ATR", "TRAILING_STOP_ATR", "TRAILING_STOP_ACTIVATION", "POSITION_SIZE_RISK", "MIN_CONSENSUS", "DD_WARNING_LEVEL", "DD_DANGER_LEVEL"]
    changed = [k for k, v in data.items() if k in allowed and hasattr(config, k)]
    for k in changed:
        setattr(config, k, data[k])
    return jsonify({"success": True, "changed": changed})


if __name__ == "__main__":
    print("""    ╔══════════════════════════════════════════╗
    ║  AUD/USD 量化交易系统 v9 — Web 面板     ║
    ║  http://localhost:6060                   ║
    ╚══════════════════════════════════════════╝""")
    app.run(host="0.0.0.0", port=6060, debug=False)
