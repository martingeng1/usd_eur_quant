"""
AUD/USD 量化交易系统 v9 — 实盘引擎 (ML + 日线)
=============================================
使用方式：
    python live_trading.py --paper    # 模拟盘（推荐先跑）
    python live_trading.py --live     # 实盘（确认无误后再用）

前置条件：
    1. 安装 IB Gateway（推荐）或 TWS，启用 API 连接
    2. pip install ib_insync xgboost pandas numpy matplotlib
    3. 确保 strategies/xgboost_audusd.pkl 已训练（先跑 python main.py --train-ml --interval 1d）
"""

import sys
import os
import argparse
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import config_live as cfg
from strategies.trend_following import compute_trend_signal, get_trend_confidence
from strategies.mean_reversion import compute_mean_reversion_signal, get_reversion_confidence
from strategies.momentum import compute_momentum_signal, get_momentum_confidence
from ensemble.ensemble_engine import compute_ensemble_signal


class LiveTradingEngineV9:
    """
    v9 实盘交易引擎 — 日线 ML 驱动

    流程：
    1. 连接 IBKR → 获取账户信息
    2. 加载历史数据 → 训练/更新 ML 模型
    3. 每日收盘后：获取最新日线 K 线 → 生成信号 → 开仓/平仓
    4. 盘中每 N 小时：监控止损/止盈/保本止损（不生成新信号）
    """

    def __init__(self, paper_mode=True):
        self.paper_mode = paper_mode
        self.ib = None
        self.contract = None
        self.connected = False
        self.account_id = None
        self.df = None              # 完整历史数据缓存
        self.signal_cache = 0       # 上一次计算的信号（日线级别）
        self.signal_strength = 1.0
        self.last_signal_day = None # 上次生成信号的日期

        # 持仓状态
        self.in_position = False
        self.entry_price = 0.0
        self.entry_direction = 0
        self.entry_bar = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_activated = False
        self.breakeven_activated = False
        self.best_price = 0.0
        self.partial_tp_triggered = False
        self.position_size = 0
        self.trade_history = []

        # 风控状态
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.daily_pnl = 0.0
        self.last_day = None
        self.peak_equity = cfg.INITIAL_CAPITAL_USD

        # 连接
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_wait_seconds = 15

        # ML 重训计时
        self.last_ml_train = None

    # ======================== IBKR 连接 ========================
    def connect_ibkr(self):
        """连接到 IB Gateway 或 TWS"""
        try:
            from ib_insync import IB, Forex

            try:
                if self.ib and self.ib.isConnected():
                    self.ib.disconnect()
            except Exception:
                pass

            self.ib = IB()
            host = cfg.IBKR_HOST
            port = cfg.IBKR_PORT

            print(f"[IBKR] 正在连接到 {host}:{port} ...")
            self.ib.connect(host, port, clientId=cfg.IBKR_CLIENT_ID, timeout=30)

            self.contract = Forex("AUDUSD")
            self.ib.qualifyContracts(self.contract)

            accounts = self.ib.managedAccounts()
            self.account_id = accounts[0] if accounts else None

            self.connected = True
            self.reconnect_attempts = 0
            print(f"[IBKR] 连接成功！账户: {self.account_id}")
            print(f"[IBKR] 模式: {'模拟盘' if self.paper_mode else '⚠️ 实盘'}")

            self.print_account_info()
            return True

        except ImportError:
            print("[错误] ib_insync 未安装: pip install ib_insync")
            return False
        except Exception as e:
            self.connected = False
            print(f"[错误] 连接 IBKR 失败: {e}")
            print("       请确认 IB Gateway 或 TWS 已启动并启用了 API 连接")
            return False

    def is_connected(self):
        try:
            return bool(self.ib and self.ib.isConnected())
        except Exception:
            return False

    def ensure_connected(self):
        if self.is_connected():
            self.connected = True
            return True

        self.connected = False
        self.reconnect_attempts += 1
        print(f"[IBKR] 连接断开，重连 {self.reconnect_attempts}/{self.max_reconnect_attempts} ...")

        try:
            if self.ib:
                self.ib.disconnect()
        except Exception:
            pass

        if self.reconnect_attempts > self.max_reconnect_attempts:
            print("[IBKR] 重连次数过多，暂停60秒")
            time.sleep(60)
            self.reconnect_attempts = 0
            return False

        time.sleep(self.reconnect_wait_seconds)
        return self.connect_ibkr()

    def safe_sleep(self, seconds):
        try:
            if self.is_connected():
                self.ib.sleep(seconds)
            else:
                time.sleep(seconds)
        except Exception:
            self.connected = False
            time.sleep(seconds)

    def disconnect(self):
        if self.ib:
            try:
                if self.is_connected():
                    self.cancel_all_orders()
                    self.ib.disconnect()
            except Exception:
                pass
            finally:
                self.connected = False
                print("[IBKR] 已断开连接")

    def print_account_info(self):
        try:
            summary = self.ib.accountSummary()
            print("\n  ╔══════════════════════════════╗")
            print("  ║      账户摘要                ║")
            print("  ╠══════════════════════════════╣")
            for s in summary:
                if s.tag in ("NetLiquidation", "AvailableFunds", "BuyingPower"):
                    print(f"  ║  {s.tag:<20} {float(s.value):>10,.2f} {s.currency:<3} ║")
            print("  ╚══════════════════════════════╝\n")
        except Exception as e:
            print(f"  无法获取账户信息: {e}")

    # ======================== 数据获取 ========================
    def get_current_equity(self):
        """获取当前账户净值"""
        try:
            if not self.ensure_connected():
                return None
            summary = self.ib.accountSummary()
            for s in summary:
                if s.tag == "NetLiquidation":
                    return float(s.value)
        except Exception:
            pass
        return None

    def fetch_daily_data(self, days_back=500):
        """从 IBKR 获取日线数据"""
        from ib_insync import util

        if not self.ensure_connected():
            print("[数据] IBKR 未连接，使用 Yahoo Finance 备用")
            return self._fallback_yahoo()

        duration_str = f"{days_back} D"
        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
        )

        df = util.df(bars)
        if df is None or df.empty:
            print("[警告] IBKR 未返回数据，使用 Yahoo Finance 备用")
            return self._fallback_yahoo()

        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.rename(columns={
            "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        }, inplace=True)

        print(f"[数据] 从 IBKR 获取 {len(df)} 条日线")
        return df[["open", "high", "low", "close", "volume"]]

    def get_market_price(self):
        """获取 AUD/USD 最新价格"""
        if not self.ensure_connected():
            return None

        ticker = self.ib.reqMktData(self.contract, "", False, False)
        self.safe_sleep(2)

        price = ticker.last or ticker.close or ticker.marketPrice()
        if not price or price <= 0:
            bid, ask = ticker.bid, ticker.ask
            if bid and ask and bid > 0 and ask > 0:
                price = (bid + ask) / 2

        try:
            self.ib.cancelMktData(self.contract)
        except Exception:
            pass

        return float(price) if price and price > 0 else None

    def _fallback_yahoo(self):
        """备用：从 Yahoo Finance 获取日线数据"""
        print("[数据] 使用 Yahoo Finance 备用数据源")
        from data.fetch_data import fetch_audusd
        start = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
        return fetch_audusd(start=start, interval="1d", save_csv=False)

    # ======================== 信号生成 ========================
    def update_ml_model(self):
        """检查是否需要重训练 ML 模型"""
        now = datetime.now()
        if self.last_ml_train is not None:
            days_since = (now - self.last_ml_train).days
            if days_since < cfg.RETRAIN_INTERVAL_DAYS:
                return

        if self.df is None or len(self.df) < 500:
            return

        print("[ML] 重训练 XGBoost 模型...")
        try:
            from strategies.ml_model import train_ml_model
            train_ml_model(self.df, retrain=True)
            self.last_ml_train = now
            print(f"[ML] 模型已更新 ({now.strftime('%Y-%m-%d %H:%M')})")
        except Exception as e:
            print(f"[ML] 训练失败: {e}")

    def generate_signal(self, df):
        """根据日线数据生成交易信号"""
        if len(df) < 200:
            print(f"[信号] 数据不足 ({len(df)} 条)，需要至少 200 条日线")
            return 0, {}

        try:
            ensemble = compute_ensemble_signal(df, use_ml=True)
            signal = ensemble["ensemble_signal"].iloc[-1]
            strength = ensemble["signal_strength"].iloc[-1]
            score = ensemble["ensemble_score"].iloc[-1]
            ml_loaded = ensemble.get("ml_loaded", False)

            now = datetime.now()
            print(f"[信号] 日线 {now.strftime('%Y-%m-%d')} → 信号: {signal:+.0f} "
                  f"(得分: {score:+.3f}, 强度: {strength:.3f}, ML: {'✓' if ml_loaded else '✗'})")
            return int(signal), ensemble
        except Exception as e:
            print(f"[信号] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return 0, {}

    # ======================== 风控检查 ========================
    def get_drawdown(self):
        equity = self.get_current_equity()
        if equity is None:
            equity = self.peak_equity
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity <= 0:
            return 0.0
        return (equity - self.peak_equity) / self.peak_equity

    def get_position_scaler(self):
        """阶梯式回撤仓位缩放"""
        dd = abs(self.get_drawdown())
        if dd < cfg.DD_WARNING_LEVEL:
            return 1.0
        elif dd < cfg.DD_DANGER_LEVEL:
            return 0.50
        elif dd < cfg.DD_NEAR_LIMIT_LEVEL:
            return 0.25
        else:
            return 0.0

    def can_open_new_trade(self):
        """检查是否可以开新仓"""
        if self.in_position:
            return False, "已有持仓"
        if self.consecutive_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            if self.cooldown_until and datetime.now() < self.cooldown_until:
                return False, f"冷静期中，直到 {self.cooldown_until.strftime('%Y-%m-%d')}"
        dd = self.get_drawdown()
        if abs(dd) >= cfg.MAX_DRAWDOWN_LIMIT:
            return False, f"触发最大回撤 ({abs(dd):.1%})"
        if abs(dd) >= cfg.DD_NEAR_LIMIT_LEVEL:
            return False, f"回撤接近上限 ({abs(dd):.1%})"
        return True, "ok"

    # ======================== ATR 计算 ========================
    def _get_latest_atr(self, period=14):
        if self.df is None or len(self.df) < period:
            return 0.005
        from risk.risk_manager import compute_atr
        atr = compute_atr(self.df, period)
        return float(atr.iloc[-1]) if len(atr) > 0 else 0.005

    # ======================== 持仓管理 ========================
    def update_trailing_stop(self, current_high, current_low, current_atr):
        """更新移动止损（v9：含保本止损）"""
        if not self.in_position:
            return

        trailing_dist = cfg.TRAILING_STOP_ATR * current_atr

        if self.entry_direction == 1:  # 多头
            if current_high > self.best_price:
                self.best_price = current_high
            profit = self.best_price - self.entry_price

            # 保本止损
            if not self.breakeven_activated and profit >= cfg.BREAKEVEN_STOP_ACTIVATION * current_atr:
                self.breakeven_activated = True
                self.stop_loss = max(self.stop_loss, self.entry_price)
                print(f"[保本止损] 已激活，止损移至入场价 {self.entry_price:.5f}")

            # 移动止损
            if not self.trailing_activated and profit >= cfg.TRAILING_STOP_ACTIVATION * current_atr:
                self.trailing_activated = True
                print(f"[移动止损] 已激活 (盈利 {profit:.5f})")

            if self.trailing_activated:
                new_sl = self.best_price - trailing_dist
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl

        else:  # 空头
            if current_low < self.best_price:
                self.best_price = current_low
            profit = self.entry_price - self.best_price

            if not self.breakeven_activated and profit >= cfg.BREAKEVEN_STOP_ACTIVATION * current_atr:
                self.breakeven_activated = True
                self.stop_loss = min(self.stop_loss, self.entry_price)
                print(f"[保本止损] 已激活，止损移至入场价 {self.entry_price:.5f}")

            if not self.trailing_activated and profit >= cfg.TRAILING_STOP_ACTIVATION * current_atr:
                self.trailing_activated = True
                print(f"[移动止损] 已激活 (盈利 {profit:.5f})")

            if self.trailing_activated:
                new_sl = self.best_price + trailing_dist
                if new_sl < self.stop_loss:
                    self.stop_loss = new_sl

    def check_partial_tp(self, current_high, current_low, current_atr):
        """检查部分止盈"""
        if self.partial_tp_triggered or not self.in_position:
            return False, None

        if self.entry_direction == 1:
            tp_price = self.entry_price + cfg.PARTIAL_TP_ATR * current_atr
            if current_high >= tp_price:
                self.partial_tp_triggered = True
                return True, tp_price
        else:
            tp_price = self.entry_price - cfg.PARTIAL_TP_ATR * current_atr
            if current_low <= tp_price:
                self.partial_tp_triggered = True
                return True, tp_price

        return False, None

    # ======================== 交易执行 ========================
    def place_trade(self, direction, entry_price, atr_value, signal_strength=1.0):
        """开仓"""
        ok, reason = self.can_open_new_trade()
        if not ok:
            print(f"[风控] 拒绝开仓: {reason}")
            return

        dd_scale = self.get_position_scaler()
        adjusted_risk = cfg.POSITION_SIZE_RISK * dd_scale * signal_strength

        if direction == 1:
            stop_loss = entry_price - atr_value * cfg.ATR_STOP_MULTIPLIER
            take_profit = entry_price + atr_value * cfg.TAKE_PROFIT_ATR
        else:
            stop_loss = entry_price + atr_value * cfg.ATR_STOP_MULTIPLIER
            take_profit = entry_price - atr_value * cfg.TAKE_PROFIT_ATR

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return

        capital = cfg.INITIAL_CAPITAL_USD
        equity_now = self.get_current_equity()
        if equity_now is not None and equity_now > 0:
            capital = equity_now

        position_size = capital * adjusted_risk / risk_per_unit
        max_position = capital * cfg.MAX_LEVERAGE / entry_price
        position_size = min(position_size, max_position)
        position_size = max(position_size, cfg.MIN_POSITION_USD)
        position_size = round(position_size / 1000) * 1000
        position_size = int(position_size)

        if position_size <= 0:
            print("[交易] 仓位为0，跳过")
            return

        action = "BUY" if direction == 1 else "SELL"

        try:
            from ib_insync import MarketOrder, StopOrder

            order = MarketOrder(action=action, totalQuantity=position_size)
            trade = self.ib.placeOrder(self.contract, order)

            print(f"\n{'='*60}")
            print(f"  📊 开仓: {'多' if direction == 1 else '空'} {position_size} 单位 AUD/USD")
            print(f"  入场: {entry_price:.5f} | 止损: {stop_loss:.5f} | 止盈: {take_profit:.5f}")
            print(f"  风险: ${risk_per_unit * position_size:,.0f} | 仓位缩放: {dd_scale:.0%} | 信号强度: {signal_strength:.2f}")
            print(f"{'='*60}\n")

            self.in_position = True
            self.entry_price = entry_price
            self.entry_direction = direction
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.trailing_activated = False
            self.breakeven_activated = False
            self.best_price = entry_price
            self.partial_tp_triggered = False
            self.position_size = position_size
            from datetime import datetime
            self.entry_time = datetime.now()

            # 止损单
            sl_action = "SELL" if direction == 1 else "BUY"
            sl_order = StopOrder(action=sl_action, totalQuantity=position_size, stopPrice=stop_loss)
            self.ib.placeOrder(self.contract, sl_order)

            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "action": action,
                "quantity": position_size,
                "entry": float(entry_price),
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "dd_scale": dd_scale,
                "signal_strength": signal_strength,
            })

        except Exception as e:
            print(f"[错误] 开仓失败: {e}")

    def close_position(self, price, reason="signal"):
        """平仓"""
        if not self.in_position:
            return

        action = "SELL" if self.entry_direction == 1 else "BUY"
        quantity = self.position_size

        try:
            from ib_insync import MarketOrder

            self.cancel_all_orders()

            order = MarketOrder(action=action, totalQuantity=quantity)
            self.ib.placeOrder(self.contract, order)

            pnl = (price - self.entry_price) * self.entry_direction * quantity

            print(f"\n{'='*60}")
            print(f"  📊 平仓: {'多' if self.entry_direction == 1 else '空'} → 平仓")
            print(f"  出场: {price:.5f} | 盈亏: ${pnl:+.2f}")
            print(f"  原因: {reason}")
            print(f"{'='*60}\n")

            # 连亏追踪
            if pnl <= 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
                    self.cooldown_until = datetime.now() + timedelta(days=cfg.COOLDOWN_DAYS)
                    print(f"[风控] 连续亏损{self.consecutive_losses}笔，冷静期至 {self.cooldown_until.strftime('%Y-%m-%d')}")
            else:
                self.consecutive_losses = 0
                self.cooldown_until = None

            self.in_position = False
            self.entry_price = 0.0
            self.entry_direction = 0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.trailing_activated = False
            self.breakeven_activated = False
            self.best_price = 0.0
            self.partial_tp_triggered = False
            self.position_size = 0

            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "action": f"CLOSE_{action}",
                "quantity": quantity,
                "exit": float(price),
                "pnl": float(pnl),
                "reason": reason,
            })

        except Exception as e:
            print(f"[错误] 平仓失败: {e}")

    def handle_partial_close(self, tp_price):
        """处理部分止盈（平仓50%仓位）"""
        if not self.in_position:
            return

        quantity = int(self.position_size * cfg.PARTIAL_TP_RATIO)
        if quantity < cfg.MIN_POSITION_USD:
            return

        action = "SELL" if self.entry_direction == 1 else "BUY"
        try:
            from ib_insync import MarketOrder
            order = MarketOrder(action=action, totalQuantity=quantity)
            self.ib.placeOrder(self.contract, order)

            pnl = (tp_price - self.entry_price) * self.entry_direction * quantity
            self.position_size -= quantity

            print(f"\n{'='*60}")
            print(f"  📊 部分止盈: 平仓 {quantity} 单位 (50%)")
            print(f"  价格: {tp_price:.5f} | 盈亏: ${pnl:+.2f}")
            print(f"  剩余仓位: {self.position_size} 单位")
            print(f"{'='*60}\n")

            # 更新止损单
            self.cancel_all_orders()
            from ib_insync import StopOrder
            sl_action = "SELL" if self.entry_direction == 1 else "BUY"
            sl_order = StopOrder(action=sl_action, totalQuantity=self.position_size, stopPrice=self.stop_loss)
            self.ib.placeOrder(self.contract, sl_order)

            # 盈利了重置连亏
            if pnl > 0:
                self.consecutive_losses = 0
                self.cooldown_until = None

        except Exception as e:
            print(f"[错误] 部分止盈失败: {e}")

    def cancel_all_orders(self):
        if not self.connected:
            return
        try:
            trades = self.ib.trades()
            for t in trades:
                self.ib.cancelOrder(t.order)
        except Exception:
            pass

    # ======================== 主循环 ========================
    def run(self):
        print("\n" + "=" * 60)
        print("  AUD/USD 量化交易系统 v9 — 实盘引擎 (ML + 日线)")
        print(f"  配置: {cfg.INITIAL_CAPITAL_AUD} AUD / ~{cfg.INITIAL_CAPITAL_USD} USD")
        print(f"  模式: {'模拟盘' if self.paper_mode else '⚠️ 实盘'}")
        print(f"  时间框架: 日线 | 信号检查: 每{cfg.CHECK_INTERVAL_SECONDS//3600}小时")
        print(f"  ML 重训: 每{cfg.RETRAIN_INTERVAL_DAYS}天")
        print("=" * 60 + "\n")

        # 1. 连接
        if not self.connect_ibkr():
            return

        # 2. 获取历史数据
        print("[步骤1] 获取历史日线数据...")
        self.df = self.fetch_daily_data(days_back=730)  # 2年数据
        if self.df is None or len(self.df) < 200:
            print("[错误] 无法获取足够日线数据")
            return

        print(f"[数据] 已加载 {len(self.df)} 条日线 ({self.df.index[0].strftime('%Y-%m-%d')} → {self.df.index[-1].strftime('%Y-%m-%d')})")

        # 3. 训练 ML
        print("[步骤2] 训练 ML 模型...")
        try:
            from strategies.ml_model import train_ml_model
            train_ml_model(self.df, retrain=True)
            self.last_ml_train = datetime.now()
            print("[ML] 模型就绪")
        except Exception as e:
            print(f"[ML] 训练失败: {e}")
            print("[ML] 将使用已有模型（需确认 strategies/xgboost_audusd.pkl 存在）")

        # 4. 初始信号
        print("[步骤3] 生成初始信号...")
        signal, ensemble = self.generate_signal(self.df)
        self.signal_cache = signal
        self.signal_strength = ensemble.get("signal_strength", pd.Series([1.0], index=self.df.index)).iloc[-1]
        self.last_signal_day = datetime.now().date()

        # 5. 检查现有持仓
        self.check_existing_position()

        print(f"\n[启动] 进入监控模式")
        print(f"  信号检查间隔: {cfg.CHECK_INTERVAL_SECONDS//3600}小时")
        print(f"  当前信号: {self.signal_cache:+d}")
        print(f"  当前持仓: {'多' if self.entry_direction == 1 else '空' if self.entry_direction == -1 else '无'}\n")

        try:
            while True:
                try:
                    now = datetime.now()
                    print(f"\n{'─'*60}")
                    print(f"[{now.strftime('%Y-%m-%d %H:%M')}] 检查...")

                    # 获取最新价格
                    price = self.get_market_price()
                    if not price or price <= 0:
                        print("[跳过] 无有效价格")
                        self.safe_sleep(cfg.CHECK_INTERVAL_SECONDS)
                        continue

                    # 计算 ATR
                    atr_val = self._get_latest_atr()

                    # ---------- 持仓管理 ----------
                    if self.in_position:
                        # 部分止盈检查
                        partial_tp, tp_price = self.check_partial_tp(
                            float(self.df["high"].iloc[-1]),
                            float(self.df["low"].iloc[-1]),
                            atr_val
                        )
                        if partial_tp:
                            self.handle_partial_close(tp_price)

                        # 更新移动止损
                        self.update_trailing_stop(
                            float(self.df["high"].iloc[-1]),
                            float(self.df["low"].iloc[-1]),
                            atr_val
                        )

                        # 止损/止盈检查
                        if self.entry_direction == 1:
                            if float(self.df["low"].iloc[-1]) <= self.stop_loss:
                                self.close_position(self.stop_loss, "stop_loss")
                            elif float(self.df["high"].iloc[-1]) >= self.take_profit:
                                self.close_position(self.take_profit, "take_profit")
                        else:
                            if float(self.df["high"].iloc[-1]) >= self.stop_loss:
                                self.close_position(self.stop_loss, "stop_loss")
                            elif float(self.df["low"].iloc[-1]) <= self.take_profit:
                                self.close_position(self.take_profit, "take_profit")

                    # ---------- 日线信号更新（每日一次）----------
                    today = now.date()
                    if today != self.last_signal_day:
                        # 更新时间数据
                        print("[数据] 更新日线数据...")
                        self.df = self.fetch_daily_data(days_back=730)
                        if self.df is not None and len(self.df) >= 200:
                            # ML 重训练
                            self.update_ml_model()
                            # 生成新信号
                            self.signal_cache, ensemble = self.generate_signal(self.df)
                            self.signal_strength = ensemble.get("signal_strength", pd.Series([1.0], index=self.df.index)).iloc[-1]
                            self.last_signal_day = today

                    # ---------- 交易决策 ----------
                    if not self.in_position and self.signal_cache != 0:
                        self.place_trade(
                            self.signal_cache,
                            float(price),
                            atr_val,
                            signal_strength=self.signal_strength
                        )
                    elif self.in_position and self.signal_cache == 0:
                        self.close_position(float(price), "signal_flat")
                    elif self.in_position and self.signal_cache != 0 and self.signal_cache != self.entry_direction:
                        # 反转持仓
                        self.close_position(float(price), "signal_reverse")
                        self.place_trade(self.signal_cache, float(price), atr_val, signal_strength=self.signal_strength)

                    # 风控状态报告
                    dd = self.get_drawdown()
                    print(f"[风控] 当前回撤: {dd:.2%} | 连亏: {self.consecutive_losses}/{cfg.MAX_CONSECUTIVE_LOSSES}")
                    print(f"[持仓] {'多' if self.entry_direction == 1 else '空' if self.entry_direction == -1 else '无'} | "
                          f"信号: {self.signal_cache:+d} | 价格: {price:.5f}")

                    # 等待下一次检查
                    next_check = now + timedelta(seconds=cfg.CHECK_INTERVAL_SECONDS)
                    print(f"[下次检查] {next_check.strftime('%H:%M:%S')}")
                    self.safe_sleep(cfg.CHECK_INTERVAL_SECONDS)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[警告] 本轮检查出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.connected = False
                    self.ensure_connected()
                    self.safe_sleep(min(cfg.CHECK_INTERVAL_SECONDS, 300))

        except KeyboardInterrupt:
            print("\n[退出] 用户中断")
        finally:
            print("[清理] 平仓并断开...")
            if self.in_position and self.ensure_connected():
                last_price = self.get_market_price()
                if last_price and last_price > 0:
                    self.close_position(float(last_price), "exit")
            self.disconnect()
            # 保存交易记录
            self.save_trade_history()

    def check_existing_position(self):
        """检查是否有现有持仓（比如之前会话遗留的）"""
        try:
            if not self.ensure_connected():
                return
            positions = self.ib.positions()
            for p in positions:
                if p.contract.symbol == "AUD" and p.contract.currency == "USD":
                    pos = int(p.position)
                    if pos != 0:
                        self.in_position = True
                        self.entry_direction = 1 if pos > 0 else -1
                        self.position_size = abs(pos)
                        self.entry_price = float(p.avgCost)
                        price = self.get_market_price()
                        if price:
                            self.best_price = float(price)
                        atr = self._get_latest_atr()
                        if self.entry_direction == 1:
                            self.stop_loss = self.entry_price - atr * cfg.ATR_STOP_MULTIPLIER
                            self.take_profit = self.entry_price + atr * cfg.TAKE_PROFIT_ATR
                        else:
                            self.stop_loss = self.entry_price + atr * cfg.ATR_STOP_MULTIPLIER
                            self.take_profit = self.entry_price - atr * cfg.TAKE_PROFIT_ATR
                        print(f"[持仓] 检测到现有持仓: {'多' if pos > 0 else '空'} {abs(pos)} 单位")
                    break
        except Exception as e:
            print(f"[持仓] 检查失败: {e}")

    def save_trade_history(self):
        """保存交易历史"""
        if not self.trade_history:
            return
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "trade_history_live.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.trade_history, f, indent=2, default=str)
        print(f"[记录] 交易历史已保存到 {path}")


def main():
    parser = argparse.ArgumentParser(description="AUD/USD 量化交易系统 v9 实盘引擎")
    parser.add_argument("--paper", action="store_true", default=True, help="模拟盘模式（默认）")
    parser.add_argument("--live", action="store_true", help="实盘模式（小心！）")
    args = parser.parse_args()

    if args.live:
        print("\n⚠️  实盘模式确认 ⚠️")
        print("   这将使用您的真实 IBKR 账户资金进行交易！")
        confirm = input("   输入 YES 确认: ")
        if confirm != "YES":
            print("   已取消")
            return
        mode = False
    else:
        print("\n  模拟盘模式（安全，不涉及真实资金）")
        mode = True

    engine = LiveTradingEngineV9(paper_mode=mode)
    engine.run()


if __name__ == "__main__":
    main()