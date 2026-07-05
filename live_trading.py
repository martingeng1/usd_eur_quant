"""
USD/EUR 量化交易系统 — 实盘/模拟盘交易脚本
=============================================
账户：IBKR 1000 AUD
使用方式：
    python live_trading.py --paper    # 模拟盘（推荐先跑）
    python live_trading.py --live     # 实盘（确认无误后再用）

前置条件：
    1. 安装 IB Gateway（推荐）或 TWS
    2. 启用 API 连接
    3. pip install ib_insync
"""

import sys
import os
import argparse
import time
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# 导入实盘配置
import config_live as cfg

# 导入策略引擎
from strategies.trend_following import compute_trend_signal, get_trend_confidence
from strategies.mean_reversion import compute_mean_reversion_signal, get_reversion_confidence
from strategies.momentum import compute_momentum_signal, get_momentum_confidence
from ensemble.ensemble_engine import compute_ensemble_signal


class LiveTradingEngine:
    """
    实盘交易引擎

    流程：
    1. 连接 IBKR → 获取账户信息
    2. 获取历史数据 → 训练 ML 模型
    3. 每小时：获取最新1H K线 → 生成信号 → 下单
    4. 持续监控持仓 → 移动止损
    """

    def __init__(self, paper_mode=True):
        self.paper_mode = paper_mode
        self.ib = None
        self.contract = None
        self.connected = False
        self.account_id = None
        self.current_position = 0  # 当前持仓: 1=多, -1=空, 0=无
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_activated = False
        self.best_price = 0.0
        self.trade_history = []
        self.df = None              # 历史数据缓存

    # ---------- IBKR 连接 ----------
    def connect_ibkr(self):
        """连接到 IB Gateway 或 TWS"""
        try:
            from ib_insync import IB, Forex

            self.ib = IB()
            host = cfg.IBKR_HOST
            port = cfg.IBKR_PORT

            print(f"[IBKR] 正在连接到 {host}:{port} ...")
            self.ib.connect(host, port, clientId=cfg.IBKR_CLIENT_ID, timeout=30)

            # 定义合约
            self.contract = Forex("EURUSD")
            self.ib.qualifyContracts(self.contract)

            # 获取账户信息
            accounts = self.ib.managedAccounts()
            self.account_id = accounts[0] if accounts else None

            self.connected = True
            print(f"[IBKR] 连接成功！账户: {self.account_id}")
            print(f"[IBKR] 模式: {'模拟盘' if self.paper_mode else '实盘'}")

            # 打印账户摘要
            self.print_account_info()
            return True

        except ImportError:
            print("[错误] ib_insync 未安装，请运行：")
            print("       pip install ib_insync")
            return False
        except Exception as e:
            print(f"[错误] 连接 IBKR 失败: {e}")
            print("       请确认 IB Gateway 或 TWS 已启动并启用了 API 连接")
            return False

    def disconnect(self):
        """断开连接"""
        if self.ib and self.connected:
            # 如果有持仓，先取消所有挂单
            self.cancel_all_orders()
            self.ib.disconnect()
            self.connected = False
            print("[IBKR] 已断开连接")

    def print_account_info(self):
        """打印账户信息"""
        try:
            summary = self.ib.accountSummary()
            print("\n  ╔══════════════════════════════════╗")
            print("  ║      账户摘要                    ║")
            print("  ╠══════════════════════════════════╣")
            for s in summary:
                if s.tag in ("NetLiquidation", "AvailableFunds", "BuyingPower"):
                    print(f"  ║  {s.tag:<20} {float(s.value):>10,.2f} {s.currency:<3} ║")
            print("  ╚══════════════════════════════════╝\n")
        except Exception as e:
            print(f"  无法获取账户信息: {e}")

    # ---------- 数据获取 ----------
    def fetch_recent_data(self, hours_back=500):
        """从 IBKR 获取近期 K 线数据"""
        from ib_insync import util

        # IBKR 外汇 1h K线
        # IBKR 不支持 H（小时），只支持 S/D/W/M/Y，所以把小时转换成天数
        duration_days = max(1, int((hours_back + 23) // 24))
        duration_str = f"{duration_days} D"

        bars = self.ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 hour",
            whatToShow="MIDPOINT",
            useRTH=False,
            formatDate=1,
        )

        df = util.df(bars)
        if df is None or df.empty:
            print("[警告] IBKR 未返回数据，尝试 Yahoo Finance 备用...")
            return self._fallback_yahoo()

        df.set_index("date", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True)

        print(f"[数据] 从 IBKR 获取 {len(df)} 条 1h K线")
        return df[["open", "high", "low", "close", "volume"]]

    def _fallback_yahoo(self):
        """备用：从 Yahoo Finance 获取数据"""
        print("[数据] 使用 Yahoo Finance 备用数据源")
        from data.fetch_data import fetch_eurusd

        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        return fetch_eurusd(start=start, interval="1h", save_csv=False)

    # ---------- 信号生成 ----------
    def generate_signal(self, df):
        """根据当前数据生成交易信号"""
        if len(df) < 200:
            print(f"[信号] 数据不足 ({len(df)} 条)，需要至少 200 条 K线")
            return 0, {}

        try:
            ensemble = compute_ensemble_signal(df, use_ml=True)
            signal = ensemble["ensemble_signal"].iloc[-1]
            score = ensemble["ensemble_score"].iloc[-1]

            print(f"[信号] 当前信号: {signal:+.0f} (得分: {score:+.3f})")
            return int(signal), ensemble
        except Exception as e:
            print(f"[信号] 生成失败: {e}")
            return 0, {}

    # ---------- 交易执行 ----------
    def place_trade(self, direction, price, atr_value):
        """
        下单

        direction: 1=多, -1=空
        price: 当前价格
        atr_value: 当前 ATR
        """
        if direction == self.current_position:
            print(f"[交易] 方向未变 ({'多' if direction == 1 else '空'})，跳过")
            return

        # 先平仓旧持仓
        if self.current_position != 0:
            self.close_position(price, "flip")

        # 计算仓位大小（微型手）
        quantity = self._calc_position_size(price, atr_value)

        if quantity <= 0:
            print("[交易] 仓位为0，跳过")
            return

        action = "BUY" if direction == 1 else "SELL"
        stop_price = price - atr_value * cfg.ATR_STOP_MULTIPLIER * direction
        take_profit = price + atr_value * cfg.TAKE_PROFIT_ATR * direction

        try:
            # 市价单
            from ib_insync import MarketOrder

            order = MarketOrder(action=action, totalQuantity=quantity)
            trade = self.ib.placeOrder(self.contract, order)

            print(f"\n{'='*60}")
            print(f"  📊 开仓: {'多' if direction == 1 else '空'} {quantity} 单位 EUR/USD")
            print(f"  入场价格: {price:.5f}")
            print(f"  止损: {stop_price:.5f} | 止盈: {take_profit:.5f}")
            print(f"{'='*60}\n")

            # 更新状态
            self.current_position = direction
            self.entry_price = price
            self.stop_loss = stop_price
            self.take_profit = take_profit
            self.trailing_activated = False
            self.best_price = price

            # 记录
            self.trade_history.append({
                "time": datetime.now().isoformat(),
                "action": action,
                "quantity": quantity,
                "entry": float(price),
                "sl": float(stop_price),
                "tp": float(take_profit),
            })

            # 下止损单
            sl_action = "SELL" if direction == 1 else "BUY"
            from ib_insync import StopOrder
            sl_order = StopOrder(action=sl_action, totalQuantity=quantity, stopPrice=stop_price)
            self.ib.placeOrder(self.contract, sl_order)

        except Exception as e:
            print(f"[错误] 下单失败: {e}")

    def close_position(self, price, reason="manual"):
        """平仓"""
        if self.current_position == 0:
            return

        action = "SELL" if self.current_position == 1 else "BUY"
        quantity = self._get_current_quantity()

        try:
            from ib_insync import MarketOrder

            # 先取消所有挂单
            self.cancel_all_orders()

            order = MarketOrder(action=action, totalQuantity=quantity)
            self.ib.placeOrder(self.contract, order)

            pnl = (price - self.entry_price) * self.current_position * quantity
            print(f"\n{'='*60}")
            print(f"  📊 平仓: {'多' if self.current_position == 1 else '空'} → 平仓")
            print(f"  退出价格: {price:.5f}")
            print(f"  盈亏: ${pnl:+.2f}")
            print(f"  原因: {reason}")
            print(f"{'='*60}\n")

            self.current_position = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            self.trailing_activated = False

        except Exception as e:
            print(f"[错误] 平仓失败: {e}")

    def cancel_all_orders(self):
        """取消所有未成交订单"""
        if not self.connected:
            return
        try:
            trades = self.ib.trades()
            for t in trades:
                self.ib.cancelOrder(t.order)
        except Exception:
            pass

    def _calc_position_size(self, price, atr_value):
        """
        计算仓位大小（小额账户适配）

        1000 AUD ≈ 620 USD，EUR/USD 最小单位是 1000 EUR（1微型手）
        """
        capital = cfg.INITIAL_CAPITAL_USD
        risk_per_unit = atr_value * cfg.ATR_STOP_MULTIPLIER
        if risk_per_unit == 0:
            return 0

        max_risk = capital * cfg.POSITION_SIZE_RISK
        position = max_risk / risk_per_unit

        # 外汇最小单位 1000（1微型手）
        position = max(position, 1000)
        position = round(position / 1000) * 1000  # 取整到 1000

        # 杠杆限制
        max_position = capital * cfg.MAX_LEVERAGE / price
        position = min(position, max_position)
        position = round(position / 1000) * 1000

        print(f"[仓位] 建议 {int(position)} 单位 (${position * price:,.0f} 名义)")
        return int(position)

    def _get_current_quantity(self):
        """获取当前持仓量"""
        try:
            positions = self.ib.positions()
            for p in positions:
                if p.contract.symbol == "EUR" and p.contract.currency == "USD":
                    return int(abs(p.position))
        except Exception:
            pass
        return 0

    # ---------- 监控循环 ----------
    def update_trailing_stop(self, current_high, current_low, current_atr):
        """更新移动止损"""
        if self.current_position == 0:
            return

        trailing_dist = cfg.TRAILING_STOP_ATR * current_atr

        if self.current_position == 1:  # 多
            if current_high > self.best_price:
                self.best_price = current_high
            profit = self.best_price - self.entry_price
            if not self.trailing_activated and profit >= cfg.TRAILING_STOP_ACTIVATION * current_atr:
                self.trailing_activated = True
                print(f"[移动止损] 已激活 (盈利 {profit:.5f})")
            if self.trailing_activated:
                new_sl = self.best_price - trailing_dist
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
        else:  # 空
            if current_low < self.best_price:
                self.best_price = current_low
            profit = self.entry_price - self.best_price
            if not self.trailing_activated and profit >= cfg.TRAILING_STOP_ACTIVATION * current_atr:
                self.trailing_activated = True
                print(f"[移动止损] 已激活 (盈利 {profit:.5f})")
            if self.trailing_activated:
                new_sl = self.best_price + trailing_dist
                if new_sl < self.stop_loss:
                    self.stop_loss = new_sl

    def run(self):
        """主循环"""
        print("\n" + "=" * 60)
        print("  USD/EUR 量化交易系统 — 实盘引擎")
        print(f"  配置: {cfg.INITIAL_CAPITAL_AUD} AUD / {cfg.INITIAL_CAPITAL_USD} USD")
        print(f"  模式: {'模拟盘' if self.paper_mode else '实盘'}")
        print("=" * 60 + "\n")

        # 1. 连接
        if not self.connect_ibkr():
            return

        # 2. 获取数据
        print("[步骤1] 获取历史数据...")
        self.df = self.fetch_recent_data(hours_back=500)
        if self.df is None or len(self.df) < 200:
            print("[错误] 无法获取足够数据")
            return

        # 3. 训练 ML（如果有新数据）
        print("[步骤2] 准备 ML 模型...")
        try:
            from strategies.ml_model import train_ml_model
            train_ml_model(self.df, retrain=True)
            print("[ML] 模型就绪")
        except Exception as e:
            print(f"[ML] 跳过训练: {e}")

        print(f"\n[启动] 进入监控模式，每 {cfg.CHECK_INTERVAL_SECONDS/3600:.0f} 小时检查一次信号\n")

        try:
            while True:
                now = datetime.now()
                print(f"\n{'─'*60}")
                print(f"[{now.strftime('%Y-%m-%d %H:%M')}] 检查信号...")

                # 获取最新价格
                ticker = self.ib.reqMktData(self.contract, "", False, False)
                self.ib.sleep(2)
                price = ticker.last if ticker.last else ticker.close
                if not price or price <= 0:
                    print("[跳过] 无有效价格，等待下一次检查")
                    time.sleep(cfg.CHECK_INTERVAL_SECONDS)
                    continue

                # 更新数据
                self.df = self.fetch_recent_data(hours_back=200)
                if self.df is None or len(self.df) < 50:
                    time.sleep(cfg.CHECK_INTERVAL_SECONDS)
                    continue

                atr_val = self._get_latest_atr()

                # 检查持仓
                if self.current_position != 0:
                    hi = float(self.df["high"].iloc[-1])
                    lo = float(self.df["low"].iloc[-1])
                    self.update_trailing_stop(hi, lo, atr_val)

                    # 止损检查
                    if (self.current_position == 1 and lo <= self.stop_loss) or \
                       (self.current_position == -1 and hi >= self.stop_loss):
                        self.close_position(self.stop_loss, "stop_loss")
                        continue

                # 生成信号
                signal, ensemble = self.generate_signal(self.df)
                if signal != 0 and signal != self.current_position:
                    self.place_trade(signal, float(price), atr_val)
                    print(f"[状态] 当前持仓: {'多' if self.current_position == 1 else '空' if self.current_position == -1 else '无'}")

                # 等待下一次检查
                wait = cfg.CHECK_INTERVAL_SECONDS
                next_check = now + timedelta(seconds=wait)
                print(f"[下次检查] {next_check.strftime('%H:%M:%S')}")
                time.sleep(wait)

        except KeyboardInterrupt:
            print("\n[退出] 用户中断")
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[清理] 平仓并断开...")
            if self.current_position != 0:
                ticker = self.ib.reqMktData(self.contract, "", False, False)
                self.ib.sleep(1)
                last_price = ticker.last or ticker.close or 1.0
                self.close_position(float(last_price), "exit")
            self.disconnect()

    def _get_latest_atr(self, period=14):
        """获取最新 ATR"""
        if self.df is None or len(self.df) < period:
            return 0.001
        from risk.risk_manager import compute_atr
        atr = compute_atr(self.df, period)
        return float(atr.iloc[-1]) if len(atr) > 0 else 0.001


def main():
    parser = argparse.ArgumentParser(description="USD/EUR 实盘交易引擎")
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

    engine = LiveTradingEngine(paper_mode=mode)
    engine.run()


if __name__ == "__main__":
    main()