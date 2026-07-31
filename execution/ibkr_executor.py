"""IBKR 执行模块 — 对接 Interactive Brokers Gateway

Flask + ib_insync 的线程安全方案：
1. 连接在独立线程中完成（有自己的 event loop）
2. 所有后续 API 调用通过 asyncio.run_coroutine_threadsafe 使用 *Async 版本"""

import asyncio
import threading
import time


class IBKRExecutor:

    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.contract = None
        self.connected = False
        self._event_loop = None
        self._api_errors = []
        self._api_errors_lock = threading.Lock()

    @staticmethod
    def _parse_account_snapshot(values):
        """Normalize IBKR base-currency account values to USD for risk sizing."""
        rows = list(values or [])

        def find_value(tag, currency=None):
            for item in rows:
                if item.tag == tag and (currency is None or item.currency == currency):
                    try:
                        return float(item.value), item.currency
                    except (TypeError, ValueError):
                        continue
            return None, None

        native_equity, base_currency = find_value("NetLiquidation")
        native_available, available_currency = find_value("AvailableFunds", base_currency)
        native_buying_power, _ = find_value("BuyingPower", base_currency)

        if native_equity is None:
            return {}

        # IBKR's $LEDGER-ExchangeRate for USD is expressed as units of the
        # account's base currency per USD. A USD-base account needs no
        # conversion.
        usd_rate = 1.0
        if base_currency != "USD":
            ledger_rate, _ = find_value("$LEDGER-ExchangeRate", "USD")
            if ledger_rate and ledger_rate > 0:
                usd_rate = ledger_rate
            else:
                # Do not pretend a non-USD value is USD when conversion data
                # is unavailable.
                return {
                    "equity_native": native_equity,
                    "account_currency": base_currency,
                    "conversion_error": f"缺少 {base_currency}/USD 账户换算汇率",
                }

        return {
            "equity": native_equity / usd_rate,
            "equity_native": native_equity,
            "available_funds": (
                native_available / usd_rate if native_available is not None else None
            ),
            "available_funds_native": native_available,
            "buying_power": (
                native_buying_power / usd_rate if native_buying_power is not None else None
            ),
            "buying_power_native": native_buying_power,
            "account_currency": base_currency or available_currency,
            "base_per_usd": usd_rate,
        }

    def connect(self):
        """连接 IB Gateway/TWS，同时获取账户信息。返回 (success, msg, account_info)"""
        result = {}

        def _worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from ib_insync import IB, Forex
                self.ib = IB()
                self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)

                def _capture_api_error(req_id, error_code, error_string, contract=None):
                    error = {
                        "req_id": req_id,
                        "code": error_code,
                        "message": str(error_string),
                        "time": time.time(),
                    }
                    with self._api_errors_lock:
                        self._api_errors.append(error)
                        self._api_errors = self._api_errors[-200:]
                    # Informational farm/connectivity messages are noisy; order
                    # errors are retained and reported by place_market_order.
                    if error_code >= 200:
                        print(f"[IBKR] API {error_code} (reqId={req_id}): {error_string}")

                self.ib.errorEvent += _capture_api_error
                self.connected = True
                self.contract = Forex("AUDUSD")
                qualified = self.ib.qualifyContracts(self.contract)
                if not qualified or not self.contract.conId:
                    raise RuntimeError("AUD/USD IDEALPRO 合约验证失败")

                # 获取账户信息
                try:
                    accounts = self.ib.managedAccounts()
                    result["account_id"] = accounts[0] if accounts else None
                    loop.run_until_complete(self.ib.accountSummaryAsync())
                    result.update(self._parse_account_snapshot(self.ib.accountValues()))
                except Exception as e:
                    print(f"[IBKR] 账户信息获取失败: {e}")

                # 保存 event loop 引用并启动
                self._event_loop = loop
                msg = f"已连接到 {self.host}:{self.port} (Client ID: {self.client_id})"
                print(f"[IBKR] {msg}")
                # Publish success only after run_forever has actually started,
                # so the first API call cannot race the event loop startup.
                loop.call_soon(
                    lambda: result.update({"ok": True, "msg": msg})
                )
                loop.run_forever()

            except ImportError:
                result["ok"], result["msg"] = False, "ib_insync 未安装"
            except ConnectionRefusedError:
                result["ok"], result["msg"] = False, f"连接被拒绝: {self.host}:{self.port}"
            except Exception as e:
                s = str(e)
                if "clientId" in s.lower() or "already" in s.lower():
                    result["ok"], result["msg"] = False, f"Client ID {self.client_id} 已占用"
                else:
                    result["ok"], result["msg"] = False, f"连接失败: {s}"

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        deadline = time.time() + 15
        # Account details may be populated before the event loop has started.
        # Wait specifically for the terminal connection result, not merely for
        # any item to appear in the shared result dictionary.
        while "ok" not in result and time.time() < deadline:
            time.sleep(0.05)
        if "ok" not in result:
            return False, "连接超时（>15 秒）", {}
        return result.get("ok", False), result.get("msg", "未知错误"), {
            "account_id": result.get("account_id"),
            "equity": result.get("equity"),
            "equity_native": result.get("equity_native"),
            "available_funds": result.get("available_funds"),
            "available_funds_native": result.get("available_funds_native"),
            "buying_power": result.get("buying_power"),
            "buying_power_native": result.get("buying_power_native"),
            "account_currency": result.get("account_currency"),
            "base_per_usd": result.get("base_per_usd"),
            "conversion_error": result.get("conversion_error"),
        }

    @property
    def _loop(self):
        """获取 IB 内部 event loop（线程安全）"""
        loop = self._event_loop
        if loop and loop.is_running():
            return loop
        return None

    def _call(self, coro, timeout=15):
        """通过 run_coroutine_threadsafe 在 IB 的 event loop 中运行协程"""
        loop = self._loop
        if not loop:
            return None
        if not self.connected:
            print("[IBKR] 警告: 未连接，IB API 调用中止")
            return None
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[IBKR] 调用超时 ({timeout}s)")
            return None
        except Exception as e:
            print(f"[IBKR] 调用失败: {e}")
            return None

    def disconnect(self):
        if self.ib and self.connected:
            try:
                self.ib.disconnect()
            except Exception:
                pass
            self.connected = False
            self._event_loop = None

    def get_equity(self):
        """获取账户净值（线程安全）"""
        snapshot = self.get_account_snapshot()
        return snapshot.get("equity") if snapshot else None

    def get_account_snapshot(self):
        """获取账户净值、可用资金和购买力，并统一换算为 USD。"""
        async def _get():
            await self.ib.accountSummaryAsync()
            return self._parse_account_snapshot(self.ib.accountValues())
        return self._call(_get())

    def get_positions(self):
        """在线程所属的 event loop 中获取持仓。"""
        async def _get():
            return self.ib.positions()
        return self._call(_get())

    def get_market_price(self):
        """获取 AUD/USD 实时成交价（线程安全）"""
        async def _get():
            self.ib.reqMktData(self.contract, '', False, False)
            await asyncio.sleep(1.5)
            ticker = self.ib.ticker(self.contract)
            price = ticker.last if ticker.last and ticker.last > 0 else None
            if not price:
                bid = ticker.bid; ask = ticker.ask
                if bid and ask and bid > 0 and ask > 0:
                    price = (bid + ask) / 2
            try: self.ib.cancelMktData(self.contract)
            except Exception: pass
            return float(price) if price and price > 0 else None
        return self._call(_get(), timeout=5)

    def place_market_order(self, action, quantity):
        """
        下市价单。等待最终状态后返回 (success: bool, trade_or_error: any)
        """
        action = str(action).upper()
        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return False, f"无效下单数量: {quantity}"
        if action not in ("BUY", "SELL"):
            return False, f"无效下单方向: {action}"
        if quantity <= 0:
            return False, f"下单数量必须大于 0: {quantity}"
        if not self.connected or not self.ib or not self.contract:
            return False, "IBKR 未连接或 AUD/USD 合约未就绪"

        with self._api_errors_lock:
            error_start = len(self._api_errors)

        async def _do():
            from ib_insync import MarketOrder
            trade = self.ib.placeOrder(self.contract, MarketOrder(action=action, totalQuantity=quantity))
            # 市价单必须等到明确的最终状态，避免把仍在处理中的订单
            # 当作失败并在下一轮重复提交。
            for _ in range(60):
                await asyncio.sleep(0.25)
                status = trade.orderStatus.status
                if status in ('Filled', 'Cancelled', 'ApiCancelled', 'Inactive'):
                    break
            return trade
        trade = self._call(_do(), timeout=20)
        if trade is None:
            return False, "订单提交失败（API 错误）"
        status = trade.orderStatus.status
        filled = float(trade.orderStatus.filled or 0)
        remaining = float(trade.orderStatus.remaining or 0)
        if status == 'Filled' or (filled > 0 and remaining == 0):
            print(f"[IBKR] 订单成交: {trade.order.orderId}, {trade.order.totalQuantity}@{trade.orderStatus.avgFillPrice}")
            return True, trade

        details = []
        if trade.orderStatus.whyHeld:
            details.append(str(trade.orderStatus.whyHeld))
        for entry in getattr(trade, "log", []):
            code = getattr(entry, "errorCode", 0)
            message = getattr(entry, "message", "")
            if code or message:
                details.append(f"{code}: {message}".strip(": "))
        with self._api_errors_lock:
            recent_errors = self._api_errors[error_start:]
        for error in recent_errors:
            if error["req_id"] in (-1, trade.order.orderId):
                details.append(f'{error["code"]}: {error["message"]}')

        # Preserve order and fill state in the error so the caller can avoid
        # unsafe blind retries.
        unique_details = list(dict.fromkeys(item for item in details if item))
        detail_text = " | ".join(unique_details) or "IBKR 未返回具体错误文本"
        msg = (
            f"Order {trade.order.orderId} 状态={status or 'Unknown'}, "
            f"已成交={filled:g}, 剩余={remaining:g}; {detail_text}"
        )
        print(f"[IBKR] 订单未成交: {msg}")
        return False, msg

    def place_limit_order(self, action, quantity, limit_price):
        async def _do():
            from ib_insync import LimitOrder
            return self.ib.placeOrder(self.contract, LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price))
        return self._call(_do())

    def place_stop_order(self, action, quantity, stop_price):
        async def _do():
            from ib_insync import StopOrder
            return self.ib.placeOrder(self.contract, StopOrder(action=action, totalQuantity=quantity, stopPrice=stop_price))
        return self._call(_do())

    def cancel_all_orders(self):
        async def _do():
            for t in self.ib.trades():
                self.ib.cancelOrder(t.order)
        self._call(_do(), timeout=10)
