"""
IBKR 执行模块 — 对接 Interactive Brokers Gateway
"""
import time
import threading


class IBKRExecutor:
    """
    IBKR 实盘/模拟盘执行器

    通过 ib_insync 库连接 IB Gateway/TWS
    需要先运行 IB Gateway 或 TWS，并在配置中启用 API 连接
    """

    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.contract = None
        self.connected = False

    def connect(self):
        """连接到 IB Gateway/TWS"""
        try:
            from ib_insync import IB, Forex
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True

            # 定义 EUR/USD 外汇合约
            self.contract = Forex("EURUSD")
            self.ib.qualifyContracts(self.contract)

            print(f"[IBKR] 已连接到 {self.host}:{self.port}")
            return True
        except ImportError:
            print("[IBKR] ib_insync 未安装，请运行: pip install ib_insync")
            return False
        except Exception as e:
            print(f"[IBKR] 连接失败: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """断开连接"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            print("[IBKR] 已断开连接")

    def get_account_summary(self):
        """获取账户摘要"""
        if not self.connected:
            return None
        summary = self.ib.accountSummary()
        return summary

    def get_positions(self):
        """获取当前持仓"""
        if not self.connected:
            return []
        return self.ib.positions()

    def get_market_data(self):
        """获取实时市场数据"""
        if not self.connected:
            return None
        self.ib.reqMktData(self.contract, "", False, False)
        self.ib.sleep(1)
        ticker = self.ib.ticker(self.contract)
        return ticker

    def place_market_order(self, action, quantity):
        """
        下市价单

        参数
        ----
        action : str, 'BUY' 或 'SELL'
        quantity : int, 数量

        返回
        ----
        trade : Trade 对象 或 None
        """
        if not self.connected:
            print("[IBKR] 未连接，无法下单")
            return None

        try:
            from ib_insync import MarketOrder

            order = MarketOrder(action=action, totalQuantity=quantity)
            trade = self.ib.placeOrder(self.contract, order)
            print(f"[IBKR] 已下单: {action} {quantity} EUR/USD")
            return trade
        except Exception as e:
            print(f"[IBKR] 下单失败: {e}")
            return None

    def place_limit_order(self, action, quantity, limit_price):
        """下限价单"""
        if not self.connected:
            return None

        try:
            from ib_insync import LimitOrder

            order = LimitOrder(action=action, totalQuantity=quantity, lmtPrice=limit_price)
            trade = self.ib.placeOrder(self.contract, order)
            print(f"[IBKR] 已下限价单: {action} {quantity} @ {limit_price}")
            return trade
        except Exception as e:
            print(f"[IBKR] 下单失败: {e}")
            return None

    def place_stop_order(self, action, quantity, stop_price):
        """下止损单"""
        if not self.connected:
            return None

        try:
            from ib_insync import StopOrder

            order = StopOrder(action=action, totalQuantity=quantity, stopPrice=stop_price)
            trade = self.ib.placeOrder(self.contract, order)
            print(f"[IBKR] 已下止损单: {action} {quantity} @ stop {stop_price}")
            return trade
        except Exception as e:
            print(f"[IBKR] 下单失败: {e}")
            return None

    def cancel_all_orders(self):
        """取消所有未成交订单"""
        if not self.connected:
            return
        trades = self.ib.trades()
        for trade in trades:
            self.ib.cancelOrder(trade.order)
        print("[IBKR] 已取消所有订单")


class IBRKDataFeed:
    """
    IBKR 实时数据源 — 替代 yfinance 获取实时数据
    """

    def __init__(self, executor):
        self.executor = executor
        self.data_buffer = []

    def start_streaming(self, callback=None):
        """
        开始接收实时行情

        参数
        ----
        callback : function, 每收到新数据时调用的回调函数
        """
        if not self.executor.connected:
            print("[IBKR] 数据源未连接，无法流式接收")
            return

        def on_tick(ticker):
            if callback:
                callback({
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "last": ticker.last,
                    "volume": ticker.volume,
                })

        self.executor.contract = self.executor.ib.Forex("EURUSD")
        self.executor.ib.qualifyContracts(self.executor.contract)
        self.executor.ib.reqMktData(self.executor.contract, "", False, False)
        self.executor.contract.pendingTickersEvent += on_tick

        # 在后台运行事件循环
        def run_loop():
            while self.executor.connected:
                self.executor.ib.sleep(1)

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()