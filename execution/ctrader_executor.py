"""
cTrader 执行模块 — 对接 IC Markets cTrader 平台
"""
import json
import threading
import time
import requests


class CTraderExecutor:
    """
    cTrader 执行器

    通过 cTrader Open API 进行连接和下单
    支持 IC Markets 等使用 cTrader 的经纪商

    前置条件：
    1. 安装并启动 cTrader 桌面版
    2. 在设置中启用 Open API
    3. 获取 Access Token
    """

    def __init__(self, access_token=None, app_id="quant_audusd", api_url="https://api.ctrader.com"):
        self.access_token = access_token
        self.app_id = app_id
        self.api_url = api_url.rstrip("/")
        self.connected = False
        self.account_id = None
        self.session = requests.Session()

    def connect(self, access_token=None):
        """
        连接到 cTrader

        参数
        ----
        access_token : str, cTrader Open API Access Token
        """
        if access_token:
            self.access_token = access_token

        if not self.access_token:
            print("[cTrader] 需要提供 Access Token")
            print("         获取方式: cTrader → 设置 → Open API → 生成 Token")
            return False

        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        })

        try:
            # 获取账户信息
            resp = self.session.get(f"{self.api_url}/v2/accounts")
            if resp.status_code == 200:
                accounts = resp.json()
                if accounts:
                    self.account_id = accounts[0].get("accountId")
                    self.connected = True
                    print(f"[cTrader] 已连接, 账户 ID: {self.account_id}")
                    return True
            print(f"[cTrader] 连接失败: {resp.status_code} {resp.text}")
            return False
        except Exception as e:
            print(f"[cTrader] 连接异常: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.session.close()
        self.session = requests.Session()
        print("[cTrader] 已断开连接")

    def get_account_info(self):
        """获取账户信息"""
        if not self.connected:
            return None
        resp = self.session.get(f"{self.api_url}/v2/accounts/{self.account_id}")
        if resp.status_code == 200:
            return resp.json()
        return None

    def get_positions(self):
        """获取当前持仓"""
        if not self.connected:
            return []
        resp = self.session.get(
            f"{self.api_url}/v2/accounts/{self.account_id}/positions"
        )
        if resp.status_code == 200:
            return resp.json()
        return []

    def get_symbol_info(self, symbol="AUDUSD"):
        """获取品种信息"""
        if not self.connected:
            return None
        resp = self.session.get(f"{self.api_url}/v2/symbols/{symbol}")
        if resp.status_code == 200:
            return resp.json()
        return None

    def place_market_order(self, symbol="AUDUSD", direction="BUY", volume=10000):
        """
        下市价单

        参数
        ----
        symbol : str, 交易品种代码
        direction : str, 'BUY' 或 'SELL'
        volume : int, 交易量（以基础货币计，外汇通常以 1000 为单位）

        返回
        ----
        dict : 订单结果 或 None
        """
        if not self.connected:
            print("[cTrader] 未连接，无法下单")
            return None

        payload = {
            "symbol": symbol,
            "tradeSide": direction,
            "volume": volume,
            "type": "MARKET",
        }

        try:
            resp = self.session.post(
                f"{self.api_url}/v2/accounts/{self.account_id}/orders",
                json=payload,
            )
            if resp.status_code in (200, 201):
                result = resp.json()
                print(f"[cTrader] 市价单已提交: {direction} {volume} {symbol}")
                return result
            print(f"[cTrader] 下单失败: {resp.status_code} {resp.text}")
            return None
        except Exception as e:
            print(f"[cTrader] 下单异常: {e}")
            return None

    def place_limit_order(self, symbol="AUDUSD", direction="BUY", volume=10000,
                          limit_price=1.0):
        """下限价单"""
        if not self.connected:
            return None

        payload = {
            "symbol": symbol,
            "tradeSide": direction,
            "volume": volume,
            "type": "LIMIT",
            "limitPrice": limit_price,
        }

        try:
            resp = self.session.post(
                f"{self.api_url}/v2/accounts/{self.account_id}/orders",
                json=payload,
            )
            if resp.status_code in (200, 201):
                result = resp.json()
                print(f"[cTrader] 限价单已提交: {direction} {volume} @ {limit_price}")
                return result
            print(f"[cTrader] 下单失败: {resp.status_code}")
            return None
        except Exception as e:
            print(f"[cTrader] 下单异常: {e}")
            return None

    def place_stop_order(self, symbol="AUDUSD", direction="BUY", volume=10000,
                         stop_price=1.0):
        """下止损单"""
        if not self.connected:
            return None

        payload = {
            "symbol": symbol,
            "tradeSide": direction,
            "volume": volume,
            "type": "STOP",
            "stopPrice": stop_price,
        }

        try:
            resp = self.session.post(
                f"{self.api_url}/v2/accounts/{self.account_id}/orders",
                json=payload,
            )
            if resp.status_code in (200, 201):
                result = resp.json()
                print(f"[cTrader] 止损单已提交: {direction} {volume} @ stop {stop_price}")
                return result
            print(f"[cTrader] 下单失败: {resp.status_code}")
            return None
        except Exception as e:
            print(f"[cTrader] 下单异常: {e}")
            return None

    def close_all_positions(self):
        """平掉所有持仓"""
        if not self.connected:
            return
        positions = self.get_positions()
        for pos in positions:
            payload = {
                "positionId": pos.get("positionId"),
                "volume": pos.get("volume"),
            }
            resp = self.session.delete(
                f"{self.api_url}/v2/accounts/{self.account_id}/positions/{pos.get('positionId')}",
                json=payload,
            )
            if resp.status_code == 200:
                print(f"[cTrader] 已平仓: {pos.get('symbol')} {pos.get('tradeSide')}")
            else:
                print(f"[cTrader] 平仓失败: {resp.status_code}")

    def get_candles(self, symbol="AUDUSD", timeframe="H1", count=500):
        """
        获取K线数据

        参数
        ----
        symbol : str
        timeframe : str, 'M1', 'M5', 'M15', 'H1', 'H4', 'D1'
        count : int, 获取K线数量

        返回
        ----
        list : K线数据 或 None
        """
        if not self.connected:
            return None

        resp = self.session.get(
            f"{self.api_url}/v2/symbols/{symbol}/candles",
            params={"timeframe": timeframe, "count": count},
        )
        if resp.status_code == 200:
            return resp.json()
        return None


def get_ctrader_access_token_instructions():
    """打印如何获取 cTrader Access Token 的说明"""
    print("""
    ╔══════════════════════════════════════════════════╗
    ║  获取 cTrader Open API Access Token 步骤:        ║
    ╠══════════════════════════════════════════════════╣
    ║  1. 打开 cTrader 桌面版                           ║
    ║  2. 点击左下角齿轮图标 → 设置                     ║
    ║  3. 选择 "Open API" 选项卡                        ║
    ║  4. 点击 "生成新 Token"                           ║
    ║  5. 复制生成的 Access Token                       ║
    ║  6. 在代码中设置:                                 ║
    ║     executor = CTraderExecutor(                   ║
    ║         access_token="你的token"                  ║
    ║     )                                             ║
    ╚══════════════════════════════════════════════════╝
    """)