import unittest
from unittest.mock import patch

from execution.ig_executor import IGExecutor


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self.content = b"{}" if body is not None else b""
        self.text = str(self._body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self):
        self.calls = []
        self.positions = []

    def close(self):
        pass

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/session"):
            return FakeResponse(
                body={
                    "currentAccountId": "DEMO1",
                    "accounts": [{
                        "accountId": "DEMO1",
                        "accountType": "CFD",
                        "currency": "AUD",
                        "preferred": True,
                        "dealingEnabled": True,
                    }],
                },
                headers={"CST": "client-token", "X-SECURITY-TOKEN": "account-token"},
            )
        raise AssertionError(url)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        path = url.split("/gateway/deal", 1)[1]
        if path == "/markets" and kwargs.get("params", {}).get("searchTerm") == "AUD/USD":
            return FakeResponse(body={"markets": [{
                "instrumentName": "AUD/USD",
                "epic": "CS.D.AUDUSD.CFD.IP",
                "expiry": "DFB",
            }]})
        if path == "/markets" and kwargs.get("params", {}).get("searchTerm") == "AUD/USD":
            return FakeResponse(body={"markets": [{
                "instrumentName": "AUD/USD",
                "bid": 0.65,
                "offer": 0.65,
            }]})
        if path == "/accounts":
            return FakeResponse(body={"accounts": [{
                "accountId": "DEMO1",
                "balance": {
                    "balance": 50000,
                    "profitLoss": 100,
                    "available": 40000,
                },
            }]})
        if path == "/positions":
            return FakeResponse(body={"positions": self.positions})
        if path == "/history/transactions":
            return FakeResponse(body={"transactions": [{
                "instrumentName": "AUD/USD Mini",
                "dateUtc": "2026-07-31T10:20:32",
                "openDateUtc": "2026-07-31T10:13:16",
                "openLevel": "1.15079",
                "closeLevel": "1.15034",
                "profitAndLoss": "A$12.50",
                "currency": "AUD",
                "size": "-2.2",
                "transactionType": "POSITION",
                "reference": "REF-CLOSE",
            }]})
        if path == "/positions/otc":
            return FakeResponse(body={"dealReference": "REF1"})
        if path == "/confirms/REF1":
            return FakeResponse(body={
                "dealStatus": "ACCEPTED",
                "dealId": "D1",
                "level": 1.15,
            })
        if path.startswith("/markets/"):
            return FakeResponse(body={
                "instrument": {
                    "name": "AUD/USD",
                    "expiry": "DFB",
                    "currencies": [{"code": "USD", "isDefault": False}],
                },
                "dealingRules": {
                    "minDealSize": {"unit": "POINTS", "value": 0.1},
                    "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 2.0},
                },
                "snapshot": {
                    "bid": 1.1499,
                    "offer": 1.1501,
                    "decimalPlacesFactor": 5,
                    "scalingFactor": 10000,
                },
            })
        if path == "/session" and method == "DELETE":
            return FakeResponse(body={})
        raise AssertionError((method, path, kwargs))


class IGExecutorTests(unittest.TestCase):
    def make_executor(self):
        return IGExecutor(
            environment="demo",
            api_key="test-key",
            identifier="test-user",
            password="test-password",
            account_id="DEMO1",
            units_per_contract=100000,
            session=FakeSession(),
        )

    def test_environment_urls_are_isolated(self):
        demo = self.make_executor()
        live = IGExecutor(
            environment="live",
            api_key="k",
            identifier="u",
            password="p",
            session=FakeSession(),
        )
        self.assertIn("demo-api.ig.com", demo.base_url)
        self.assertEqual("https://api.ig.com/gateway/deal", live.base_url)

    def test_email_is_rejected_as_api_identifier(self):
        executor = IGExecutor(
            environment="demo",
            api_key="k",
            identifier="person@example.com",
            password="p",
            session=FakeSession(),
        )
        ok, message, _ = executor.connect()
        self.assertFalse(ok)
        self.assertIn("不接受邮箱", message)

    @patch("execution.ig_executor._load_local_env")
    def test_demo_and_live_credentials_are_separate(self, load_env):
        load_env.return_value = {
            "IG_DEMO_API_KEY": "demo-key",
            "IG_DEMO_IDENTIFIER": "demo-user",
            "IG_DEMO_PASSWORD": "demo-password",
            "IG_LIVE_API_KEY": "live-key",
            "IG_LIVE_IDENTIFIER": "live-user",
            "IG_LIVE_PASSWORD": "live-password",
        }
        demo = IGExecutor(environment="demo", session=FakeSession())
        live = IGExecutor(environment="live", session=FakeSession())
        self.assertEqual("demo-user", demo.identifier)
        self.assertEqual("demo-password", demo.password)
        self.assertEqual("live-user", live.identifier)
        self.assertEqual("live-password", live.password)

    @patch("execution.ig_executor.time.sleep", return_value=None)
    @patch("execution.ig_executor.requests.Session")
    def test_stale_client_token_retries_once_with_fresh_session(
        self, session_factory, _sleep
    ):
        class StaleSession(FakeSession):
            def post(self, url, **kwargs):
                return FakeResponse(
                    status=401,
                    body={
                        "errorCode":
                        "service.security.authentication.failure-invalid-client-security-token"
                    },
                )

        fresh = FakeSession()
        session_factory.return_value = fresh
        executor = IGExecutor(
            environment="demo",
            api_key="k",
            identifier="demo-user",
            password="p",
            account_id="DEMO1",
            session=StaleSession(),
        )
        ok, message, _ = executor.connect()
        self.assertTrue(ok, message)
        session_factory.assert_called_once_with()

    def test_connect_and_snapshot(self):
        executor = self.make_executor()
        ok, message, snapshot = executor.connect()
        self.assertTrue(ok, message)
        self.assertEqual("DEMO1", snapshot["account_id"])
        self.assertEqual("AUD", snapshot["account_currency"])
        self.assertAlmostEqual(50000 * 0.65 + 100 * 0.65, snapshot["equity"])
        self.assertAlmostEqual(40000 * 0.65 * 30, snapshot["buying_power"])
        self.assertEqual("CS.D.AUDUSD.CFD.IP", executor.epic)

    def test_aud_usd_conversion_is_cached(self):
        executor = self.make_executor()
        self.assertTrue(executor.connect()[0])
        before = len([
            call for call in executor.session.calls
            if call[1].endswith("/markets")
            and call[2].get("params", {}).get("searchTerm") == "AUD/USD"
        ])
        executor.get_account_snapshot()
        after = len([
            call for call in executor.session.calls
            if call[1].endswith("/markets")
            and call[2].get("params", {}).get("searchTerm") == "AUD/USD"
        ])
        self.assertEqual(before, after)

    def test_market_order_uses_contract_size_and_confirms(self):
        executor = self.make_executor()
        self.assertTrue(executor.connect()[0])
        ok, result = executor.place_market_order("SELL", 35000)
        self.assertTrue(ok, result)
        order_calls = [
            call for call in executor.session.calls
            if call[0] == "POST" and call[1].endswith("/positions/otc")
        ]
        self.assertEqual(0.3, order_calls[-1][2]["json"]["size"])
        self.assertEqual("USD", order_calls[-1][2]["json"]["currencyCode"])
        self.assertEqual("D1", result.order.orderId)
        self.assertEqual(1.15, result.orderStatus.avgFillPrice)
        self.assertEqual(30000, result.order.totalQuantity)

    def test_mini_contract_uses_10000_units_and_point_one_step(self):
        executor = self.make_executor()
        executor.epic = "CS.D.AUDUSD.MINI.IP"
        self.assertTrue(executor.connect()[0])
        ok, result = executor.place_market_order("SELL", 23651)
        self.assertTrue(ok, result)
        order_calls = [
            call for call in executor.session.calls
            if call[0] == "POST" and call[1].endswith("/positions/otc")
        ]
        self.assertEqual(2.3, order_calls[-1][2]["json"]["size"])
        self.assertEqual("USD", order_calls[-1][2]["json"]["currencyCode"])
        self.assertEqual(23000, result.order.totalQuantity)

    def test_market_order_includes_ig_stop_and_limit_levels(self):
        executor = self.make_executor()
        executor.epic = "CS.D.AUDUSD.MINI.IP"
        self.assertTrue(executor.connect()[0])
        ok, result = executor.place_market_order(
            "SELL", 23000, stop_loss=0.65969, take_profit=0.62112
        )
        self.assertTrue(ok, result)
        order_calls = [
            call for call in executor.session.calls
            if call[0] == "POST" and call[1].endswith("/positions/otc")
        ]
        payload = order_calls[-1][2]["json"]
        self.assertEqual(1.15969, payload["stopLevel"])
        self.assertEqual(1.12112, payload["limitLevel"])

    def test_opposite_order_closes_existing_position(self):
        executor = self.make_executor()
        self.assertTrue(executor.connect()[0])
        executor.session.positions = [{
            "market": {"epic": executor.epic},
            "position": {
                "dealId": "OPEN1",
                "direction": "BUY",
                "size": 0.2,
            },
        }]
        ok, result = executor.place_market_order("SELL", 20000)
        self.assertTrue(ok, result)
        close_calls = [
            call for call in executor.session.calls
            if (
                call[0] == "POST"
                and call[1].endswith("/positions/otc")
                and call[2]["headers"].get("_method") == "DELETE"
            )
        ]
        self.assertEqual("OPEN1", close_calls[-1][2]["json"]["dealId"])
        self.assertEqual(0.2, close_calls[-1][2]["json"]["size"])

    def test_trade_history_uses_real_ig_transactions(self):
        executor = self.make_executor()
        self.assertTrue(executor.connect()[0])
        trades = executor.get_trade_history()
        self.assertEqual(1, len(trades))
        self.assertEqual("SHORT", trades[0]["direction"])
        self.assertEqual(12.5, trades[0]["pnl"])
        self.assertEqual("AUD", trades[0]["currency"])


if __name__ == "__main__":
    unittest.main()
