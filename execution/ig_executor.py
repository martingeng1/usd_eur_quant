"""IG Australia REST executor for AUD/USD CFD trading."""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from types import SimpleNamespace

import requests


class IGAPIError(RuntimeError):
    pass


def _load_local_env():
    """Read project-local settings without leaking them into process logs."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return {}
    values = {}
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                values[key] = value
    return values


class IGExecutor:
    BASE_URLS = {
        "demo": "https://demo-api.ig.com/gateway/deal",
        "live": "https://api.ig.com/gateway/deal",
    }

    def __init__(
        self,
        environment="demo",
        api_key=None,
        identifier=None,
        password=None,
        account_id=None,
        epic=None,
        units_per_contract=None,
        max_leverage=30,
        timeout=15,
        session=None,
    ):
        local_env = _load_local_env()
        self.environment = str(environment).lower()
        if self.environment not in self.BASE_URLS:
            raise ValueError("IG environment must be 'demo' or 'live'")
        prefix = f"IG_{self.environment.upper()}_"
        self.api_key = (
            api_key or local_env.get(prefix + "API_KEY") or os.getenv(prefix + "API_KEY")
        )
        self.identifier = (
            identifier
            or local_env.get(prefix + "IDENTIFIER")
            or os.getenv(prefix + "IDENTIFIER")
        )
        self.password = (
            password
            or local_env.get(prefix + "PASSWORD")
            or os.getenv(prefix + "PASSWORD")
        )
        self.account_id = (
            account_id
            or local_env.get(prefix + "ACCOUNT_ID")
            or os.getenv(prefix + "ACCOUNT_ID")
        )
        self.epic = (
            epic or local_env.get("IG_AUDUSD_EPIC") or os.getenv("IG_AUDUSD_EPIC")
        )
        configured_units = (
            local_env.get("IG_AUDUSD_UNITS_PER_CONTRACT")
            or os.getenv("IG_AUDUSD_UNITS_PER_CONTRACT")
        )
        self.units_per_contract = int(
            units_per_contract or configured_units or 100000
        )
        self._units_explicitly_configured = bool(
            units_per_contract or configured_units
        )
        self.deal_currency = None
        self.expiry = "DFB"
        self.min_deal_size = Decimal("0.01")
        self.max_leverage = min(
            float(
                local_env.get("IG_MAX_LEVERAGE")
                or os.getenv("IG_MAX_LEVERAGE")
                or str(max_leverage)
            ),
            30.0,
        )
        self.timeout = timeout
        self.base_url = self.BASE_URLS[self.environment]
        self.session = session or requests.Session()
        self.connected = False
        self.currency = None
        self._cst = None
        self._xst = None
        self._aud_rate_cache = None
        self._aud_rate_cache_time = 0.0
        self.aud_rate_cache_seconds = 3600
        self.price_decimals = 5
        self.price_scaling_factor = 10000.0
        self.min_stop_limit_points = 0.0

    def _headers(self, version=1):
        headers = {
            "X-IG-API-KEY": self.api_key or "",
            "Accept": "application/json; charset=UTF-8",
            "Content-Type": "application/json",
            "Version": str(version),
        }
        if self._cst:
            headers["CST"] = self._cst
        if self._xst:
            headers["X-SECURITY-TOKEN"] = self._xst
        return headers

    @staticmethod
    def _error_message(response):
        try:
            body = response.json()
            return body.get("errorCode") or body.get("error") or str(body)
        except Exception:
            return (response.text or f"HTTP {response.status_code}")[:500]

    def _request(self, method, path, version=1, **kwargs):
        response = self.session.request(
            method,
            self.base_url + path,
            headers=self._headers(version),
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code == 401 and self.connected and path != "/session":
            self.connected = False
            self.connect()
            response = self.session.request(
                method,
                self.base_url + path,
                headers=self._headers(version),
                timeout=self.timeout,
                **kwargs,
            )
        if response.status_code >= 400:
            raise IGAPIError(
                f"IG {self.environment.upper()} {method} {path} failed "
                f"({response.status_code}): {self._error_message(response)}"
            )
        if not response.content:
            return {}, response
        return response.json(), response

    def connect(self):
        missing = [
            name
            for name, value in (
                ("API_KEY", self.api_key),
                ("IDENTIFIER", self.identifier),
                ("PASSWORD", self.password),
            )
            if not value
        ]
        if missing:
            prefix = f"IG_{self.environment.upper()}_"
            return (
                False,
                "Missing environment variables: "
                + ", ".join(prefix + item for item in missing),
                {},
            )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,30}", self.identifier):
            return (
                False,
                "IG identifier 格式无效：REST API 不接受邮箱地址。"
                "请在 .env 的 IG_"
                f"{self.environment.upper()}_IDENTIFIER 中填写 IG 用户名/Client ID"
                "（仅字母、数字、连字符或下划线，1-30 位）。",
                {},
            )

        try:
            login_payload = {
                "identifier": self.identifier,
                "password": self.password,
                "encryptedPassword": False,
            }
            response = self.session.post(
                self.base_url + "/session",
                headers=self._headers(version=2),
                json=login_payload,
                timeout=self.timeout,
            )
            login_error = self._error_message(response) if response.status_code >= 400 else ""
            if (
                response.status_code == 401
                and "invalid-client-security-token" in login_error
            ):
                # A stale IG client session can survive a broker/environment
                # switch. Discard all local cookies and retry authentication
                # once with a completely fresh HTTP session.
                self.session.close()
                self.session = requests.Session()
                self._cst = None
                self._xst = None
                time.sleep(0.5)
                response = self.session.post(
                    self.base_url + "/session",
                    headers=self._headers(version=2),
                    json=login_payload,
                    timeout=self.timeout,
                )
            if response.status_code >= 400:
                raise IGAPIError(
                    f"IG login failed ({response.status_code}): "
                    f"{self._error_message(response)}"
                )
            body = response.json()
            self._cst = response.headers.get("CST")
            self._xst = response.headers.get("X-SECURITY-TOKEN")
            if not self._cst or not self._xst:
                raise IGAPIError("IG login response did not include session tokens")

            selected = None
            accounts = body.get("accounts") or []
            if self.account_id:
                selected = next(
                    (a for a in accounts if a.get("accountId") == self.account_id),
                    None,
                )
                if selected is None:
                    raise IGAPIError(
                        f"Configured IG account {self.account_id} was not returned"
                    )
                if body.get("currentAccountId") != self.account_id:
                    self._request(
                        "PUT",
                        "/session",
                        version=1,
                        json={"accountId": self.account_id, "defaultAccount": False},
                    )
            else:
                selected = next(
                    (
                        a
                        for a in accounts
                        if a.get("preferred")
                        and a.get("accountType") == "CFD"
                        and a.get("dealingEnabled", True)
                    ),
                    None,
                ) or next(
                    (
                        a
                        for a in accounts
                        if a.get("accountType") == "CFD"
                        and a.get("dealingEnabled", True)
                    ),
                    None,
                )
                if selected:
                    self.account_id = selected.get("accountId")

            if selected and selected.get("accountType") != "CFD":
                raise IGAPIError("Selected IG account is not a CFD account")
            self.currency = (
                (selected or {}).get("currency")
                or (selected or {}).get("currencyIsoCode")
                or body.get("currencyIsoCode")
                or "AUD"
            )
            self.connected = True
            self._ensure_audusd_epic()
            self._load_market_rules()
            snapshot = self.get_account_snapshot()
            snapshot["account_id"] = self.account_id or body.get("currentAccountId")
            return (
                True,
                f"Connected to IG {self.environment.upper()}",
                snapshot,
            )
        except Exception as exc:
            self.connected = False
            return False, str(exc), {}

    def disconnect(self):
        if self.connected:
            try:
                self._request("DELETE", "/session", version=1)
            except Exception:
                pass
        self.connected = False
        self._cst = None
        self._xst = None

    def _ensure_audusd_epic(self):
        if self.epic:
            return self.epic
        data, _ = self._request(
            "GET",
            "/markets",
            version=1,
            params={"searchTerm": "AUD/USD"},
        )
        candidates = data.get("markets") or []
        preferred = next(
            (
                market
                for market in candidates
                if "AUD/USD" in str(market.get("instrumentName", "")).upper()
                and market.get("expiry") in ("DFB", "-")
            ),
            None,
        ) or next(
            (
                market
                for market in candidates
                if "AUD/USD" in str(market.get("instrumentName", "")).upper()
            ),
            None,
        )
        if not preferred:
            raise IGAPIError("AUD/USD CFD market was not found for this IG account")
        self.epic = preferred.get("epic")
        return self.epic

    def _account_row(self):
        data, _ = self._request("GET", "/accounts", version=1)
        accounts = data.get("accounts") or []
        if self.account_id:
            return next(
                (row for row in accounts if row.get("accountId") == self.account_id),
                None,
            )
        return accounts[0] if accounts else None

    def _load_market_rules(self):
        data, _ = self._request(
            "GET", f"/markets/{self._ensure_audusd_epic()}", version=3
        )
        instrument = data.get("instrument") or {}
        dealing_rules = data.get("dealingRules") or {}
        snapshot = data.get("snapshot") or {}
        currencies = instrument.get("currencies") or []
        default_currency = next(
            (item.get("code") for item in currencies if item.get("isDefault")),
            None,
        )
        self.deal_currency = default_currency or next(
            (item.get("code") for item in currencies if item.get("code")),
            None,
        )
        self.expiry = instrument.get("expiry") or "DFB"
        minimum = (dealing_rules.get("minDealSize") or {}).get("value")
        if minimum is not None and Decimal(str(minimum)) > 0:
            self.min_deal_size = Decimal(str(minimum))
        self.price_decimals = int(snapshot.get("decimalPlacesFactor") or 5)
        self.price_scaling_factor = float(snapshot.get("scalingFactor") or 10000)
        normal_distance = dealing_rules.get("minNormalStopOrLimitDistance") or {}
        if str(normal_distance.get("unit") or "").upper() == "POINTS":
            self.min_stop_limit_points = float(normal_distance.get("value") or 0)

        # IG's AUD/USD Mini contract represents 10,000 base-currency units.
        # Standard FX contracts represent 100,000. The API's instrument
        # lotSize is a price-value field and must not be used as base units.
        if ".MINI." in str(self.epic).upper():
            self.units_per_contract = 10000
        elif not self._units_explicitly_configured:
            self.units_per_contract = 100000
        if not self.deal_currency:
            raise IGAPIError(
                f"IG market {self.epic} did not provide an allowed deal currency"
            )
        return data

    def _aud_per_usd(self):
        if self.currency == "USD":
            return 1.0
        if self.currency != "AUD":
            return None
        if (
            self._aud_rate_cache
            and time.time() - self._aud_rate_cache_time
            < self.aud_rate_cache_seconds
        ):
            return self._aud_rate_cache
        try:
            data, _ = self._request(
                "GET", "/markets", version=1, params={"searchTerm": "AUD/USD"}
            )
            market = next(
                (
                    item
                    for item in data.get("markets", [])
                    if "AUD/USD" in str(item.get("instrumentName", "")).upper()
                ),
                None,
            )
            if market:
                bid = float(market.get("bid") or 0)
                offer = float(market.get("offer") or 0)
                aud_usd = (bid + offer) / 2 if bid and offer else bid or offer
                if aud_usd > 0:
                    self._aud_rate_cache = 1.0 / aud_usd
                    self._aud_rate_cache_time = time.time()
                    return self._aud_rate_cache
        except Exception:
            pass
        return None

    def get_account_snapshot(self):
        row = self._account_row()
        if not row:
            return {}
        balance = row.get("balance") or {}
        equity_native = float(
            balance.get("balance", 0) + balance.get("profitLoss", 0)
        )
        available_native = float(balance.get("available", 0))
        rate = self._aud_per_usd()
        if not rate:
            return {
                "equity_native": equity_native,
                "available_funds_native": available_native,
                "buying_power_native": available_native,
                "account_currency": self.currency,
                "conversion_error": f"Unable to obtain {self.currency}/USD rate",
            }
        return {
            "equity": equity_native / rate,
            "equity_native": equity_native,
            "available_funds": available_native / rate,
            "available_funds_native": available_native,
            "buying_power": available_native / rate * self.max_leverage,
            "buying_power_native": available_native * self.max_leverage,
            "account_currency": self.currency,
            "base_per_usd": rate,
        }

    def get_positions(self):
        data, _ = self._request("GET", "/positions", version=2)
        return data.get("positions") or []

    def get_trade_history(self, days=90, limit=100):
        """Return real completed deal transactions from the active IG account."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(1, int(days)))
        data, _ = self._request(
            "GET",
            "/history/transactions",
            version=2,
            params={
                # Some IG Demo accounts return an empty collection for
                # ALL_DEAL even when ALL contains completed deal rows.
                "type": "ALL",
                "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "to": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "pageSize": min(max(int(limit), 20), 500),
                "pageNumber": 1,
            },
        )
        result = []
        for item in data.get("transactions") or []:
            instrument = str(item.get("instrumentName") or "")
            if "AUD/USD" not in instrument.upper():
                continue
            size_text = str(item.get("size") or "0").replace(",", "")
            size_match = re.search(r"[-+]?\d+(?:\.\d+)?", size_text)
            signed_size = float(size_match.group()) if size_match else 0.0
            pnl_text = str(item.get("profitAndLoss") or "0").replace(",", "")
            pnl_match = re.search(r"[-+]?\d+(?:\.\d+)?", pnl_text)
            pnl = float(pnl_match.group()) if pnl_match else 0.0
            result.append({
                "exit_time": item.get("dateUtc") or item.get("date") or "",
                "open_time": item.get("openDateUtc") or "",
                "direction": "LONG" if signed_size > 0 else "SHORT",
                "size": abs(signed_size),
                "entry_price": item.get("openLevel") or "",
                "exit_price": item.get("closeLevel") or "",
                "pnl": pnl,
                "currency": item.get("currency") or self.currency or "AUD",
                "reason": item.get("transactionType") or "DEAL",
                "reference": item.get("reference") or "",
                "instrument": instrument,
            })
        return result

    def get_market_price(self):
        epic = self._ensure_audusd_epic()
        data, _ = self._request("GET", f"/markets/{epic}", version=3)
        snapshot = data.get("snapshot") or {}
        bid = float(snapshot.get("bid") or 0)
        offer = float(snapshot.get("offer") or 0)
        price = (bid + offer) / 2 if bid and offer else bid or offer
        return price or None

    def _contracts_for_units(self, quantity):
        raw = Decimal(str(quantity)) / Decimal(str(self.units_per_contract))
        increment = self.min_deal_size
        contracts = (raw / increment).to_integral_value(
            rounding=ROUND_DOWN
        ) * increment
        if contracts < increment:
            raise ValueError(
                f"IG order {quantity} AUD is below the minimum "
                f"{increment} contract ({int(increment * self.units_per_contract)} AUD)"
            )
        return float(contracts)

    def _matching_position(self, action):
        closing_direction = str(action).upper()
        for item in self.get_positions():
            market = item.get("market") or {}
            position = item.get("position") or {}
            if market.get("epic") != self.epic:
                continue
            if position.get("direction") != closing_direction:
                return item
        return None

    def _confirm(self, deal_reference):
        last_error = None
        for _ in range(20):
            try:
                data, _ = self._request(
                    "GET", f"/confirms/{deal_reference}", version=1
                )
                status = data.get("dealStatus")
                if status == "ACCEPTED":
                    return data
                if status == "REJECTED":
                    reason = data.get("reason")
                    if not reason or reason == "UNKNOWN":
                        reason = f"UNKNOWN; confirmation={data}"
                    raise IGAPIError(
                        f"IG order rejected: {reason}"
                    )
            except IGAPIError as exc:
                if "404" not in str(exc):
                    raise
                last_error = exc
            time.sleep(0.25)
        raise IGAPIError(
            f"Timed out waiting for IG deal confirmation: {last_error or deal_reference}"
        )

    def _normalise_protection_levels(self, action, stop_loss, take_profit):
        if stop_loss is None and take_profit is None:
            return None, None
        market, _ = self._request(
            "GET", f"/markets/{self._ensure_audusd_epic()}", version=3
        )
        snapshot = market.get("snapshot") or {}
        bid = float(snapshot.get("bid") or 0)
        offer = float(snapshot.get("offer") or 0)
        decimals = int(snapshot.get("decimalPlacesFactor") or self.price_decimals)
        scaling = float(snapshot.get("scalingFactor") or self.price_scaling_factor)
        minimum = self.min_stop_limit_points / scaling if scaling else 0.0
        action = str(action).upper()
        sl = float(stop_loss) if stop_loss is not None else None
        tp = float(take_profit) if take_profit is not None else None
        if action == "BUY":
            if sl is not None and bid:
                sl = min(sl, bid - minimum)
            if tp is not None and offer:
                tp = max(tp, offer + minimum)
        else:
            if sl is not None and offer:
                sl = max(sl, offer + minimum)
            if tp is not None and bid:
                tp = min(tp, bid - minimum)
        return (
            round(sl, decimals) if sl is not None else None,
            round(tp, decimals) if tp is not None else None,
        )

    def update_position_protection(self, deal_id, action, stop_loss, take_profit):
        sl, tp = self._normalise_protection_levels(action, stop_loss, take_profit)
        payload = {
            "guaranteedStop": False,
            "trailingStop": False,
            "stopLevel": sl,
            "limitLevel": tp,
        }
        data, _ = self._request(
            "PUT", f"/positions/otc/{deal_id}", version=2, json=payload
        )
        confirmation = self._confirm(data["dealReference"])
        return sl, tp, confirmation

    def place_market_order(self, action, quantity, stop_loss=None, take_profit=None):
        if not self.connected:
            return False, "IG is not connected"
        action = str(action).upper()
        if action not in ("BUY", "SELL"):
            return False, f"Invalid order direction: {action}"
        try:
            quantity = int(quantity)
            size = self._contracts_for_units(quantity)
            if size <= 0:
                raise ValueError("quantity must be positive")

            current = self._matching_position(action)
            if current:
                position = current.get("position") or {}
                close_size = min(size, float(position.get("size") or size))
                # IG documents POST + `_method: DELETE` as the compatible
                # alternative for clients/proxies that discard DELETE bodies.
                headers = self._headers(version=1)
                headers["_method"] = "DELETE"
                response = self.session.request(
                    "POST",
                    self.base_url + "/positions/otc",
                    headers=headers,
                    timeout=self.timeout,
                    json={
                        "dealId": position.get("dealId"),
                        "direction": action,
                        "size": close_size,
                        "orderType": "MARKET",
                        "timeInForce": "FILL_OR_KILL",
                    },
                )
                if response.status_code >= 400:
                    raise IGAPIError(
                        f"IG {self.environment.upper()} close position failed "
                        f"({response.status_code}): {self._error_message(response)}"
                    )
                data = response.json()
            else:
                stop_loss, take_profit = self._normalise_protection_levels(
                    action, stop_loss, take_profit
                )
                payload = {
                    "currencyCode": self.deal_currency,
                    "direction": action,
                    "epic": self._ensure_audusd_epic(),
                    "expiry": self.expiry,
                    "forceOpen": True,
                    "guaranteedStop": False,
                    "orderType": "MARKET",
                    "size": size,
                    "timeInForce": "FILL_OR_KILL",
                    "trailingStop": False,
                }
                if stop_loss is not None:
                    payload["stopLevel"] = stop_loss
                if take_profit is not None:
                    payload["limitLevel"] = take_profit
                data, _ = self._request(
                    "POST",
                    "/positions/otc",
                    version=2,
                    json=payload,
                )
            confirmation = self._confirm(data["dealReference"])
            level = float(confirmation.get("level") or 0)
            order_id = confirmation.get("dealId") or data["dealReference"]
            actual_quantity = int(round(size * self.units_per_contract))
            result = SimpleNamespace(
                order=SimpleNamespace(
                    orderId=order_id,
                    totalQuantity=actual_quantity,
                ),
                orderStatus=SimpleNamespace(avgFillPrice=level),
                confirmation=confirmation,
                stopLevel=stop_loss,
                limitLevel=take_profit,
            )
            return True, result
        except Exception as exc:
            return False, str(exc)
