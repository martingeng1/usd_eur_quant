from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class InstitutionalCrossAssetCTAV2(QCAlgorithm):
    """Cross-asset managed-futures research implementation.

    This is deliberately different from a top-N signal strategy. Every liquid
    market can contribute a risk-scaled position. The dominant forecast is
    multi-horizon time-series momentum; commodity curve carry and group-neutral
    relative value are secondary, independently calculated forecasts.

    It is a transparent CTA-style research model, not a claim of institutional
    execution quality or a promise of positive performance.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        # Standard futures are integer-lot instruments. A $5m notional
        # research account lets the 30+ market risk-parity construction work;
        # percentage returns remain comparable with other account sizes.
        self.set_cash(5_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True

        self.base_target_volatility = 0.10
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31)),
        ]
        self.oos_equity = {label: {"start": None, "end": None}
                           for label, _, _ in self.oos_blocks}

        specs = {
            # Metals
            "gold": (Futures.Metals.GOLD, "metals", True),
            "silver": (Futures.Metals.SILVER, "metals", True),
            "copper": (Futures.Metals.COPPER, "metals", True),
            "platinum": (Futures.Metals.PLATINUM, "metals", True),
            "palladium": (Futures.Metals.PALLADIUM, "metals", True),
            # Energy
            "crude": (Futures.Energy.CRUDE_OIL_WTI, "energy", True),
            "natgas": (Futures.Energy.NATURAL_GAS, "energy", True),
            "gasoline": (Futures.Energy.GASOLINE, "energy", True),
            "heating_oil": (Futures.Energy.HEATING_OIL, "energy", True),
            # Agriculture
            "corn": (Futures.Grains.CORN, "grains", True),
            "wheat": (Futures.Grains.WHEAT, "grains", True),
            "soybeans": (Futures.Grains.SOYBEANS, "grains", True),
            "soymeal": (Futures.Grains.SOYBEAN_MEAL, "grains", True),
            "soyoil": (Futures.Grains.SOYBEAN_OIL, "grains", True),
            "oats": (Futures.Grains.OATS, "grains", True),
            "live_cattle": (Futures.Meats.LIVE_CATTLE, "meats", True),
            "lean_hogs": (Futures.Meats.LEAN_HOGS, "meats", True),
            "feeder_cattle": (Futures.Meats.FEEDER_CATTLE, "meats", True),
            "sugar": (Futures.Softs.SUGAR_11, "softs", True),
            "lumber": (Futures.Forestry.LUMBER, "softs", True),
            # Equity indices
            "sp500": (Futures.Indices.SP_500_E_MINI, "equity", False),
            "nasdaq": (Futures.Indices.NASDAQ_100_E_MINI, "equity", False),
            "russell": (Futures.Indices.RUSSELL_2000_E_MINI, "equity", False),
            "dow": (Futures.Indices.DOW_30_E_MINI, "equity", False),
            # Government rates
            "ust2": (Futures.Financials.Y_2_TREASURY_NOTE, "rates", False),
            "ust5": (Futures.Financials.Y_5_TREASURY_NOTE, "rates", False),
            "ust10": (Futures.Financials.Y_10_TREASURY_NOTE, "rates", False),
            "ust30": (Futures.Financials.Y_30_TREASURY_BOND, "rates", False),
            # G10/major FX
            "aud": (Futures.Currencies.AUD, "fx", False),
            "cad": (Futures.Currencies.CAD, "fx", False),
            "chf": (Futures.Currencies.CHF, "fx", False),
            "eur": (Futures.Currencies.EUR, "fx", False),
            "gbp": (Futures.Currencies.GBP, "fx", False),
            "jpy": (Futures.Currencies.JPY, "fx", False),
            "mxn": (Futures.Currencies.MXN, "fx", False),
        }

        self.markets, self.states, self.group_of = {}, {}, {}
        self.held_symbols = set()
        for name, (ticker, group, is_commodity) in specs.items():
            future = self.add_future(
                ticker, Resolution.DAILY, extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                contract_depth_offset=0)
            # The chain supplies real near/deferred prices for the carry leg.
            future.set_filter(0, 360)
            self.markets[name] = {"future": future, "commodity": is_commodity}
            self.states[name] = {"closes": RollingWindow[float](300), "carry": None}
            self.group_of[name] = group

        self.high_water = self.portfolio.total_portfolio_value
        self.counts = {"ready": 0, "trend": 0, "carry": 0, "relative_value": 0,
                       "targets": 0, "orders": 0, "band_skips": 0}
        self.schedule.on(self.date_rules.week_start(), self.time_rules.at(13, 0), self.rebalance)
        self.schedule.on(self.date_rules.month_end(), self.time_rules.at(15, 30), self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for name, market in self.markets.items():
            symbol = market["future"].symbol
            state = self.states[name]
            if symbol in data.bars:
                price = float(data.bars[symbol].close)
                if price > 0:
                    state["closes"].add(price)

            if market["commodity"]:
                chain = data.future_chains.get(symbol)
                if chain:
                    contracts = sorted(
                        [contract for contract in chain
                         if contract.last_price > 0
                         and contract.expiry > self.time + timedelta(days=20)],
                        key=lambda contract: contract.expiry)
                    if len(contracts) >= 2:
                        front = contracts[0]
                        deferred = next((contract for contract in contracts[1:]
                                         if (contract.expiry - front.expiry).days >= 45), None)
                        if deferred is not None:
                            days = (deferred.expiry - front.expiry).days
                            if days > 0:
                                # Positive means backwardation / favourable long carry.
                                state["carry"] = -math.log(float(deferred.last_price) /
                                                            float(front.last_price)) * 365.0 / days

    def rebalance(self):
        if self.is_warming_up:
            return
        self.high_water = max(self.high_water, self.portfolio.total_portfolio_value)
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(self.high_water, 1.0)
        target_vol = self.base_target_volatility * self._drawdown_scale(drawdown)

        rows, by_group, commodity_carries = [], {}, []
        for name, state in self.states.items():
            closes = state["closes"]
            if closes.count < 253:
                continue
            vol = self._daily_volatility(closes, 63)
            if vol <= 0:
                continue
            r1 = closes[0] / closes[21] - 1.0
            r3 = closes[0] / closes[63] - 1.0
            r6 = closes[0] / closes[126] - 1.0
            r12 = closes[0] / closes[252] - 1.0
            trend = (self._sign(r1) + self._sign(r3) + self._sign(r6) + self._sign(r12)) / 4.0
            row = {"name": name, "vol": vol, "r12": r12, "trend": trend,
                   "carry": state["carry"], "carry_forecast": 0.0, "rv": 0.0}
            rows.append(row)
            by_group.setdefault(self.group_of[name], []).append(row)
            if state["carry"] is not None:
                commodity_carries.append(state["carry"])

        if not rows:
            return
        # Cross-sectional carry removes the market-wide curve level. It is a
        # separate commodity forecast, not a directional bet on all curves.
        carry_median, carry_scale = self._robust_location_scale(commodity_carries)
        for row in rows:
            if row["carry"] is not None and carry_scale > 0:
                row["carry_forecast"] = self._clip((row["carry"] - carry_median) /
                                                    (2.0 * carry_scale), -1.0, 1.0)
            peers = by_group[self.group_of[row["name"]]]
            if len(peers) > 1:
                median_return, scale = self._robust_location_scale([peer["r12"] for peer in peers])
                if scale > 0:
                    row["rv"] = self._clip((row["r12"] - median_return) / (2.0 * scale), -1.0, 1.0)

        active = []
        for row in rows:
            # Fixed ex-ante sleeve blend: 65% time-series trend, 25% commodity
            # carry, and 10% within-group relative value.
            forecast = 0.65 * row["trend"] + 0.25 * row["carry_forecast"] + 0.10 * row["rv"]
            if abs(forecast) < 0.20:
                continue
            row["forecast"] = forecast
            active.append(row)
            self.counts["ready"] += 1
            self.counts["trend"] += int(row["trend"] != 0)
            self.counts["carry"] += int(row["carry_forecast"] != 0)
            self.counts["relative_value"] += int(row["rv"] != 0)

        # Equal risk between asset groups, then equal risk within each group.
        active_by_group = {}
        for row in active:
            active_by_group.setdefault(self.group_of[row["name"]], []).append(row)
        targets = {}
        group_count = len(active_by_group)
        for group, members in active_by_group.items():
            per_market_budget = self.portfolio.total_portfolio_value * target_vol
            per_market_budget /= math.sqrt(252.0 * group_count * len(members))
            for row in members:
                mapped = self.markets[row["name"]]["future"].mapped
                quantity = self._target_quantity(mapped, row["vol"], per_market_budget, row["forecast"])
                if quantity != 0:
                    targets[mapped] = quantity

        self._apply_targets(targets)
        self.counts["targets"] += len(targets)

    def _target_quantity(self, symbol, daily_vol, risk_budget, forecast):
        if symbol is None or symbol not in self.securities or not self.securities[symbol].is_tradable:
            return 0
        security = self.securities[symbol]
        per_contract_risk = float(security.price) * float(security.symbol_properties.contract_multiplier) * daily_vol
        if per_contract_risk <= 0:
            return 0
        raw_quantity = risk_budget * abs(forecast) / per_contract_risk
        quantity = int(math.floor(raw_quantity))
        # Do not force a contract when it would exceed a 25% risk tolerance.
        if quantity < 1 and raw_quantity < 0.80:
            return 0
        return self._sign(forecast) * max(1, quantity)

    def _apply_targets(self, targets):
        for symbol in list(self.held_symbols):
            if symbol not in targets and symbol in self.securities and self.securities[symbol].invested:
                self.liquidate(symbol, tag="removed forecast")
            if symbol not in targets:
                self.held_symbols.discard(symbol)
        for symbol, target in targets.items():
            current = int(self.portfolio[symbol].quantity)
            delta = target - current
            # A one-contract no-trade band controls weekly turnover.
            if current != 0 and target != 0 and abs(delta) < 2:
                self.counts["band_skips"] += 1
                continue
            self.market_order(symbol, int(delta), tag="risk parity CTA")
            self.held_symbols.add(symbol)
            self.counts["orders"] += 1

    def on_symbol_changed_events(self, symbol_changed_events):
        for _, event in symbol_changed_events.items():
            old_symbol = event.old_symbol
            if old_symbol in self.held_symbols and self.portfolio[old_symbol].invested:
                self.liquidate(old_symbol, tag="continuous contract roll")
                self.held_symbols.discard(old_symbol)

    def _record_oos(self):
        if self.is_warming_up:
            return
        for label, start, end in self.oos_blocks:
            if start <= self.time <= end:
                record = self.oos_equity[label]
                if record["start"] is None:
                    record["start"] = self.portfolio.total_portfolio_value
                record["end"] = self.portfolio.total_portfolio_value

    @staticmethod
    def _daily_volatility(closes, lookback):
        returns = [closes[i] / closes[i + 1] - 1.0 for i in range(lookback)
                   if closes[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))

    @staticmethod
    def _robust_location_scale(values):
        if len(values) < 2:
            return 0.0, 0.0
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        deviations = sorted([abs(value - median) for value in values])
        return median, max(deviations[len(deviations) // 2], 0.0001)

    @staticmethod
    def _drawdown_scale(drawdown):
        return 1.0 if drawdown < 0.08 else 0.75 if drawdown < 0.12 else 0.50 if drawdown < 0.18 else 0.25

    @staticmethod
    def _clip(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def _sign(value):
        return 1 if value > 0 else -1 if value < 0 else 0

    def on_end_of_algorithm(self):
        self._record_oos()
        oos_returns = {}
        for label, record in self.oos_equity.items():
            if record["start"] is None or record["end"] is None:
                oos_returns[label] = None
            else:
                oos_returns[label] = round(100.0 * (record["end"] / record["start"] - 1.0), 2)
        self.debug("INSTITUTIONAL CTA COUNTS: {}".format(self.counts))
        self.debug("FIXED-PARAMETER OOS RETURNS (%): {}".format(oos_returns))
