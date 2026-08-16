from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class RealFuturesMultiSleeveV1(QCAlgorithm):
    """35-market managed-futures prototype with independent return sleeves.

    Trend uses a backwards-ratio continuous series. Carry reads the actual
    nearest and next eligible contracts from each FutureChain, whose prices
    remain RAW settlement prices. Orders are sent only to mapped contracts.
    """
    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True
        # This is a portfolio-level *ex-ante* annual volatility budget.  The
        # order quantity calculation below converts it to each contract's
        # estimated daily dollar risk; it is not a performance target.
        self.target_portfolio_volatility = 0.15
        # Parameters are fixed before the out-of-sample windows below.  These
        # records make it possible to reject the framework if one or more
        # independent periods fail, rather than selecting a flattering total.
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
            # Equity index
            "sp500": (Futures.Indices.SP_500_E_MINI, "equity", False),
            "nasdaq": (Futures.Indices.NASDAQ_100_E_MINI, "equity", False),
            "russell": (Futures.Indices.RUSSELL_2000_E_MINI, "equity", False),
            "dow": (Futures.Indices.DOW_30_E_MINI, "equity", False),
            # Rates
            "ust2": (Futures.Financials.Y_2_TREASURY_NOTE, "rates", False),
            "ust5": (Futures.Financials.Y_5_TREASURY_NOTE, "rates", False),
            "ust10": (Futures.Financials.Y_10_TREASURY_NOTE, "rates", False),
            "ust30": (Futures.Financials.Y_30_TREASURY_BOND, "rates", False),
            # FX
            "aud": (Futures.Currencies.AUD, "fx", False),
            "cad": (Futures.Currencies.CAD, "fx", False),
            "chf": (Futures.Currencies.CHF, "fx", False),
            "eur": (Futures.Currencies.EUR, "fx", False),
            "gbp": (Futures.Currencies.GBP, "fx", False),
            "jpy": (Futures.Currencies.JPY, "fx", False),
            "mxn": (Futures.Currencies.MXN, "fx", False),
        }
        self.markets = {}
        self.states = {}
        self.group_of = {}
        self.roots = {}
        self.held_symbols = set()
        self.emergency_until = datetime.min
        for name, (ticker, group, has_carry) in specs.items():
            trend = self.add_future(ticker, Resolution.DAILY, extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                contract_depth_offset=0)
            # The wider chain includes both near and deferred raw contracts
            # used by the carry sleeve. This avoids duplicate continuous
            # subscriptions for one root market.
            trend.set_filter(0, 360)
            self.markets[name] = {"trend": trend, "has_carry": has_carry}
            self.states[name] = {"closes": RollingWindow[float](260), "carry": None}
            self.group_of[name] = group
            self.roots[name] = ticker

        self.peak = self.portfolio.total_portfolio_value
        self.counts = {"ready": 0, "trend": 0, "carry": 0, "relative_value": 0,
                       "selected": 0, "orders": 0, "mapped_missing": 0,
                       "volatility_skipped": 0, "risk_off": 0}
        self.schedule.on(self.date_rules.month_start(), self.time_rules.at(13, 0), self.rebalance)
        self.schedule.on(self.date_rules.month_end(), self.time_rules.at(15, 30), self._record_oos_equity)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for name, market in self.markets.items():
            state = self.states[name]
            trend_symbol = market["trend"].symbol
            if trend_symbol in data.bars:
                price = float(data.bars[trend_symbol].close)
                if price > 0:
                    state["closes"].add(price)
            if market["has_carry"]:
                # FutureChain contracts are individual, unadjusted futures.
                # Selecting the two nearest valid expiries makes this a true
                # front/deferred curve factor, not an ETF approximation.
                chain = data.future_chains.get(trend_symbol)
                if chain:
                    contracts = sorted(
                        [contract for contract in chain
                         if contract.last_price > 0
                         and contract.expiry > self.time + timedelta(days=20)],
                        key=lambda contract: contract.expiry)
                    if len(contracts) >= 2:
                        front_contract = contracts[0]
                        # Skip the immediately adjacent contract if necessary.
                        # A 45-day spread measures the investable curve, rather
                        # than a delivery-week technical dislocation.
                        far_contract = next(
                            (contract for contract in contracts[1:]
                             if (contract.expiry - front_contract.expiry).days >= 45), None)
                        if far_contract is not None:
                            front = float(front_contract.last_price)
                            far = float(far_contract.last_price)
                            days = (far_contract.expiry - front_contract.expiry).days
                            if front > 0 and far > 0 and days > 0:
                                state["carry"] = math.log(far / front) * 365.0 / days

    def rebalance(self):
        if self.is_warming_up:
            return
        self.peak = max(self.peak, self.portfolio.total_portfolio_value)
        dd = 1.0 - self.portfolio.total_portfolio_value / max(self.peak, 1.0)
        if self.time < self.emergency_until:
            self.counts["risk_off"] += 1
            self._liquidate_all("emergency circuit cooldown")
            return
        # A permanent cash stop cannot recover because equity cannot make a
        # new high in cash. This is a graduated circuit: it cuts risk from six
        # markets to four, two, then one while retaining a controlled path to
        # recovery and preserving a hard 25% emergency stop.
        slots = 6 if dd < 0.08 else 4 if dd < 0.12 else 2 if dd < 0.20 else 1
        if dd >= 0.25:
            self.counts["risk_off"] += 1
            self._liquidate_all("25 percent emergency circuit")
            # A circuit breaker should not create a permanent all-cash
            # backtest. After one quarter it resets the high-water reference
            # and allows one risk slot; recovery must still be earned.
            self.emergency_until = self.time + timedelta(days=90)
            self.peak = self.portfolio.total_portfolio_value
            return

        rows = []
        group_returns = {}
        for name, state in self.states.items():
            closes = state["closes"]
            if closes.count < 253:
                continue
            r1, r3, r12 = closes[0] / closes[20] - 1.0, closes[0] / closes[63] - 1.0, closes[0] / closes[252] - 1.0
            trend = (self._sign(r1) + self._sign(r3) + self._sign(r12)) / 3.0
            daily_vol = self._daily_volatility(closes, 60)
            if daily_vol <= 0:
                continue
            # Do not rank all unanimous trends as equal. This comparable
            # strength statistic favours the largest move per unit of recent
            # realised volatility across the full futures universe.
            trend_strength = abs(0.20 * r1 + 0.30 * r3 + 0.50 * r12) / daily_vol
            ret63 = r3
            group_returns.setdefault(self.group_of[name], []).append(ret63)
            rows.append({"name": name, "trend": trend, "trend_strength": trend_strength,
                         "ret63": ret63, "carry": state["carry"]})

        # Keep the three sources independent.  Their candidates are selected
        # independently before de-duplication, so a good carry signal cannot
        # be hidden by a weak trend score (or vice versa).
        trend_candidates, carry_candidates, rv_candidates = [], [], []
        for row in rows:
            group = self.group_of[row["name"]]
            peers = group_returns[group]
            median = sorted(peers)[len(peers) // 2]
            rv = self._sign(row["ret63"] - median) if len(peers) > 1 else 0
            carry_signal = -self._sign(row["carry"]) if row["carry"] is not None else 0
            self.counts["ready"] += 1
            if row["trend"] != 0:
                self.counts["trend"] += 1
                trend_candidates.append((row["trend_strength"], row["name"], row["trend"], "trend"))
            # A positive far-minus-front curve is contango.  The carry sleeve
            # is therefore short contango / long backwardation.  It is only
            # used where a physically meaningful commodity curve exists.
            if carry_signal != 0:
                self.counts["carry"] += 1
                carry_candidates.append((abs(row["carry"]), row["name"], carry_signal, "carry"))
            if rv != 0:
                self.counts["relative_value"] += 1
                rv_candidates.append((abs(row["ret63"] - median), row["name"], rv, "relative_value"))

        # 3/2/1 is an intentional, fixed sleeve budget: medium/long horizon
        # trend gets half the risk, real commodity curve carry one third, and
        # within-sector relative value the remaining sixth.
        selected = self._select_independent_sleeves(
            [(trend_candidates, 3), (carry_candidates, 2), (rv_candidates, 1)], slots)
        self._liquidate_all("monthly sleeve rebalance")
        active_count = max(len(selected), 1)
        for _, name, direction, leg in selected:
            mapped = self.markets[name]["trend"].mapped
            if mapped is None or mapped not in self.securities or not self.securities[mapped].is_tradable:
                self.counts["mapped_missing"] += 1
                continue
            quantity = self._volatility_targeted_quantity(mapped, active_count)
            if quantity < 1:
                self.counts["volatility_skipped"] += 1
                continue
            self.market_order(mapped, self._sign(direction) * quantity, tag="{} {}".format(leg, name))
            self.held_symbols.add(mapped)
            self.counts["orders"] += 1
        self.counts["selected"] += len(selected)

    def _select_independent_sleeves(self, sleeve_budgets, total_slots):
        """Select independent legs while capping one market per asset group."""
        selected, used_names, used_groups = [], set(), set()
        for candidates, budget in sleeve_budgets:
            taken = 0
            for candidate in sorted(candidates, reverse=True):
                _, name, _, _ = candidate
                group = self.group_of[name]
                if taken >= budget or len(selected) >= total_slots:
                    break
                if name in used_names or group in used_groups:
                    continue
                selected.append(candidate)
                used_names.add(name)
                used_groups.add(group)
                taken += 1
        # Fill unused risk slots with the strongest remaining signal, but
        # retain the economic-group concentration limit.
        remainder = []
        for candidates, _ in sleeve_budgets:
            remainder.extend(candidates)
        for candidate in sorted(remainder, reverse=True):
            _, name, _, _ = candidate
            if len(selected) >= total_slots:
                break
            if name in used_names or self.group_of[name] in used_groups:
                continue
            selected.append(candidate)
            used_names.add(name)
            used_groups.add(self.group_of[name])
        return selected

    def _volatility_targeted_quantity(self, symbol, active_count):
        """Estimate integer contracts from 20-day realised volatility.

        The max(1, ...) fallback is deliberately limited by a 1.5x risk cap:
        it lets small accounts test liquid standard contracts without quietly
        turning the volatility target into unrestricted leverage.
        """
        name = next((key for key, value in self.markets.items()
                     if value["trend"].mapped == symbol), None)
        if name is None:
            return 0
        closes = self.states[name]["closes"]
        if closes.count < 22:
            return 0
        returns = [closes[i] / closes[i + 1] - 1.0 for i in range(20)
                   if closes[i + 1] > 0]
        if len(returns) < 15:
            return 0
        mean = sum(returns) / len(returns)
        daily_vol = math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))
        if daily_vol <= 0:
            return 0
        security = self.securities[symbol]
        multiplier = float(security.symbol_properties.contract_multiplier)
        per_contract_daily_risk = float(security.price) * multiplier * daily_vol
        if per_contract_daily_risk <= 0:
            return 0
        risk_budget = self.portfolio.total_portfolio_value * self.target_portfolio_volatility
        risk_budget /= math.sqrt(252.0 * active_count)
        quantity = int(math.floor(risk_budget / per_contract_daily_risk))
        if quantity >= 1:
            return quantity
        return 1 if per_contract_daily_risk <= 1.5 * risk_budget else 0

    @staticmethod
    def _daily_volatility(closes, lookback):
        if closes.count < lookback + 1:
            return 0.0
        returns = [closes[i] / closes[i + 1] - 1.0 for i in range(lookback)
                   if closes[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))

    def _liquidate_all(self, reason):
        # Never liquidate a continuous *canonical* subscription. It is a
        # data symbol, not the integer-lot traded contract, and attempting to
        # target it is what produces fractional-lot order warnings.
        for symbol in list(self.held_symbols):
            if symbol in self.securities and self.securities[symbol].invested:
                self.liquidate(symbol, tag=reason)
            self.held_symbols.discard(symbol)

    def _record_oos_equity(self):
        if self.is_warming_up:
            return
        value = self.portfolio.total_portfolio_value
        now = self.time
        for label, start, end in self.oos_blocks:
            if start <= now <= end:
                record = self.oos_equity[label]
                if record["start"] is None:
                    record["start"] = value
                record["end"] = value

    @staticmethod
    def _sign(value):
        return 1 if value > 0 else -1 if value < 0 else 0

    def on_end_of_algorithm(self):
        self._record_oos_equity()
        oos_returns = {}
        for label, record in self.oos_equity.items():
            if record["start"] is not None and record["end"] is not None:
                oos_returns[label] = round(100.0 * (record["end"] / record["start"] - 1.0), 2)
            else:
                oos_returns[label] = None
        self.debug("REAL FUTURES COUNTS: {}".format(self.counts))
        self.debug("FIXED-PARAMETER OOS RETURNS (%): {}".format(oos_returns))
