from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class AcademicTimeSeriesMomentumV1(QCAlgorithm):
    """Transparent implementation of cross-asset 12-month TSMOM.

    Based on the public time-series momentum methodology: each liquid future
    is long when its own trailing 12-month return is positive and short when
    negative. This version deliberately excludes unvalidated AI, carry and
    relative-value overlays. Risk, rather than forecast complexity, is the
    portfolio construction mechanism.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        # Standard futures require whole contracts. This research NAV permits
        # 30+ markets to receive an actual risk budget; results are reported
        # as percentages and remain scale independent.
        self.set_cash(5_000_000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True
        self.target_volatility = 0.10

        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31)),
        ]
        self.oos_equity = {name: {"start": None, "end": None}
                           for name, _, _ in self.oos_blocks}

        specs = {
            "gold": (Futures.Metals.GOLD, "metals"),
            "silver": (Futures.Metals.SILVER, "metals"),
            "copper": (Futures.Metals.COPPER, "metals"),
            "platinum": (Futures.Metals.PLATINUM, "metals"),
            "palladium": (Futures.Metals.PALLADIUM, "metals"),
            "crude": (Futures.Energy.CRUDE_OIL_WTI, "energy"),
            "natgas": (Futures.Energy.NATURAL_GAS, "energy"),
            "gasoline": (Futures.Energy.GASOLINE, "energy"),
            "heating_oil": (Futures.Energy.HEATING_OIL, "energy"),
            "corn": (Futures.Grains.CORN, "grains"),
            "wheat": (Futures.Grains.WHEAT, "grains"),
            "soybeans": (Futures.Grains.SOYBEANS, "grains"),
            "soymeal": (Futures.Grains.SOYBEAN_MEAL, "grains"),
            "soyoil": (Futures.Grains.SOYBEAN_OIL, "grains"),
            "oats": (Futures.Grains.OATS, "grains"),
            "live_cattle": (Futures.Meats.LIVE_CATTLE, "meats"),
            "lean_hogs": (Futures.Meats.LEAN_HOGS, "meats"),
            "feeder_cattle": (Futures.Meats.FEEDER_CATTLE, "meats"),
            "sugar": (Futures.Softs.SUGAR_11, "softs"),
            "lumber": (Futures.Forestry.LUMBER, "softs"),
            "sp500": (Futures.Indices.SP_500_E_MINI, "equity"),
            "nasdaq": (Futures.Indices.NASDAQ_100_E_MINI, "equity"),
            "russell": (Futures.Indices.RUSSELL_2000_E_MINI, "equity"),
            "dow": (Futures.Indices.DOW_30_E_MINI, "equity"),
            "ust2": (Futures.Financials.Y_2_TREASURY_NOTE, "rates"),
            "ust5": (Futures.Financials.Y_5_TREASURY_NOTE, "rates"),
            "ust10": (Futures.Financials.Y_10_TREASURY_NOTE, "rates"),
            "ust30": (Futures.Financials.Y_30_TREASURY_BOND, "rates"),
            "aud": (Futures.Currencies.AUD, "fx"),
            "cad": (Futures.Currencies.CAD, "fx"),
            "chf": (Futures.Currencies.CHF, "fx"),
            "eur": (Futures.Currencies.EUR, "fx"),
            "gbp": (Futures.Currencies.GBP, "fx"),
            "jpy": (Futures.Currencies.JPY, "fx"),
            "mxn": (Futures.Currencies.MXN, "fx"),
        }
        self.markets, self.groups, self.closes = {}, {}, {}
        self.held_symbols = set()
        for name, (ticker, group) in specs.items():
            future = self.add_future(
                ticker, Resolution.DAILY, extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                contract_depth_offset=0)
            future.set_filter(0, 180)
            self.markets[name] = future
            self.groups[name] = group
            self.closes[name] = RollingWindow[float](300)

        self.high_water = self.portfolio.total_portfolio_value
        self.counts = {"ready": 0, "long": 0, "short": 0, "targets": 0,
                       "orders": 0, "vol_skips": 0, "risk_scale": 0}
        self.schedule.on(self.date_rules.month_start(), self.time_rules.at(13, 0), self.rebalance)
        self.schedule.on(self.date_rules.month_end(), self.time_rules.at(15, 30), self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for name, future in self.markets.items():
            if future.symbol in data.bars:
                close = float(data.bars[future.symbol].close)
                if close > 0:
                    self.closes[name].add(close)

    def rebalance(self):
        if self.is_warming_up:
            return
        self.high_water = max(self.high_water, self.portfolio.total_portfolio_value)
        dd = 1.0 - self.portfolio.total_portfolio_value / max(self.high_water, 1.0)
        risk_scale = 1.0 if dd < 0.10 else 0.60 if dd < 0.15 else 0.35
        self.counts["risk_scale"] += int(risk_scale < 1.0)

        rows, by_group = [], {}
        for name, closes in self.closes.items():
            if closes.count < 253:
                continue
            daily_vol = self._daily_volatility(closes, 63)
            if daily_vol <= 0:
                continue
            direction = self._sign(closes[0] / closes[252] - 1.0)
            if direction == 0:
                continue
            row = {"name": name, "direction": direction, "vol": daily_vol}
            rows.append(row)
            by_group.setdefault(self.groups[name], []).append(row)
            self.counts["ready"] += 1
            self.counts["long" if direction > 0 else "short"] += 1

        targets = {}
        group_count = len(by_group)
        for group, members in by_group.items():
            # Equal risk between economic groups, then equal risk between the
            # markets inside a group. This is intentionally fixed ex-ante.
            risk_budget = self.portfolio.total_portfolio_value * self.target_volatility * risk_scale
            risk_budget /= math.sqrt(252.0 * group_count * len(members))
            for row in members:
                mapped = self.markets[row["name"]].mapped
                quantity = self._quantity(mapped, row["vol"], risk_budget, row["direction"])
                if quantity:
                    targets[mapped] = quantity
                else:
                    self.counts["vol_skips"] += 1
        self._apply_targets(targets)
        self.counts["targets"] += len(targets)

    def _quantity(self, symbol, daily_vol, risk_budget, direction):
        if symbol is None or symbol not in self.securities or not self.securities[symbol].is_tradable:
            return 0
        security = self.securities[symbol]
        risk_per_contract = float(security.price) * float(security.symbol_properties.contract_multiplier) * daily_vol
        if risk_per_contract <= 0:
            return 0
        quantity = int(math.floor(risk_budget / risk_per_contract))
        return direction * quantity if quantity >= 1 else 0

    def _apply_targets(self, targets):
        for symbol in list(self.held_symbols):
            if symbol not in targets and symbol in self.securities and self.portfolio[symbol].invested:
                self.liquidate(symbol, tag="monthly momentum rebalance")
            if symbol not in targets:
                self.held_symbols.discard(symbol)
        for symbol, target in targets.items():
            current = int(self.portfolio[symbol].quantity)
            delta = target - current
            if delta:
                self.market_order(symbol, int(delta), tag="12m time-series momentum")
                self.counts["orders"] += 1
            self.held_symbols.add(symbol)

    def on_symbol_changed_events(self, events):
        for _, event in events.items():
            old_symbol = event.old_symbol
            if old_symbol in self.held_symbols and self.portfolio[old_symbol].invested:
                self.liquidate(old_symbol, tag="continuous contract roll")
            self.held_symbols.discard(old_symbol)

    def _record_oos(self):
        if self.is_warming_up:
            return
        for name, start, end in self.oos_blocks:
            if start <= self.time <= end:
                record = self.oos_equity[name]
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
    def _sign(value):
        return 1 if value > 0 else -1 if value < 0 else 0

    def on_end_of_algorithm(self):
        self._record_oos()
        oos = {}
        for name, record in self.oos_equity.items():
            oos[name] = (round(100 * (record["end"] / record["start"] - 1), 2)
                         if record["start"] is not None and record["end"] is not None else None)
        self.debug("ACADEMIC TSMOM COUNTS: {}".format(self.counts))
        self.debug("FIXED-PARAMETER OOS RETURNS (%): {}".format(oos))
