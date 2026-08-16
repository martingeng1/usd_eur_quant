from AlgorithmImports import *
from datetime import timedelta
import math


class GoldAuctionOrderFlowV1(QCAlgorithm):
    """Auction/effort-vs-result strategy inspired by the supplied article.

    This is an executable proxy for footprint trading: free futures history
    does not contain historical bid/ask-at-price or option GEX, so signed
    volume (bar direction * volume) is used as a conservative Delta proxy.
    """
    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True

        self.future = self.add_future(
            Futures.Metals.GOLD, Resolution.MINUTE,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0)
        self.future.set_filter(0, 60)

        self.h1 = RollingWindow[TradeBar](120)
        self.m5 = RollingWindow[TradeBar](48)
        self.signed_volume = RollingWindow[float](48)
        self.volume_window = RollingWindow[float](48)
        self.atr_h = self.atr(self.future.symbol, 20, MovingAverageType.WILDERS, Resolution.HOUR)
        self.atr_5 = AverageTrueRange(20, MovingAverageType.WILDERS)
        self.consolidate(self.future.symbol, timedelta(hours=1), self.on_hour)
        self.consolidate(self.future.symbol, timedelta(minutes=5), self.on_five)

        self.setup = None       # first failed auction attempt
        self.position = None
        self.trades_today = 0
        self.losses_today = 0
        self.session_date = None
        self.counts = {"hour_bars": 0, "five_bars": 0, "context": 0,
                       "effort_failures": 0, "second_tests": 0,
                       "entries": 0, "exits": 0}
        self.set_warm_up(timedelta(days=90))

    def on_hour(self, bar):
        self.h1.add(bar)
        self.counts["hour_bars"] += 1

    def on_five(self, bar):
        self.counts["five_bars"] += 1
        self.atr_5.update(bar)
        if self.is_warming_up or not self.atr_5.is_ready or self.h1.count < 45:
            self.m5.add(bar)
            # Inline the proxy update so pasted QuantConnect files cannot
            # fail because a helper method was omitted during copying.
            signed = float(bar.volume) * (1.0 if bar.close > bar.open else -1.0 if bar.close < bar.open else 0.0)
            self.signed_volume.add(signed)
            self.volume_window.add(float(bar.volume))
            return

        if self.session_date != self.time.date():
            self.session_date = self.time.date()
            self.trades_today = 0
            self.losses_today = 0
            self.setup = None

        self._manage_position(bar)
        self.m5.add(bar)
        signed = float(bar.volume) * (1.0 if bar.close > bar.open else -1.0 if bar.close < bar.open else 0.0)
        self.signed_volume.add(signed)
        self.volume_window.add(float(bar.volume))
        if self.position is not None or self.trades_today >= 2 or self.losses_today >= 2:
            return

        # Article focuses on the liquid New York opening auction.
        if not (self.time.hour == 9 and self.time.minute >= 30 or self.time.hour == 10):
            self.setup = None
            return

        direction, zone_low, zone_high = self._auction_context()
        if direction == 0:
            self.setup = None
            return
        self.counts["context"] += 1
        price = float(bar.close)
        if not (zone_low <= price <= zone_high):
            return

        effort = self._effort_failure(bar, direction)
        if self.setup is None and effort:
            extreme = float(bar.low if direction > 0 else bar.high)
            self.setup = {"direction": direction, "extreme": extreme,
                          "created": self.time, "tests": 0}
            self.counts["effort_failures"] += 1
            return

        if self.setup is None or self.setup["direction"] != direction:
            return
        if (self.time - self.setup["created"]).total_seconds() > 3600:
            self.setup = None
            return

        # The second test must fail closer to value than the first attempt.
        atr = float(self.atr_5.current.value)
        second_test = (direction > 0 and bar.low <= self.setup["extreme"] + .50 * atr and bar.close > self.setup["extreme"] + .20 * atr) or \
                      (direction < 0 and bar.high >= self.setup["extreme"] - .50 * atr and bar.close < self.setup["extreme"] - .20 * atr)
        if second_test:
            self.setup["tests"] = 1
            self.counts["second_tests"] += 1

        if self.setup.get("tests", 0) == 1 and self._confirmation(bar, direction):
            self._enter(direction, bar)
            self.setup = None

    def _add_flow(self, bar):
        signed = float(bar.volume) * (1.0 if bar.close > bar.open else -1.0 if bar.close < bar.open else 0.0)
        self.signed_volume.add(signed)
        self.volume_window.add(float(bar.volume))

    def _auction_context(self):
        """Value migration plus Fibonacci location, using hourly bars."""
        if self.h1.count < 45 or not self.atr_h.is_ready:
            return 0, 0.0, 0.0
        recent = [self.h1[i] for i in range(20)]
        prior = [self.h1[i] for i in range(20, 40)]
        recent_vwap = sum(x.close * max(float(x.volume), 1) for x in recent) / sum(max(float(x.volume), 1) for x in recent)
        prior_vwap = sum(x.close * max(float(x.volume), 1) for x in prior) / sum(max(float(x.volume), 1) for x in prior)
        threshold = float(self.atr_h.current.value) * .20
        direction = 1 if recent_vwap > prior_vwap + threshold else -1 if recent_vwap < prior_vwap - threshold else 0
        if direction == 0:
            return 0, 0.0, 0.0
        swing_high = max(float(x.high) for x in [self.h1[i] for i in range(40)])
        swing_low = min(float(x.low) for x in [self.h1[i] for i in range(40)])
        span = max(swing_high - swing_low, 1e-8)
        if direction > 0:
            return direction, swing_high - .886 * span, swing_high - .705 * span
        return direction, swing_low + .705 * span, swing_low + .886 * span

    def _effort_failure(self, bar, direction):
        if self.signed_volume.count < 20 or self.volume_window.count < 20:
            return False
        avg_vol = sum(self.volume_window[i] for i in range(20)) / 20.0
        avg_abs = sum(abs(self.signed_volume[i]) for i in range(20)) / 20.0
        signed = self.signed_volume[0]
        rng = max(float(bar.high - bar.low), 1e-8)
        body = abs(float(bar.close - bar.open))
        # High effort, but small result and rejection wick.
        if float(bar.volume) < 1.25 * avg_vol or abs(signed) < 1.5 * max(avg_abs, 1):
            return False
        if direction > 0:
            return signed < 0 and body < .65 * rng and bar.close > bar.low + .55 * rng
        return signed > 0 and body < .65 * rng and bar.close < bar.high - .55 * rng

    def _confirmation(self, bar, direction):
        # on_five adds the current bar before confirmation; compare only
        # against completed bars preceding it, never against itself.
        if self.m5.count < 4:
            return False
        prior_high = max(float(self.m5[i].high) for i in range(1, 4))
        prior_low = min(float(self.m5[i].low) for i in range(1, 4))
        signed = self.signed_volume[0]
        return (direction > 0 and bar.close > prior_high and signed > 0) or (direction < 0 and bar.close < prior_low and signed < 0)

    def _enter(self, direction, bar):
        mapped = self.future.mapped
        if mapped is None or mapped not in self.securities or not self.securities[mapped].is_tradable:
            return
        price = float(self.securities[mapped].price)
        atr = float(self.atr_5.current.value)
        extreme = float(self.setup["extreme"])
        stop = extreme - .25 * atr if direction > 0 else extreme + .25 * atr
        risk = abs(price - stop)
        if risk <= 0 or price <= 0:
            return
        # Micro-style risk cap; one contract maximum keeps futures exposure bounded.
        multiplier = float(self.securities[mapped].symbol_properties.contract_multiplier)
        if risk * multiplier > self.portfolio.total_portfolio_value * .025:
            return
        self.market_order(mapped, direction, tag="auction effort failure")
        self.position = {"symbol": mapped, "direction": direction, "entry": price,
                         "stop": stop, "target": price + direction * 1.8 * risk,
                         "bars": 0}
        self.trades_today += 1
        self.counts["entries"] += 1

    def _manage_position(self, bar):
        if self.position is None:
            return
        p = self.position
        if p["symbol"] not in self.portfolio or not self.portfolio[p["symbol"]].invested:
            self.position = None
            return
        p["bars"] += 1
        price = float(self.securities[p["symbol"]].price)
        hit_stop = price <= p["stop"] if p["direction"] > 0 else price >= p["stop"]
        hit_target = price >= p["target"] if p["direction"] > 0 else price <= p["target"]
        if hit_stop or hit_target or p["bars"] >= 18:
            self.liquidate(p["symbol"], tag="auction exit")
            if hit_stop:
                self.losses_today += 1
            self.counts["exits"] += 1
            self.position = None

    def on_end_of_algorithm(self):
        self.debug(f"AUCTION COUNTS: {self.counts}")
