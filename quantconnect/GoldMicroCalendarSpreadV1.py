from AlgorithmImports import *
from datetime import time, timedelta
import math


class GoldMicroCalendarSpreadV1(QCAlgorithm):
    """Five-minute, market-neutral Micro Gold calendar-spread strategy.

    The strategy trades two actual MGC contracts, not a continuous-price
    proxy. It measures the log near/far spread, enters only when that spread
    is unusually far from its intraday mean, and exits on normalization or
    before the COMEX session ends. This is a testable relative-value design,
    not a directional forecast of gold and not a performance guarantee.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True

        self.gold = self.add_future(
            Futures.Metals.MICRO_GOLD, Resolution.MINUTE,
            fill_forward=False, extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.RAW,
            contract_depth_offset=0)
        # A wide filter makes both legs observable in the FutureChain. We
        # subscribe to the selected individual contracts below before trading.
        self.gold.set_filter(0, 240)

        self.near = None
        self.far = None
        self.near_expiry = None
        self.far_expiry = None
        self.spreads = RollingWindow[float](96)  # eight hours of 5m samples
        self.entry_bar = None
        self.position = 0
        self.day_start_value = self.portfolio.total_portfolio_value
        self.day = None
        self.counts = {"pair_changes": 0, "five_bars": 0, "not_ready": 0,
                       "liquidity_skips": 0, "entries": 0, "mean_exits": 0,
                       "time_exits": 0, "daily_stops": 0}
        self.set_warm_up(timedelta(days=10))

    def on_data(self, data):
        if self.day != self.time.date():
            self.day = self.time.date()
            self.day_start_value = self.portfolio.total_portfolio_value

        chain = data.future_chains.get(self.gold.symbol)
        if chain:
            self._select_pair(chain)

        if self.near is None or self.far is None:
            return
        if self.time.minute % 5 != 0:
            return
        near_bar = data.bars.get(self.near)
        far_bar = data.bars.get(self.far)
        if near_bar is None or far_bar is None:
            return
        self._on_five_minute(near_bar, far_bar)

    def _select_pair(self, chain):
        # Avoid delivery-period noise. The selected pair has a stable, real
        # expiry gap and both legs can be traded as individual contracts.
        contracts = sorted(
            [contract for contract in chain
             if contract.last_price > 0 and contract.expiry > self.time + timedelta(days=15)],
            key=lambda contract: contract.expiry)
        if len(contracts) < 2:
            return
        front = contracts[0]
        deferred = next(
            (contract for contract in contracts[1:]
             if (contract.expiry - front.expiry).days >= 45), None)
        if deferred is None:
            return
        if front.symbol == self.near and deferred.symbol == self.far:
            return

        self._close_spread("contract pair roll")
        self.near = self.add_future_contract(
            front.symbol, Resolution.MINUTE, fill_forward=False,
            extended_market_hours=True).symbol
        self.far = self.add_future_contract(
            deferred.symbol, Resolution.MINUTE, fill_forward=False,
            extended_market_hours=True).symbol
        self.near_expiry, self.far_expiry = front.expiry, deferred.expiry
        self.spreads.reset()
        self.position, self.entry_bar = 0, None
        self.counts["pair_changes"] += 1

    def _on_five_minute(self, near_bar, far_bar):
        if self.is_warming_up:
            return
        self.counts["five_bars"] += 1
        # Regular COMEX liquidity window. This also avoids scheduled release
        # minutes, whose price changes are usually directional rather than a
        # tradable calendar-spread dislocation.
        now = self.time.time()
        if now < time(8, 35) or now > time(15, 40):
            if self.position != 0:
                self._close_spread("outside liquid session")
                self.counts["time_exits"] += 1
            return
        if near_bar.volume <= 0 or far_bar.volume <= 0:
            self.counts["liquidity_skips"] += 1
            return

        spread = math.log(float(near_bar.close) / float(far_bar.close))
        if self.spreads.is_ready:
            values = [self.spreads[i] for i in range(self.spreads.count)]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            deviation = math.sqrt(variance)
            z_score = (spread - mean) / deviation if deviation > 1e-6 else 0.0
            self._trade_signal(z_score)
        else:
            self.counts["not_ready"] += 1
        self.spreads.add(spread)

    def _trade_signal(self, z_score):
        # Portfolio-level daily stop: the strategy is market-neutral, so a
        # 0.75% daily loss indicates model or liquidity stress, not an alpha.
        if self.portfolio.total_portfolio_value < 0.9925 * self.day_start_value:
            if self.position != 0:
                self._close_spread("daily spread risk stop")
            self.counts["daily_stops"] += 1
            return

        if self.position == 0:
            if abs(z_score) < 2.25:
                return
            quantity = self._pair_quantity()
            if quantity < 1:
                return
            # Positive z: near is expensive vs far -> short near, long far.
            direction = -1 if z_score > 0 else 1
            self.market_order(self.near, direction * quantity, tag="MGC calendar near")
            self.market_order(self.far, -direction * quantity, tag="MGC calendar far")
            self.position = direction
            self.entry_bar = self.counts["five_bars"]
            self.counts["entries"] += 1
            return

        held_bars = self.counts["five_bars"] - self.entry_bar
        # Mean reversion, a modest stop, and a maximum 90-minute holding time
        # prevent the strategy from converting a curve anomaly into an
        # overnight macro position.
        if abs(z_score) < 0.40 or abs(z_score) > 3.50 or held_bars >= 18:
            self._close_spread("spread normalization/stop/time")
            self.counts["mean_exits"] += 1

    def _pair_quantity(self):
        if self.near not in self.securities or self.far not in self.securities:
            return 0
        if self.securities[self.near].price <= 0 or self.securities[self.far].price <= 0:
            return 0
        # Micro Gold has the same multiplier on each leg. Cap at five pairs:
        # the intended test is an intraday relative-value signal, not leverage.
        return min(5, max(1, int(self.portfolio.total_portfolio_value // 100000)))

    def _close_spread(self, reason):
        for symbol in (self.near, self.far):
            if symbol is not None and symbol in self.securities and self.portfolio[symbol].invested:
                self.liquidate(symbol, tag=reason)
        self.position, self.entry_bar = 0, None

    def on_end_of_algorithm(self):
        self.debug("MGC CALENDAR COUNTS: {}".format(self.counts))
