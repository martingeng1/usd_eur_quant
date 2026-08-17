from AlgorithmImports import *
from datetime import datetime, timedelta


class VixCarryProxyV2(QCAlgorithm):
    """Higher-return, capped short-volatility proxy with trading bands.

    V2 deliberately takes more risk than V1: 30% SVXY and 65% equity in a
    calm volatility regime. It is a research variant, not a low-drawdown or
    crash-safe strategy.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.svxy = self.add_equity("SVXY", Resolution.DAILY).symbol
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.qqq = self.add_equity("QQQ", Resolution.DAILY).symbol
        self.gld = self.add_equity("GLD", Resolution.DAILY).symbol
        self.ief = self.add_equity("IEF", Resolution.DAILY).symbol
        self.shy = self.add_equity("SHY", Resolution.DAILY).symbol
        self.vix = self.add_index("VIX", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)
        self.spy_sma200 = self.sma(self.spy, 200, Resolution.DAILY)
        self.spy_sma100 = self.sma(self.spy, 100, Resolution.DAILY)
        self.vix_sma20 = self.sma(self.vix, 20, Resolution.DAILY)
        self.peak, self.cooldown, self.brake_armed = 250000.0, 0, True
        self.trade_band = 0.04
        self.counts = {"rebalances": 0, "carry_on": 0, "defensive": 0,
                       "vix_rejected": 0, "trend_rejected": 0,
                       "drawdown_brakes": 0, "orders": 0, "band_skips": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}
        self.schedule.on(self.date_rules.every_day(self.spy), self.time_rules.before_market_close(self.spy, 1), self._update_peak)
        self.schedule.on(self.date_rules.week_start(self.spy), self.time_rules.after_market_open(self.spy, 30), self.rebalance)
        self.schedule.on(self.date_rules.month_end(self.spy), self.time_rules.before_market_close(self.spy, 1), self._record_oos)
        self.set_warm_up(timedelta(days=300))

    def on_data(self, data):
        pass

    def _update_peak(self):
        if self.is_warming_up:
            return
        equity = self.portfolio.total_portfolio_value
        self.peak = max(self.peak, equity)
        dd = 1.0 - equity / max(1.0, self.peak)
        if dd < 0.08:
            self.brake_armed = True
        self.cooldown = max(0, self.cooldown - 1)

    def rebalance(self):
        if self.is_warming_up or not self.spy_sma200.is_ready or not self.vix_sma20.is_ready:
            return
        self.counts["rebalances"] += 1
        dd = 1.0 - self.portfolio.total_portfolio_value / max(1.0, self.peak)
        if dd >= 0.12 and self.brake_armed:
            self.cooldown, self.brake_armed = 21, False
            self.counts["drawdown_brakes"] += 1
        spy_price, vix_price = self.securities[self.spy].price, self.securities[self.vix].price
        trend_ok = spy_price > self.spy_sma200.current.value and spy_price > self.spy_sma100.current.value
        vix_ok = vix_price > 0 and vix_price < 25 and vix_price <= 1.15 * self.vix_sma20.current.value
        if self.cooldown == 0 and trend_ok and vix_ok:
            # Higher return target: 30% short-vol proxy, 65% equity growth.
            weights = {self.svxy: 0.30, self.qqq: 0.45, self.spy: 0.20,
                       self.gld: 0.05, self.ief: 0.0, self.shy: 0.0}
            self.counts["carry_on"] += 1
        else:
            if not trend_ok:
                self.counts["trend_rejected"] += 1
            if not vix_ok:
                self.counts["vix_rejected"] += 1
            weights = {self.svxy: 0.0, self.qqq: 0.0, self.spy: 0.0,
                       self.gld: 0.20, self.ief: 0.35, self.shy: 0.45}
            self.counts["defensive"] += 1
        self._apply_with_band(weights)

    def _apply_with_band(self, weights):
        value = max(1.0, self.portfolio.total_portfolio_value)
        should_trade = False
        for symbol, target in weights.items():
            current = self.portfolio[symbol].holdings_value / value
            if abs(target - current) >= self.trade_band:
                should_trade = True
            else:
                self.counts["band_skips"] += 1
        if should_trade:
            targets = [PortfolioTarget(symbol, target) for symbol, target in weights.items()]
            self.set_holdings(targets, liquidate_existing_holdings=True)
            self.counts["orders"] += len(targets)

    def _record_oos(self):
        if self.is_warming_up:
            return
        for label, start, end in self.oos_blocks:
            if start <= self.time <= end:
                if self.oos[label]["start"] is None:
                    self.oos[label]["start"] = self.portfolio.total_portfolio_value
                self.oos[label]["end"] = self.portfolio.total_portfolio_value

    def on_end_of_algorithm(self):
        self._record_oos()
        output = {}
        for label, values in self.oos.items():
            if values["start"] and values["end"]:
                output[label] = round(100 * (values["end"] / values["start"] - 1.0), 2)
        self.debug("VIX CARRY PROXY V2 COUNTS: {}".format(self.counts))
        self.debug("VIX CARRY PROXY V2 FIXED-PARAMETER OOS RETURNS (%): {}".format(output))
