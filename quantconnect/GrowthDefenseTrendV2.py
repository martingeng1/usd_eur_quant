from AlgorithmImports import *
from datetime import datetime, timedelta


class GrowthDefenseTrendV2(QCAlgorithm):
    """Growth core with faster trend de-risking and atomic target rebalancing.

    V2 fixes the sequential-order buying-power issue in V1.  It also uses a
    short/medium trend downgrade to reduce equity exposure before a 200-day
    trend break. This is research code, not a performance guarantee.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.tickers = ["SPY", "QQQ", "GLD", "IEF", "TLT", "SHY"]
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)
        self.set_benchmark(self.symbols["SPY"])

        self.high_water_mark = 250000.0
        self.cooldown, self.brake_armed = 0, True
        self.trade_band = 0.04
        self.counts = {"rebalances": 0, "full_growth": 0, "reduced_growth": 0,
                       "defense": 0, "brakes": 0, "orders": 0, "band_skips": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}

        spy = self.symbols["SPY"]
        self.schedule.on(self.date_rules.every_day(spy), self.time_rules.before_market_close(spy, 1),
                         self._update_risk_state)
        self.schedule.on(self.date_rules.month_start(spy), self.time_rules.after_market_open(spy, 30),
                         self.rebalance)
        self.schedule.on(self.date_rules.month_end(spy), self.time_rules.before_market_close(spy, 1),
                         self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for ticker, symbol in self.symbols.items():
            if symbol in data.bars:
                close = float(data.bars[symbol].close)
                if close > 0:
                    self.closes[ticker].add(close)

    def _update_risk_state(self):
        if self.is_warming_up:
            return
        equity = self.portfolio.total_portfolio_value
        self.high_water_mark = max(self.high_water_mark, equity)
        drawdown = 1.0 - equity / max(self.high_water_mark, 1.0)
        if drawdown < 0.08:
            self.brake_armed = True
        self.cooldown = max(0, self.cooldown - 1)

    def rebalance(self):
        if self.is_warming_up or not all(self._ready(ticker) for ticker in self.tickers):
            return
        self.counts["rebalances"] += 1
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(1.0, self.high_water_mark)
        if drawdown >= 0.15 and self.brake_armed:
            self.cooldown, self.brake_armed = 21, False
            self.counts["brakes"] += 1

        weights = {ticker: 0.0 for ticker in self.tickers}
        slow_up = self._slow_trend_up("SPY") and self._slow_trend_up("QQQ")
        fast_weak = self._fast_trend_weak("SPY") or self._fast_trend_weak("QQQ")
        if self.cooldown == 0 and slow_up and not fast_weak:
            weights.update({"QQQ": 0.55, "SPY": 0.30, "GLD": 0.15})
            self.counts["full_growth"] += 1
        elif self.cooldown == 0 and self._slow_trend_up("SPY"):
            # A trend remains positive but has weakened: preserve participation
            # while cutting the concentrated growth exposure in half.
            weights.update({"QQQ": 0.25 if self._slow_trend_up("QQQ") else 0.0,
                            "SPY": 0.30, "GLD": 0.20, "IEF": 0.20, "SHY": 0.05})
            weights["SHY"] += 1.0 - sum(weights.values())
            self.counts["reduced_growth"] += 1
        else:
            self._defensive_allocation(weights)
            self.counts["defense"] += 1
        self._apply_atomic_targets(weights)

    def _defensive_allocation(self, weights):
        remaining = 1.0
        for ticker, weight in [("GLD", 0.30), ("IEF", 0.35), ("TLT", 0.20)]:
            if self._slow_trend_up(ticker):
                weights[ticker] = weight
                remaining -= weight
        weights["SHY"] = max(0.0, remaining)

    def _slow_trend_up(self, ticker):
        p = self.closes[ticker]
        ma200 = sum(p[i] for i in range(1, 201)) / 200.0
        return p[1] > ma200 and p[1] / p[252] - 1.0 > 0

    def _fast_trend_weak(self, ticker):
        p = self.closes[ticker]
        ma100 = sum(p[i] for i in range(1, 101)) / 100.0
        return p[1] < ma100 and p[1] / p[63] - 1.0 < 0

    def _apply_atomic_targets(self, weights):
        value = max(self.portfolio.total_portfolio_value, 1.0)
        targets = [PortfolioTarget(self.symbols[ticker], target) for ticker, target in weights.items()]
        changed = False
        for ticker, target in weights.items():
            symbol = self.symbols[ticker]
            current = self.portfolio[symbol].holdings_value / value
            if abs(target - current) < self.trade_band:
                self.counts["band_skips"] += 1
            else:
                changed = True
        if changed:
            # Lean processes reductions before increases for a target basket,
            # avoiding the V1 insufficient-buying-power error.
            self.set_holdings(targets, liquidate_existing_holdings=True)
            self.counts["orders"] += len(targets)

    def _ready(self, ticker):
        p = self.closes[ticker]
        return p.count >= 254 and p[252] > 0

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
        self.debug("GROWTH DEFENSE V2 COUNTS: {}".format(self.counts))
        self.debug("GROWTH DEFENSE V2 FIXED-PARAMETER OOS RETURNS (%): {}".format(output))
