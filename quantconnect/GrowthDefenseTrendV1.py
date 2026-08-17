from AlgorithmImports import *
from datetime import datetime, timedelta


class GrowthDefenseTrendV1(QCAlgorithm):
    """Concentrated growth allocation with an independent trend-defense sleeve.

    This is intentionally a return-seeking alternative to broad rotation:
    QQQ/SPY lead during persistent equity trends; GLD, IEF and SHY take over
    when the equity trend breaks. It is not a return or drawdown guarantee.
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
        self.cooldown = 0
        self.trade_band = 0.04
        self.counts = {"rebalances": 0, "full_growth": 0, "partial_growth": 0,
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
        value = self.portfolio.total_portfolio_value
        self.high_water_mark = max(self.high_water_mark, value)
        self.cooldown = max(0, self.cooldown - 1)

    def rebalance(self):
        if self.is_warming_up or not all(self._ready(ticker) for ticker in self.tickers):
            return
        self.counts["rebalances"] += 1
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(1.0, self.high_water_mark)
        if drawdown >= 0.15:
            self.cooldown = max(self.cooldown, 21)
            self.counts["brakes"] += 1

        weights = {ticker: 0.0 for ticker in self.tickers}
        spy_up, qqq_up = self._trend_up("SPY"), self._trend_up("QQQ")
        if self.cooldown == 0 and spy_up and qqq_up:
            # Growth core: 85% equities while the broad and growth trends agree.
            weights.update({"QQQ": 0.55, "SPY": 0.30, "GLD": 0.15})
            self.counts["full_growth"] += 1
        elif self.cooldown == 0 and spy_up:
            # Avoid forcing QQQ ownership when its own trend is broken.
            weights.update({"SPY": 0.55, "GLD": 0.20, "IEF": 0.20, "SHY": 0.05})
            self.counts["partial_growth"] += 1
        else:
            self._defensive_allocation(weights)
            self.counts["defense"] += 1
        self._apply_targets(weights)

    def _defensive_allocation(self, weights):
        # Allocate defensive risk only to assets in their own long-term trend;
        # leftover capital remains in SHY rather than forecasting bond returns.
        remaining = 1.0
        for ticker, candidate_weight in [("GLD", 0.30), ("IEF", 0.35), ("TLT", 0.20)]:
            if self._trend_up(ticker):
                weights[ticker] = candidate_weight
                remaining -= candidate_weight
        weights["SHY"] = max(0.0, remaining)

    def _trend_up(self, ticker):
        prices = self.closes[ticker]
        ma200 = sum(prices[i] for i in range(1, 201)) / 200.0
        return prices[1] > ma200 and prices[1] / prices[252] - 1.0 > 0

    def _apply_targets(self, weights):
        value = max(self.portfolio.total_portfolio_value, 1.0)
        for ticker, target in weights.items():
            symbol = self.symbols[ticker]
            current = self.portfolio[symbol].holdings_value / value
            if target == 0.0 and abs(current) >= 0.005:
                self.liquidate(symbol)
                self.counts["orders"] += 1
            elif abs(target - current) >= self.trade_band:
                self.set_holdings(symbol, target)
                self.counts["orders"] += 1
            else:
                self.counts["band_skips"] += 1

    def _ready(self, ticker):
        prices = self.closes[ticker]
        return prices.count >= 254 and prices[252] > 0

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
        results = {}
        for label, values in self.oos.items():
            if values["start"] and values["end"]:
                results[label] = round(100 * (values["end"] / values["start"] - 1.0), 2)
        self.debug("GROWTH DEFENSE COUNTS: {}".format(self.counts))
        self.debug("GROWTH DEFENSE FIXED-PARAMETER OOS RETURNS (%): {}".format(results))
