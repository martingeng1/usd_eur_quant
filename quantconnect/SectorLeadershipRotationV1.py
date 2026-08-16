from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class SectorLeadershipRotationV1(QCAlgorithm):
    """Long-only sector-leadership rotation with a separate defensive sleeve.

    The return source is cross-sectional leadership among liquid US sectors,
    not a forecast of the next daily SPY move. Parameters are fixed before the
    test and the strategy prints independent out-of-sample blocks at the end.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.sectors = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
        self.defensive = ["SHY", "IEF", "TLT", "GLD"]
        self.tickers = ["SPY"] + self.sectors + self.defensive
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)
        self.set_benchmark(self.symbols["SPY"])

        self.equity_history = RollingWindow[float](25)
        self.high_water_mark = 250000.0
        self.cooldown = 0
        self.trade_band = 0.04
        self.counts = {"rebalances": 0, "risk_on": 0, "risk_off": 0,
                       "brakes": 0, "orders": 0, "band_skips": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}

        spy = self.symbols["SPY"]
        self.schedule.on(self.date_rules.every_day(spy), self.time_rules.before_market_close(spy, 1),
                         self._record_equity)
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

    def _record_equity(self):
        if self.is_warming_up:
            return
        equity = self.portfolio.total_portfolio_value
        self.high_water_mark = max(self.high_water_mark, equity)
        self.equity_history.add(equity)
        self.cooldown = max(0, self.cooldown - 1)

    def rebalance(self):
        if self.is_warming_up or not self._ready("SPY"):
            return
        self.counts["rebalances"] += 1
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(self.high_water_mark, 1.0)
        if drawdown >= 0.12:
            self.cooldown = max(self.cooldown, 21)
            self.counts["brakes"] += 1

        risk_on = self.cooldown == 0 and self._market_trend_is_positive()
        weights = {ticker: 0.0 for ticker in self.tickers}
        if risk_on:
            self.counts["risk_on"] += 1
            picks = self._rank(self.sectors, 3)
            if picks:
                # Retain a small cash/short-bond buffer. The rest is allocated
                # to distinct sector leaders by inverse volatility.
                weights["SHY"] = 0.05
                for ticker, weight in self._bounded_inverse_vol(picks, 0.40).items():
                    weights[ticker] = 0.95 * weight
            else:
                self._defensive_weights(weights)
        else:
            self.counts["risk_off"] += 1
            self._defensive_weights(weights)
        self._apply_targets(weights)

    def _market_trend_is_positive(self):
        p = self.closes["SPY"]
        ma200 = sum(p[i] for i in range(1, 201)) / 200.0
        return p[1] > ma200 and p[1] / p[126] - 1.0 > 0

    def _defensive_weights(self, weights):
        picks = self._rank(self.defensive, 2)
        if not picks:
            weights["SHY"] = 1.0
            return
        # At least 35% remains in short bonds during defensive periods.
        weights["SHY"] = 0.35
        for ticker, weight in self._bounded_inverse_vol(picks, 0.65).items():
            weights[ticker] += 0.65 * weight

    def _rank(self, tickers, limit):
        rows = []
        for ticker in tickers:
            if not self._ready(ticker):
                continue
            p = self.closes[ticker]
            ma200 = sum(p[i] for i in range(1, 201)) / 200.0
            ret3, ret6, ret12 = p[1] / p[63] - 1.0, p[1] / p[126] - 1.0, p[1] / p[252] - 1.0
            score = 0.20 * ret3 + 0.30 * ret6 + 0.50 * ret12
            vol = self._daily_volatility(p, 20)
            if score > 0 and p[1] > ma200 and vol > 0:
                rows.append((score / (vol * math.sqrt(252)), ticker, vol))
        return sorted(rows, reverse=True)[:limit]

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

    @staticmethod
    def _bounded_inverse_vol(rows, cap):
        remaining, result, budget = list(rows), {}, 1.0
        while remaining:
            total = sum(1.0 / row[2] for row in remaining)
            capped = []
            for _, ticker, vol in remaining:
                weight = budget * (1.0 / vol) / total
                if weight > cap:
                    result[ticker], budget = cap, budget - cap
                    capped.append(ticker)
            if not capped:
                for _, ticker, vol in remaining:
                    result[ticker] = budget * (1.0 / vol) / total
                return result
            remaining = [row for row in remaining if row[1] not in capped]
        return result

    def _ready(self, ticker):
        p = self.closes[ticker]
        return p.count >= 254 and p[252] > 0

    @staticmethod
    def _daily_volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(1, lookback + 1)
                   if prices[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((x - mean) ** 2 for x in returns) / len(returns))

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
        self.debug("SECTOR ROTATION COUNTS: {}".format(self.counts))
        self.debug("SECTOR ROTATION FIXED-PARAMETER OOS RETURNS (%): {}".format(results))
