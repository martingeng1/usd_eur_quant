from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class AdaptiveDefensiveAllocationV3(QCAlgorithm):
    """Long-only adaptive allocation using liquid, free QuantConnect ETFs.

    V3 combines time-series trend, cross-sectional risk-adjusted momentum and
    conservative portfolio-level risk control. It is a research strategy, not
    a promise of return or a recommendation for live trading.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)

        self.canaries = ["VWO", "BND"]
        self.offensive = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF"]
        self.defensive = ["SHY", "IEF", "TLT", "GLD"]
        self.tickers = list(dict.fromkeys(self.canaries + self.offensive + self.defensive))
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)
        self.set_benchmark(self.symbols["SPY"])

        self.equity_history = RollingWindow[float](25)
        self.high_water_mark = 250000.0
        self.cooldown_days = 0
        self.target_volatility = 0.10
        self.no_trade_band = 0.04
        self.counts = {"rebalances": 0, "risk_on": 0, "risk_mid": 0, "risk_low": 0,
                       "brakes": 0, "orders": 0, "band_skips": 0, "no_candidates": 0}
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
        value = self.portfolio.total_portfolio_value
        self.high_water_mark = max(self.high_water_mark, value)
        self.equity_history.add(value)
        self.cooldown_days = max(0, self.cooldown_days - 1)

    def rebalance(self):
        if self.is_warming_up or not self._ready("SPY"):
            return
        self.counts["rebalances"] += 1
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(1.0, self.high_water_mark)
        if drawdown >= 0.10:
            self.cooldown_days = max(self.cooldown_days, 21)
            self.counts["brakes"] += 1

        weak = sum(not self._absolute_trend(ticker) for ticker in self.canaries)
        if self.cooldown_days > 0 or weak == 2:
            risk_budget, regime = 0.15, "risk_low"
        elif weak == 1 or not self._absolute_trend("SPY"):
            risk_budget, regime = 0.55, "risk_mid"
        else:
            risk_budget, regime = 0.90, "risk_on"
        self.counts[regime] += 1
        risk_budget *= self._portfolio_volatility_scalar()

        weights = {ticker: 0.0 for ticker in self.tickers}
        picks = self._rank_risk_adjusted(self.offensive, 3)
        if picks:
            for ticker, weight in self._bounded_inverse_vol(picks, 0.45).items():
                weights[ticker] += risk_budget * weight
        else:
            risk_budget = 0.0
            self.counts["no_candidates"] += 1

        defensive_budget = 1.0 - risk_budget
        defensive_picks = self._rank_risk_adjusted(self.defensive, 2)
        # In low-risk state retain 35% Treasury-bill exposure even when bonds
        # score well. This prevents a concealed all-duration defensive sleeve.
        shy_weight = defensive_budget * (0.35 if regime == "risk_low" else 0.0)
        weights["SHY"] += shy_weight
        deployable = defensive_budget - shy_weight
        if defensive_picks:
            for ticker, weight in self._bounded_inverse_vol(defensive_picks, 0.65).items():
                weights[ticker] += deployable * weight
        else:
            weights["SHY"] += deployable
        self._apply_targets(weights)

    def _rank_risk_adjusted(self, tickers, limit):
        rows = []
        for ticker in tickers:
            if not self._ready(ticker) or not self._absolute_trend(ticker):
                continue
            vol = self._daily_volatility(self.closes[ticker], 20)
            if vol <= 0:
                continue
            # Medium/long horizons are deliberately dominant. A very extended
            # one-month move receives a small penalty, reducing blow-off buys.
            momentum = self._momentum(ticker)
            short_return = self.closes[ticker][1] / self.closes[ticker][21] - 1.0
            overheat_penalty = max(0.0, short_return - 2.0 * vol * math.sqrt(21))
            score = momentum / (vol * math.sqrt(252)) - 0.25 * overheat_penalty
            rows.append((score, ticker, vol))
        return sorted(rows, reverse=True)[:limit]

    def _momentum(self, ticker):
        prices = self.closes[ticker]
        return (0.15 * (prices[1] / prices[63] - 1.0) +
                0.35 * (prices[1] / prices[126] - 1.0) +
                0.50 * (prices[1] / prices[252] - 1.0))

    def _absolute_trend(self, ticker):
        if not self._ready(ticker):
            return False
        prices = self.closes[ticker]
        ma200 = sum(prices[i] for i in range(1, 201)) / 200.0
        return self._momentum(ticker) > 0 and prices[1] > ma200

    def _ready(self, ticker):
        prices = self.closes[ticker]
        return prices.count >= 254 and prices[252] > 0

    def _portfolio_volatility_scalar(self):
        if self.equity_history.count < 21:
            return 1.0
        annual_vol = self._daily_volatility(self.equity_history, 20) * math.sqrt(252)
        if annual_vol <= 0:
            return 1.0
        return max(0.60, min(1.0, self.target_volatility / annual_vol))

    def _apply_targets(self, weights):
        portfolio_value = max(self.portfolio.total_portfolio_value, 1.0)
        for ticker, target in weights.items():
            symbol = self.symbols[ticker]
            current = self.portfolio[symbol].holdings_value / portfolio_value
            if target == 0.0 and abs(current) >= 0.005:
                self.liquidate(symbol)
                self.counts["orders"] += 1
            elif abs(target - current) >= self.no_trade_band:
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
                    result[ticker] = cap
                    budget -= cap
                    capped.append(ticker)
            if not capped:
                for _, ticker, vol in remaining:
                    result[ticker] = budget * (1.0 / vol) / total
                return result
            remaining = [row for row in remaining if row[1] not in capped]
        return result

    @staticmethod
    def _daily_volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(1, lookback + 1)
                   if prices[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        average = sum(returns) / len(returns)
        return math.sqrt(sum((item - average) ** 2 for item in returns) / len(returns))

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
        self.debug("DAA V3 COUNTS: {}".format(self.counts))
        self.debug("DAA V3 FIXED-PARAMETER OOS RETURNS (%): {}".format(results))
