from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class DiversifiedTrendPortfolioV1(QCAlgorithm):
    """Diversified, unlevered time-series trend portfolio.

    This intentionally changes route from long-only asset allocation. Each
    market carries its own long/short trend signal and inverse-vol risk budget.
    The portfolio has no fitted parameter search and is for research only.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.tickers = ["SPY", "TLT", "GLD", "DBC", "UUP", "EFA", "VNQ"]
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)
        self.set_benchmark(self.symbols["SPY"])

        self.equity_history = RollingWindow[float](25)
        self.high_water_mark = 250000.0
        self.brake_days = 0
        self.no_trade_band = 0.025
        self.counts = {"rebalances": 0, "long_signals": 0, "short_signals": 0,
                       "flat_signals": 0, "brakes": 0, "orders": 0, "band_skips": 0}
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
        self.brake_days = max(0, self.brake_days - 1)

    def rebalance(self):
        if self.is_warming_up or not all(self._ready(ticker) for ticker in self.tickers):
            return
        self.counts["rebalances"] += 1
        if 1.0 - self.portfolio.total_portfolio_value / max(self.high_water_mark, 1.0) >= 0.10:
            self.brake_days = max(self.brake_days, 21)
            self.counts["brakes"] += 1

        rows = []
        for ticker in self.tickers:
            direction, strength = self._trend_signal(ticker)
            vol = self._daily_volatility(self.closes[ticker], 20)
            if direction == 0 or vol <= 0:
                self.counts["flat_signals"] += 1
                continue
            self.counts["long_signals" if direction > 0 else "short_signals"] += 1
            rows.append((ticker, direction, strength, vol))

        # Gross exposure is at most 100% (no leverage). A portfolio-volatility
        # scalar and drawdown brake can only reduce it.
        gross_budget = 1.0 * self._volatility_scalar()
        if self.brake_days > 0:
            gross_budget *= 0.50
        raw = {ticker: strength / vol for ticker, _, strength, vol in rows}
        total = sum(raw.values())
        targets = {ticker: 0.0 for ticker in self.tickers}
        if total > 0:
            for ticker, direction, _, _ in rows:
                targets[ticker] = direction * min(0.25, gross_budget * raw[ticker] / total)
            # Reallocate any cap-constrained residual to cash rather than
            # concentration; this is intentional risk control.
        self._apply_targets(targets)

    def _trend_signal(self, ticker):
        prices = self.closes[ticker]
        ret_3m = prices[1] / prices[63] - 1.0
        ret_12m = prices[1] / prices[252] - 1.0
        ma200 = sum(prices[i] for i in range(1, 201)) / 200.0
        if ret_3m > 0 and ret_12m > 0 and prices[1] > ma200:
            return 1, abs(0.4 * ret_3m + 0.6 * ret_12m)
        if ret_3m < 0 and ret_12m < 0 and prices[1] < ma200:
            return -1, abs(0.4 * ret_3m + 0.6 * ret_12m)
        return 0, 0.0

    def _volatility_scalar(self):
        if self.equity_history.count < 21:
            return 1.0
        annual_vol = self._daily_volatility(self.equity_history, 20) * math.sqrt(252)
        if annual_vol <= 0:
            return 1.0
        return max(0.55, min(1.0, 0.10 / annual_vol))

    def _apply_targets(self, targets):
        value = max(self.portfolio.total_portfolio_value, 1.0)
        for ticker, target in targets.items():
            symbol = self.symbols[ticker]
            current = self.portfolio[symbol].holdings_value / value
            if target == 0 and abs(current) >= 0.005:
                self.liquidate(symbol)
                self.counts["orders"] += 1
            elif abs(target - current) >= self.no_trade_band:
                self.set_holdings(symbol, target)
                self.counts["orders"] += 1
            else:
                self.counts["band_skips"] += 1

    def _ready(self, ticker):
        prices = self.closes[ticker]
        return prices.count >= 254 and prices[252] > 0

    @staticmethod
    def _daily_volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(1, lookback + 1)
                   if prices[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((item - mean) ** 2 for item in returns) / len(returns))

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
        result = {}
        for label, values in self.oos.items():
            if values["start"] and values["end"]:
                result[label] = round(100 * (values["end"] / values["start"] - 1.0), 2)
        self.debug("DIVERSIFIED TREND COUNTS: {}".format(self.counts))
        self.debug("DIVERSIFIED TREND FIXED-PARAMETER OOS RETURNS (%): {}".format(result))
