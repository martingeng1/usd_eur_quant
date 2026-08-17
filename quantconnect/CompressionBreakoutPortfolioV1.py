from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class CompressionBreakoutPortfolioV1(QCAlgorithm):
    """Volatility-compression breakout portfolio.

    A non-momentum-rotation strategy: it waits for volatility in QQQ, GLD or
    TLT to contract materially, then enters only after a 20-day upside break
    inside that asset's 200-day trend. Idle capital remains in SHY.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.risk_assets = ["QQQ", "GLD", "TLT"]
        self.tickers = self.risk_assets + ["SHY", "SPY"]
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)
        self.set_benchmark(self.symbols["SPY"])

        self.position_ticker, self.entry_price, self.entry_time = None, None, None
        self.counts = {"signals": 0, "entries": 0, "trend_rejects": 0,
                       "compression_rejects": 0, "mean_exits": 0,
                       "time_exits": 0, "stop_exits": 0, "cash_days": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}
        spy = self.symbols["SPY"]
        self.schedule.on(self.date_rules.month_end(spy), self.time_rules.before_market_close(spy, 1), self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for ticker, symbol in self.symbols.items():
            if symbol in data.bars:
                close = float(data.bars[symbol].close)
                if close > 0:
                    self.closes[ticker].add(close)
        if self.is_warming_up or not all(self._ready(ticker) for ticker in self.risk_assets):
            return

        if self.position_ticker:
            self._manage_position()
        else:
            self._scan_for_breakout()

    def _scan_for_breakout(self):
        candidates = []
        for ticker in self.risk_assets:
            prices = self.closes[ticker]
            price = prices[0]
            ma200 = self._mean(prices, 200)
            if price <= ma200:
                self.counts["trend_rejects"] += 1
                continue
            vol10, vol40 = self._volatility(prices, 10), self._volatility(prices, 40)
            if vol10 <= 0 or vol40 <= 0 or vol10 > 0.70 * vol40:
                self.counts["compression_rejects"] += 1
                continue
            prior_high = max(prices[i] for i in range(1, 21))
            if price > prior_high:
                # Breakout quality: return above the range relative to recent
                # volatility. This ranks concurrent signals without forecasting.
                candidates.append(((price / prior_high - 1.0) / vol10, ticker))
        if not candidates:
            self._hold_cash()
            return
        _, ticker = max(candidates)
        self.set_holdings([PortfolioTarget(self.symbols[ticker], 0.95),
                           PortfolioTarget(self.symbols["SHY"], 0.05)],
                          liquidate_existing_holdings=True)
        self.position_ticker = ticker
        self.entry_price, self.entry_time = self.closes[ticker][0], self.time
        self.counts["signals"] += 1
        self.counts["entries"] += 1

    def _manage_position(self):
        ticker = self.position_ticker
        prices, price = self.closes[ticker], self.closes[ticker][0]
        held_days = (self.time.date() - self.entry_time.date()).days if self.entry_time else 0
        if price < self._mean(prices, 10):
            self._exit("mean_exits")
        elif self.entry_price and price <= 0.93 * self.entry_price:
            self._exit("stop_exits")
        elif held_days >= 20:
            self._exit("time_exits")

    def _exit(self, reason):
        self.liquidate(self.symbols[self.position_ticker])
        self.position_ticker, self.entry_price, self.entry_time = None, None, None
        self.counts[reason] += 1

    def _hold_cash(self):
        shy = self.symbols["SHY"]
        if self.portfolio[shy].holdings_value / max(self.portfolio.total_portfolio_value, 1.0) < 0.95:
            self.set_holdings([PortfolioTarget(shy, 1.0)], liquidate_existing_holdings=True)
        self.counts["cash_days"] += 1

    @staticmethod
    def _mean(prices, lookback):
        return sum(prices[i] for i in range(lookback)) / float(lookback)

    @staticmethod
    def _volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(lookback) if prices[i + 1] > 0]
        if len(returns) < lookback * 0.8:
            return 0.0
        average = sum(returns) / len(returns)
        return math.sqrt(sum((item - average) ** 2 for item in returns) / len(returns))

    def _ready(self, ticker):
        return self.closes[ticker].count >= 254

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
        self.debug("COMPRESSION BREAKOUT COUNTS: {}".format(self.counts))
        self.debug("COMPRESSION BREAKOUT FIXED-PARAMETER OOS RETURNS (%): {}".format(output))
