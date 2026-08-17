from AlgorithmImports import *
from datetime import datetime, timedelta


class QQQTrendPullbackV1(QCAlgorithm):
    """Trend-filtered QQQ pullback strategy with a defensive cash sleeve.

    This has a different source of return from monthly momentum allocation:
    it enters only after a short-term oversold move inside a long-term bull
    trend, then exits on mean reversion or a time/price stop.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.tickers = ["SPY", "QQQ", "GLD", "IEF", "SHY"]
        self.symbols = {ticker: self.add_equity(ticker, Resolution.DAILY).symbol for ticker in self.tickers}
        self.set_benchmark(self.symbols["SPY"])
        self.spy_sma = self.sma(self.symbols["SPY"], 200, Resolution.DAILY)
        self.qqq_sma = self.sma(self.symbols["QQQ"], 200, Resolution.DAILY)
        self.gld_sma = self.sma(self.symbols["GLD"], 200, Resolution.DAILY)
        self.ief_sma = self.sma(self.symbols["IEF"], 200, Resolution.DAILY)
        self.qqq_rsi = self.rsi(self.symbols["QQQ"], 2, MovingAverageType.WILDERS, Resolution.DAILY)
        self.qqq_closes = RollingWindow[float](4)
        self.entry_price, self.entry_time = None, None
        self.trade_band = 0.03
        self.counts = {"signals": 0, "entries": 0, "mean_exits": 0,
                       "time_exits": 0, "stop_exits": 0, "defensive_days": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}
        spy = self.symbols["SPY"]
        self.schedule.on(self.date_rules.month_end(spy), self.time_rules.before_market_close(spy, 1), self._record_oos)
        self.set_warm_up(timedelta(days=300))

    def on_data(self, data):
        if self.symbols["QQQ"] in data.bars:
            self.qqq_closes.add(float(data.bars[self.symbols["QQQ"]].close))
        if self.is_warming_up or not self.spy_sma.is_ready or not self.qqq_sma.is_ready or not self.qqq_rsi.is_ready:
            return
        if self.symbols["QQQ"] not in data.bars or self.symbols["SPY"] not in data.bars:
            return

        qqq_price = float(data.bars[self.symbols["QQQ"]].close)
        spy_price = float(data.bars[self.symbols["SPY"]].close)
        bull_market = spy_price > self.spy_sma.current.value and qqq_price > self.qqq_sma.current.value
        qqq_invested = self.portfolio[self.symbols["QQQ"]].invested

        if qqq_invested:
            held_days = (self.time.date() - self.entry_time.date()).days if self.entry_time else 0
            if self.qqq_rsi.current.value >= 70:
                self._exit_qqq("mean_exits")
            elif self.entry_price and qqq_price <= self.entry_price * 0.94:
                self._exit_qqq("stop_exits")
            elif held_days >= 5:
                self._exit_qqq("time_exits")
            return

        # Require two consecutive down closes as well as RSI(2) oversold. All
        # features are observable at the close; execution occurs next session.
        pullback = (self.qqq_closes.count >= 3 and self.qqq_closes[0] < self.qqq_closes[1] < self.qqq_closes[2])
        if bull_market and pullback and self.qqq_rsi.current.value <= 10:
            self._move_to_qqq(qqq_price)
            self.counts["signals"] += 1
        else:
            self._hold_defensive(spy_price)

    def _move_to_qqq(self, price):
        self.set_holdings([PortfolioTarget(self.symbols["QQQ"], 0.95),
                           PortfolioTarget(self.symbols["GLD"], 0.0),
                           PortfolioTarget(self.symbols["IEF"], 0.0),
                           PortfolioTarget(self.symbols["SHY"], 0.05)],
                          liquidate_existing_holdings=True)
        self.entry_price, self.entry_time = price, self.time
        self.counts["entries"] += 1

    def _exit_qqq(self, reason):
        self.liquidate(self.symbols["QQQ"])
        self.entry_price, self.entry_time = None, None
        self.counts[reason] += 1

    def _hold_defensive(self, spy_price):
        # Outside the equity pullback setup, capital has a defined home. In a
        # bear market it owns only assets above their own 200-day average.
        gl_d = self.securities[self.symbols["GLD"]].price
        ief = self.securities[self.symbols["IEF"]].price
        weights = {"GLD": 0.0, "IEF": 0.0, "SHY": 1.0}
        if self.gld_sma.is_ready and gl_d > self.gld_sma.current.value:
            weights["GLD"], weights["SHY"] = 0.25, 0.75
        if self.ief_sma.is_ready and ief > self.ief_sma.current.value:
            weights["IEF"], weights["SHY"] = 0.35, weights["SHY"] - 0.35
        targets = [PortfolioTarget(self.symbols[ticker], weight) for ticker, weight in weights.items()]
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self.counts["defensive_days"] += 1

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
        self.debug("QQQ PULLBACK COUNTS: {}".format(self.counts))
        self.debug("QQQ PULLBACK FIXED-PARAMETER OOS RETURNS (%): {}".format(result))
