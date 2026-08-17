from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class GoldSilverRelativeValueV1(QCAlgorithm):
    """Market-neutral GLD/SLV relative-value mean-reversion research strategy.

    This tests a structural precious-metals ratio hypothesis instead of a gold
    direction forecast. Short-sale availability, borrow cost and live slippage
    must be checked separately before any live use.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.gld = self.add_equity("GLD", Resolution.DAILY).symbol
        self.slv = self.add_equity("SLV", Resolution.DAILY).symbol
        self.shy = self.add_equity("SHY", Resolution.DAILY).symbol
        self.set_benchmark("SPY")
        self.ratios = RollingWindow[float](61)
        self.gld_closes, self.slv_closes = RollingWindow[float](25), RollingWindow[float](25)
        self.side, self.entry_time = 0, None  # +1: long GLD/short SLV; -1: inverse
        self.counts = {"observations": 0, "entries": 0, "mean_exits": 0,
                       "time_exits": 0, "stop_exits": 0, "cash_days": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}
        self.schedule.on(self.date_rules.month_end(self.gld), self.time_rules.before_market_close(self.gld, 1),
                         self._record_oos)
        self.set_warm_up(timedelta(days=100))

    def on_data(self, data):
        if self.gld not in data.bars or self.slv not in data.bars:
            return
        gld_price, slv_price = float(data.bars[self.gld].close), float(data.bars[self.slv].close)
        if gld_price <= 0 or slv_price <= 0:
            return
        self.ratios.add(math.log(gld_price / slv_price))
        self.gld_closes.add(gld_price)
        self.slv_closes.add(slv_price)
        if self.is_warming_up or self.ratios.count < 61 or self.gld_closes.count < 21:
            return
        self.counts["observations"] += 1
        z_score = self._z_score()

        if self.side != 0:
            held_days = (self.time.date() - self.entry_time.date()).days if self.entry_time else 0
            if abs(z_score) <= 0.50:
                self._exit("mean_exits")
            elif abs(z_score) >= 3.50:
                self._exit("stop_exits")
            elif held_days >= 10:
                self._exit("time_exits")
            return

        if z_score >= 2.0:
            self._enter(-1)  # ratio is rich: short GLD, long SLV
        elif z_score <= -2.0:
            self._enter(1)   # ratio is cheap: long GLD, short SLV
        else:
            self._hold_cash()

    def _enter(self, side):
        gld_vol, slv_vol = self._volatility(self.gld_closes), self._volatility(self.slv_closes)
        if gld_vol <= 0 or slv_vol <= 0:
            return
        # Gross exposure is 90%, with each leg volatility-balanced. Ten percent
        # stays in SHY as a buying-power and financing buffer.
        gld_weight = 0.90 * (1.0 / gld_vol) / (1.0 / gld_vol + 1.0 / slv_vol)
        slv_weight = 0.90 - gld_weight
        self.set_holdings([PortfolioTarget(self.gld, side * gld_weight),
                           PortfolioTarget(self.slv, -side * slv_weight),
                           PortfolioTarget(self.shy, 0.10)], liquidate_existing_holdings=True)
        self.side, self.entry_time = side, self.time
        self.counts["entries"] += 1

    def _exit(self, reason):
        self.set_holdings([PortfolioTarget(self.gld, 0.0), PortfolioTarget(self.slv, 0.0),
                           PortfolioTarget(self.shy, 1.0)], liquidate_existing_holdings=True)
        self.side, self.entry_time = 0, None
        self.counts[reason] += 1

    def _hold_cash(self):
        if not self.portfolio[self.shy].invested:
            self.set_holdings(self.shy, 1.0)
        self.counts["cash_days"] += 1

    def _z_score(self):
        values = [self.ratios[i] for i in range(1, 61)]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return (self.ratios[0] - mean) / max(math.sqrt(variance), 1e-8)

    @staticmethod
    def _volatility(prices):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(20) if prices[i + 1] > 0]
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
        output = {}
        for label, values in self.oos.items():
            if values["start"] and values["end"]:
                output[label] = round(100 * (values["end"] / values["start"] - 1.0), 2)
        self.debug("GOLD SILVER RV COUNTS: {}".format(self.counts))
        self.debug("GOLD SILVER RV FIXED-PARAMETER OOS RETURNS (%): {}".format(output))
