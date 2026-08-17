from AlgorithmImports import *
from datetime import datetime, timedelta


class QQQOvernightPremiumV1(QCAlgorithm):
    """Trend-filtered QQQ overnight-risk-premium strategy.

    The hypothesis is that a meaningful portion of equity risk compensation
    arrives outside regular cash-market hours. The strategy holds QQQ only
    from shortly before close to shortly after the following open, and uses
    SHY while flat. Minute data is required for realistic execution timing.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.qqq = self.add_equity("QQQ", Resolution.MINUTE).symbol
        self.shy = self.add_equity("SHY", Resolution.MINUTE).symbol
        self.set_benchmark(self.qqq)
        self.qqq_sma = self.sma(self.qqq, 200, Resolution.DAILY)
        self.in_overnight_position = False
        self.counts = {"eligible_nights": 0, "entries": 0, "exits": 0,
                       "trend_rejected": 0, "friday_skipped": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}

        self.schedule.on(self.date_rules.every_day(self.qqq), self.time_rules.after_market_open(self.qqq, 10),
                         self.exit_overnight)
        self.schedule.on(self.date_rules.every_day(self.qqq), self.time_rules.before_market_close(self.qqq, 10),
                         self.enter_overnight)
        self.schedule.on(self.date_rules.month_end(self.qqq), self.time_rules.before_market_close(self.qqq, 1),
                         self.record_oos)
        self.set_warm_up(timedelta(days=300))

    def on_data(self, data):
        pass

    def enter_overnight(self):
        if self.is_warming_up or not self.qqq_sma.is_ready or self.in_overnight_position:
            return
        # Friday close is excluded: its holding period contains unpriced
        # weekend event risk rather than the regular one-night risk premium.
        if self.time.weekday() == 4:
            self.counts["friday_skipped"] += 1
            return
        price = self.securities[self.qqq].price
        if price <= self.qqq_sma.current.value:
            self.counts["trend_rejected"] += 1
            return
        # Five-percent short-bond buffer avoids full-cash sizing errors at the
        # close and makes the order more conservative under gap fills.
        self.set_holdings([PortfolioTarget(self.qqq, 0.95), PortfolioTarget(self.shy, 0.05)],
                          liquidate_existing_holdings=True)
        self.in_overnight_position = True
        self.counts["eligible_nights"] += 1
        self.counts["entries"] += 1

    def exit_overnight(self):
        if self.is_warming_up or not self.in_overnight_position:
            return
        self.set_holdings([PortfolioTarget(self.qqq, 0.0), PortfolioTarget(self.shy, 1.0)],
                          liquidate_existing_holdings=True)
        self.in_overnight_position = False
        self.counts["exits"] += 1

    def record_oos(self):
        if self.is_warming_up:
            return
        for label, start, end in self.oos_blocks:
            if start <= self.time <= end:
                if self.oos[label]["start"] is None:
                    self.oos[label]["start"] = self.portfolio.total_portfolio_value
                self.oos[label]["end"] = self.portfolio.total_portfolio_value

    def on_end_of_algorithm(self):
        self.record_oos()
        result = {}
        for label, values in self.oos.items():
            if values["start"] and values["end"]:
                result[label] = round(100 * (values["end"] / values["start"] - 1.0), 2)
        self.debug("QQQ OVERNIGHT COUNTS: {}".format(self.counts))
        self.debug("QQQ OVERNIGHT FIXED-PARAMETER OOS RETURNS (%): {}".format(result))
