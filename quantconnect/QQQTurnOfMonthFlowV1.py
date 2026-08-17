from AlgorithmImports import *
from datetime import datetime, timedelta


class QQQTurnOfMonthFlowV1(QCAlgorithm):
    """Low-turnover QQQ turn-of-month flow strategy.

    This preserves the calendar-flow hypothesis behind the overnight test while
    reducing execution from roughly 500 trades a year to about 24. It holds
    QQQ only around the month-end/month-start rebalancing window when its
    long-term trend is positive; otherwise capital stays in SHY.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.qqq = self.add_equity("QQQ", Resolution.MINUTE).symbol
        self.shy = self.add_equity("SHY", Resolution.MINUTE).symbol
        self.set_benchmark(self.qqq)
        self.qqq_sma = self.sma(self.qqq, 200, Resolution.DAILY)
        self.in_flow_window = False
        self.counts = {"windows": 0, "trend_entries": 0, "trend_rejected": 0, "exits": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos = {name: {"start": None, "end": None} for name, _, _ in self.oos_blocks}

        # Enter on the fourth-last trading day of each month and exit after
        # the fourth trading day of the following month: about eight sessions
        # of calendar-flow exposure for two portfolio changes per month.
        self.schedule.on(self.date_rules.month_end(self.qqq, 3), self.time_rules.after_market_open(self.qqq, 10),
                         self.enter_flow_window)
        self.schedule.on(self.date_rules.month_start(self.qqq, 4), self.time_rules.after_market_open(self.qqq, 10),
                         self.exit_flow_window)
        self.schedule.on(self.date_rules.month_end(self.qqq), self.time_rules.before_market_close(self.qqq, 1),
                         self.record_oos)
        self.set_warm_up(timedelta(days=300))

    def on_data(self, data):
        pass

    def enter_flow_window(self):
        if self.is_warming_up or not self.qqq_sma.is_ready or self.in_flow_window:
            return
        self.counts["windows"] += 1
        if self.securities[self.qqq].price > self.qqq_sma.current.value:
            self.set_holdings([PortfolioTarget(self.qqq, 0.95), PortfolioTarget(self.shy, 0.05)],
                              liquidate_existing_holdings=True)
            self.counts["trend_entries"] += 1
        else:
            self.set_holdings(self.shy, 1.0)
            self.counts["trend_rejected"] += 1
        self.in_flow_window = True

    def exit_flow_window(self):
        if self.is_warming_up or not self.in_flow_window:
            return
        self.set_holdings([PortfolioTarget(self.qqq, 0.0), PortfolioTarget(self.shy, 1.0)],
                          liquidate_existing_holdings=True)
        self.in_flow_window = False
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
        self.debug("QQQ TURN-OF-MONTH COUNTS: {}".format(self.counts))
        self.debug("QQQ TURN-OF-MONTH FIXED-PARAMETER OOS RETURNS (%): {}".format(result))
