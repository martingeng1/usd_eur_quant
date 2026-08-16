from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class GlobalDualMomentumV1(QCAlgorithm):
    """Monthly global dual momentum with a Treasury/cash fallback.

    This is a transparent, low-turnover allocation model. It ranks assets by
    12-to-1-month momentum, requires a 200-day trend filter, holds at most
    three risk assets, and sends unused risk capital to SHY. GLD participates
    as one candidate; the model does not assume gold must always be held.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_benchmark("SPY")

        self.risk_tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
                             "GLD", "DBC", "TLT", "IEF"]
        self.defensive_ticker = "SHY"
        self.symbols = {}
        self.closes = {}
        for ticker in self.risk_tickers + [self.defensive_ticker]:
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            self.symbols[ticker] = symbol
            self.closes[ticker] = RollingWindow[float](260)

        self.high_water = self.portfolio.total_portfolio_value
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31)),
        ]
        self.oos_equity = {label: {"start": None, "end": None}
                           for label, _, _ in self.oos_blocks}
        self.counts = {"rebalance_days": 0, "eligible": 0, "selected": 0,
                       "risk_reduced": 0, "cash_months": 0, "tiny_target_skips": 0}
        # The first market day of a new month only uses already completed
        # daily bars from the previous month, avoiding same-bar look-ahead.
        self.schedule.on(self.date_rules.month_start(), self.time_rules.at(10, 0), self.rebalance)
        self.schedule.on(self.date_rules.month_end(), self.time_rules.at(15, 30), self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for ticker, symbol in self.symbols.items():
            if symbol in data.bars:
                price = float(data.bars[symbol].close)
                if price > 0:
                    self.closes[ticker].add(price)

    def rebalance(self):
        if self.is_warming_up:
            return
        self.counts["rebalance_days"] += 1
        self.high_water = max(self.high_water, self.portfolio.total_portfolio_value)
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(self.high_water, 1.0)
        risk_scale = 1.0 if drawdown < 0.10 else 0.50 if drawdown < 0.15 else 0.25
        self.counts["risk_reduced"] += int(risk_scale < 1.0)

        candidates = []
        for ticker in self.risk_tickers:
            prices = self.closes[ticker]
            if prices.count < 254:
                continue
            # Offset all observations by one completed bar.  Momentum is
            # trailing months 12 through 1, and the trend filter is 200 days.
            momentum = prices[1] / prices[252] - 1.0
            ma200 = sum(prices[i] for i in range(1, 201)) / 200.0
            vol = self._daily_volatility(prices, 20)
            if momentum > 0 and prices[1] > ma200 and vol > 0:
                candidates.append((momentum, ticker, vol))
                self.counts["eligible"] += 1

        selected = sorted(candidates, reverse=True)[:3]
        weights = {ticker: 0.0 for ticker in self.risk_tickers + [self.defensive_ticker]}
        if not selected:
            weights[self.defensive_ticker] = 1.0
            self.counts["cash_months"] += 1
        else:
            inverse_vol = [1.0 / row[2] for row in selected]
            total_inverse_vol = sum(inverse_vol)
            raw = [value / total_inverse_vol for value in inverse_vol]
            # Limit a single asset, then preserve a fully invested but
            # defensive allocation by placing unused exposure in SHY.
            capped = [min(0.45, value) for value in raw]
            risk_weight = sum(capped)
            expected_daily_vol = math.sqrt(sum((weight * row[2]) ** 2
                                               for weight, row in zip(capped, selected)))
            target_daily_vol = 0.11 / math.sqrt(252.0)
            vol_scale = min(1.0, target_daily_vol / expected_daily_vol) if expected_daily_vol > 0 else 0.0
            for weight, (_, ticker, _) in zip(capped, selected):
                weights[ticker] = weight * vol_scale * risk_scale
            weights[self.defensive_ticker] = max(0.0, 1.0 - sum(weights.values()))
            self.counts["selected"] += len(selected)

        # Floating-point arithmetic can leave a 1e-11 residual allocation.
        # LEAN correctly rejects it below its minimum target threshold; omit
        # it ourselves so the terminal stays clean and every submitted target
        # represents a meaningful allocation.
        targets = []
        for ticker, weight in weights.items():
            if abs(weight) < 1e-6:
                self.counts["tiny_target_skips"] += 1
                continue
            targets.append(PortfolioTarget(self.symbols[ticker], weight))
        self.set_holdings(targets, liquidate_existing_holdings=True)

    def _record_oos(self):
        if self.is_warming_up:
            return
        for label, start, end in self.oos_blocks:
            if start <= self.time <= end:
                record = self.oos_equity[label]
                if record["start"] is None:
                    record["start"] = self.portfolio.total_portfolio_value
                record["end"] = self.portfolio.total_portfolio_value

    @staticmethod
    def _daily_volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(1, lookback + 1)
                   if prices[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))

    def on_end_of_algorithm(self):
        self._record_oos()
        oos = {}
        for label, record in self.oos_equity.items():
            oos[label] = (round(100 * (record["end"] / record["start"] - 1), 2)
                          if record["start"] is not None and record["end"] is not None else None)
        self.debug("GLOBAL DUAL MOMENTUM COUNTS: {}".format(self.counts))
        self.debug("FIXED-PARAMETER OOS RETURNS (%): {}".format(oos))
