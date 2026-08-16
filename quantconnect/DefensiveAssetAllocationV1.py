from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class DefensiveAssetAllocationV1(QCAlgorithm):
    """Monthly Defensive Asset Allocation with independent canary breadth.

    Canary breadth determines the risk budget before relative momentum chooses
    assets. A positive canary pair permits full offensive exposure, one weak
    canary halves it, and two weak canaries move the portfolio to the
    defensive sleeve. This is an implementation of publicly described DAA
    principles, not a return guarantee.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_benchmark("SPY")

        self.canaries = ["VWO", "BND"]
        self.offensive = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
                          "GLD", "DBC", "TLT", "IEF"]
        self.defensive = ["SHY", "IEF", "TLT", "GLD"]
        self.tickers = list(dict.fromkeys(self.canaries + self.offensive + self.defensive))
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)

        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31)),
        ]
        self.oos_equity = {label: {"start": None, "end": None}
                           for label, _, _ in self.oos_blocks}
        self.counts = {"rebalances": 0, "canary_0": 0, "canary_1": 0,
                       "canary_2": 0, "offensive_selected": 0,
                       "defensive_selected": 0, "tiny_target_skips": 0}
        self.schedule.on(self.date_rules.month_start(), self.time_rules.at(10, 0), self.rebalance)
        self.schedule.on(self.date_rules.month_end(), self.time_rules.at(15, 30), self._record_oos)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for ticker, symbol in self.symbols.items():
            if symbol in data.bars:
                close = float(data.bars[symbol].close)
                if close > 0:
                    self.closes[ticker].add(close)

    def rebalance(self):
        if self.is_warming_up:
            return
        self.counts["rebalances"] += 1
        weak_canaries = sum(1 for ticker in self.canaries if not self._absolute_momentum(ticker))
        self.counts["canary_{}".format(weak_canaries)] += 1
        offensive_budget = 1.0 if weak_canaries == 0 else 0.50 if weak_canaries == 1 else 0.0
        defensive_budget = 1.0 - offensive_budget
        weights = {ticker: 0.0 for ticker in self.tickers}

        if offensive_budget > 0:
            picks = self._rank_eligible(self.offensive, 3)
            if picks:
                for ticker, weight in self._inverse_vol_weights(picks, 0.40).items():
                    weights[ticker] += offensive_budget * weight
                self.counts["offensive_selected"] += len(picks)
            else:
                defensive_budget += offensive_budget

        if defensive_budget > 0:
            # Defensive assets are ranked too; if all are weak, SHY remains
            # the true cash equivalent rather than forcing a bond position.
            picks = self._rank_eligible(self.defensive, 2)
            if not picks:
                weights["SHY"] += defensive_budget
                self.counts["defensive_selected"] += 1
            else:
                for ticker, weight in self._inverse_vol_weights(picks, 0.60).items():
                    weights[ticker] += defensive_budget * weight
                self.counts["defensive_selected"] += len(picks)

        targets = []
        for ticker, weight in weights.items():
            if abs(weight) < 1e-6:
                self.counts["tiny_target_skips"] += 1
                continue
            targets.append(PortfolioTarget(self.symbols[ticker], weight))
        self.set_holdings(targets, liquidate_existing_holdings=True)

    def _rank_eligible(self, tickers, limit):
        rows = []
        for ticker in tickers:
            prices = self.closes[ticker]
            if prices.count < 254 or not self._absolute_momentum(ticker):
                continue
            vol = self._daily_volatility(prices, 20)
            if vol <= 0:
                continue
            momentum = prices[1] / prices[252] - 1.0
            rows.append((momentum, ticker, vol))
        return sorted(rows, reverse=True)[:limit]

    def _absolute_momentum(self, ticker):
        prices = self.closes[ticker]
        if prices.count < 254:
            return False
        momentum = prices[1] / prices[252] - 1.0
        ma200 = sum(prices[index] for index in range(1, 201)) / 200.0
        return momentum > 0 and prices[1] > ma200

    @staticmethod
    def _inverse_vol_weights(rows, cap):
        remaining = list(rows)
        weights, remaining_weight = {}, 1.0
        while remaining:
            inverse_vol_total = sum(1.0 / row[2] for row in remaining)
            capped = []
            for row in remaining:
                share = remaining_weight * (1.0 / row[2]) / inverse_vol_total
                if share > cap:
                    weights[row[1]] = cap
                    remaining_weight -= cap
                    capped.append(row)
            if not capped:
                for row in remaining:
                    weights[row[1]] = remaining_weight * (1.0 / row[2]) / inverse_vol_total
                break
            remaining = [row for row in remaining if row not in capped]
        return weights

    @staticmethod
    def _daily_volatility(prices, lookback):
        returns = [prices[i] / prices[i + 1] - 1.0 for i in range(1, lookback + 1)
                   if prices[i + 1] > 0]
        if len(returns) < lookback * 0.75:
            return 0.0
        mean = sum(returns) / len(returns)
        return math.sqrt(sum((value - mean) ** 2 for value in returns) / len(returns))

    def _record_oos(self):
        if self.is_warming_up:
            return
        for label, start, end in self.oos_blocks:
            if start <= self.time <= end:
                record = self.oos_equity[label]
                if record["start"] is None:
                    record["start"] = self.portfolio.total_portfolio_value
                record["end"] = self.portfolio.total_portfolio_value

    def on_end_of_algorithm(self):
        self._record_oos()
        oos = {}
        for label, record in self.oos_equity.items():
            oos[label] = (round(100 * (record["end"] / record["start"] - 1), 2)
                          if record["start"] is not None and record["end"] is not None else None)
        self.debug("DAA COUNTS: {}".format(self.counts))
        self.debug("FIXED-PARAMETER OOS RETURNS (%): {}".format(oos))
