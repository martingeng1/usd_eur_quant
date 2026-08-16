from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class DefensiveAssetAllocationV2(QCAlgorithm):
    """Defensive Asset Allocation V2: trend, breadth and volatility controlled.

    This is a long-only, liquid ETF implementation.  It does not promise a
    target return.  The design deliberately uses a small number of fixed,
    economically motivated rules: diversified momentum horizons, independent
    canaries, a portfolio drawdown brake and a no-trade band.
    """

    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_benchmark("SPY")

        self.canaries = ["VWO", "BND"]
        self.offensive = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF"]
        self.defensive = ["SHY", "IEF", "TLT", "GLD"]
        self.tickers = list(dict.fromkeys(self.canaries + self.offensive + self.defensive))
        self.symbols, self.closes = {}, {}
        for ticker in self.tickers:
            self.symbols[ticker] = self.add_equity(ticker, Resolution.DAILY).symbol
            self.closes[ticker] = RollingWindow[float](260)

        self.equity_history = RollingWindow[float](25)
        self.high_water_mark = 250000.0
        self.cooldown_days = 0
        self.target_volatility = 0.10       # annualized target, never uses leverage
        self.no_trade_band = 0.035          # suppresses uneconomic small rebalances
        self.counts = {"rebalances": 0, "risk_on": 0, "risk_mid": 0,
                       "risk_off": 0, "drawdown_brakes": 0, "orders": 0,
                       "band_skips": 0}
        self.oos_blocks = [
            ("2017-2019", datetime(2017, 1, 1), datetime(2019, 12, 31)),
            ("2020-2021", datetime(2020, 1, 1), datetime(2021, 12, 31)),
            ("2022-2023", datetime(2022, 1, 1), datetime(2023, 12, 31)),
            ("2024-2025", datetime(2024, 1, 1), datetime(2025, 12, 31))]
        self.oos_equity = {label: {"start": None, "end": None} for label, _, _ in self.oos_blocks}

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
                price = float(data.bars[symbol].close)
                if price > 0:
                    self.closes[ticker].add(price)

    def _record_equity(self):
        if self.is_warming_up:
            return
        equity = self.portfolio.total_portfolio_value
        self.high_water_mark = max(self.high_water_mark, equity)
        self.equity_history.add(equity)
        if self.cooldown_days > 0:
            self.cooldown_days -= 1

    def rebalance(self):
        if self.is_warming_up or not self._ready("SPY"):
            return
        self.counts["rebalances"] += 1
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(self.high_water_mark, 1.0)
        if drawdown >= 0.10:
            # A meaningful portfolio loss switches the next month to capital
            # preservation. The 21-day cooldown avoids immediately re-risking.
            self.cooldown_days = max(self.cooldown_days, 21)
            self.counts["drawdown_brakes"] += 1

        weak = sum(1 for ticker in self.canaries if not self._absolute_momentum(ticker))
        spy_trend_ok = self._absolute_momentum("SPY")
        if self.cooldown_days > 0 or weak == 2 or not spy_trend_ok:
            offensive_budget, regime = 0.0, "risk_off"
        elif weak == 1:
            offensive_budget, regime = 0.45, "risk_mid"
        else:
            offensive_budget, regime = 0.90, "risk_on"
        self.counts[regime] += 1

        # Do not lever.  In volatile markets only reduce the risk sleeve.
        offensive_budget *= self._volatility_scalar()
        weights = {ticker: 0.0 for ticker in self.tickers}
        picks = self._rank(self.offensive, 3)
        if offensive_budget > 0 and picks:
            for ticker, weight in self._inverse_vol_weights(picks, 0.45).items():
                weights[ticker] = offensive_budget * weight
        else:
            offensive_budget = 0.0

        defensive_budget = 1.0 - offensive_budget
        defensive_picks = self._rank(self.defensive, 2)
        if defensive_picks:
            # Retain at least 30% SHY during defensive regimes; this is the
            # capital-preservation sleeve rather than a forced duration bet.
            active_defense = defensive_budget * (0.70 if offensive_budget == 0 else 1.0)
            weights["SHY"] += defensive_budget - active_defense
            for ticker, weight in self._inverse_vol_weights(defensive_picks, 0.65).items():
                weights[ticker] += active_defense * weight
        else:
            weights["SHY"] = defensive_budget

        self._apply_targets(weights)

    def _apply_targets(self, weights):
        for ticker, target in weights.items():
            symbol = self.symbols[ticker]
            current = self.portfolio[symbol].holdings_value / max(self.portfolio.total_portfolio_value, 1.0)
            if target == 0.0 and abs(current) >= 0.005:
                self.liquidate(symbol)
                self.counts["orders"] += 1
            elif abs(target - current) < self.no_trade_band:
                self.counts["band_skips"] += 1
            else:
                self.set_holdings(symbol, target)
                self.counts["orders"] += 1

    def _rank(self, tickers, limit):
        rows = []
        for ticker in tickers:
            if not self._ready(ticker) or not self._absolute_momentum(ticker):
                continue
            vol = self._daily_volatility(self.closes[ticker], 20)
            if vol <= 0:
                continue
            rows.append((self._composite_momentum(ticker), ticker, vol))
        return sorted(rows, reverse=True)[:limit]

    def _composite_momentum(self, ticker):
        p = self.closes[ticker]
        # 1/3/6/12-month momentum: broader than a single lookback and all
        # inputs are known at the rebalance date.
        horizons = [(21, 0.10), (63, 0.20), (126, 0.30), (252, 0.40)]
        return sum(weight * (p[1] / p[days] - 1.0) for days, weight in horizons)

    def _absolute_momentum(self, ticker):
        if not self._ready(ticker):
            return False
        p = self.closes[ticker]
        ma200 = sum(p[i] for i in range(1, 201)) / 200.0
        return self._composite_momentum(ticker) > 0 and p[1] > ma200

    def _ready(self, ticker):
        return self.closes[ticker].count >= 254 and self.closes[ticker][252] > 0

    def _volatility_scalar(self):
        if self.equity_history.count < 21:
            return 1.0
        vol = self._daily_volatility(self.equity_history, 20) * math.sqrt(252)
        if vol <= 0:
            return 1.0
        # Keep exposure between 55% and 100%; no leverage and no fragile
        # point-estimate dependence.
        return max(0.55, min(1.0, self.target_volatility / vol))

    @staticmethod
    def _inverse_vol_weights(rows, cap):
        raw = {ticker: 1.0 / vol for _, ticker, vol in rows}
        total = sum(raw.values())
        weights = {ticker: value / total for ticker, value in raw.items()}
        # With 2-3 picks a single pass cap is sufficient; redistribute excess.
        excess = sum(max(0.0, weight - cap) for weight in weights.values())
        weights = {ticker: min(cap, weight) for ticker, weight in weights.items()}
        uncapped = [ticker for ticker, weight in weights.items() if weight < cap - 1e-9]
        if excess > 0 and uncapped:
            denominator = sum(weights[ticker] for ticker in uncapped)
            for ticker in uncapped:
                weights[ticker] += excess * weights[ticker] / denominator
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
        returns = {}
        for label, record in self.oos_equity.items():
            if record["start"] and record["end"]:
                returns[label] = round(100 * (record["end"] / record["start"] - 1.0), 2)
        self.debug("DAA V2 COUNTS: {}".format(self.counts))
        self.debug("DAA V2 FIXED-PARAMETER OOS RETURNS (%): {}".format(returns))
