from AlgorithmImports import *
from datetime import timedelta
import math


class InstitutionalMultiHorizonTrendV1(QCAlgorithm):
    """Institutional-style managed-futures proxy using liquid ETF sleeves.

    This is not proprietary institutional code. It implements the public
    building blocks: diversified time-series trend horizons, volatility
    scaling, economic-sleeve diversification, and portfolio risk control.
    """
    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)

        sleeves = {
            "metals": ["GLD", "SLV"],
            "energy": ["USO", "XLE"],
            "rates": ["TLT", "IEF"],
            "equity_us": ["SPY", "XLF", "XLV"],
            "equity_global": ["EEM", "EFA"],
            "currency": ["UUP", "FXE"],
            "real_assets": ["VNQ"],
        }
        self.symbols = {}
        self.sleeve_of = {}
        self.states = {}
        for sleeve, tickers in sleeves.items():
            for ticker in tickers:
                symbol = self.add_equity(ticker, Resolution.DAILY).symbol
                self.symbols[ticker] = symbol
                self.sleeve_of[symbol] = sleeve
                self.states[symbol] = {"closes": RollingWindow[float](260),
                                       "returns": RollingWindow[float](63)}

        self.peak_equity = self.portfolio.total_portfolio_value
        self.counts = {"rebalance_days": 0, "eligible": 0, "selected": 0,
                       "risk_off": 0, "drawdown_scale_down": 0}
        # Monday afternoon uses Friday's completed daily bars and avoids
        # constant end-of-day order submission.
        self.schedule.on(self.date_rules.week_start(self.symbols["SPY"]),
                         self.time_rules.at(14, 30), self.rebalance)
        self.set_warm_up(timedelta(days=400))

    def on_data(self, data):
        for symbol, state in self.states.items():
            if symbol not in data.bars:
                continue
            price = float(data.bars[symbol].close)
            if price <= 0:
                continue
            closes = state["closes"]
            if closes.count > 0 and closes[0] > 0:
                state["returns"].add(price / closes[0] - 1.0)
            closes.add(price)

    def rebalance(self):
        if self.is_warming_up:
            return
        self.counts["rebalance_days"] += 1
        self.peak_equity = max(self.peak_equity, self.portfolio.total_portfolio_value)
        drawdown = 1.0 - self.portfolio.total_portfolio_value / max(self.peak_equity, 1.0)
        gross_target = 0.80 if drawdown < 0.10 else 0.40 if drawdown < 0.15 else 0.0
        if gross_target < 0.80:
            self.counts["drawdown_scale_down"] += 1

        candidates = []
        for symbol, state in self.states.items():
            closes, returns = state["closes"], state["returns"]
            if closes.count < 253 or returns.count < 60:
                continue
            # Equal-weighted 1, 3, and 12 month time-series momentum.
            r1 = closes[0] / closes[20] - 1.0
            r3 = closes[0] / closes[63] - 1.0
            r12 = closes[0] / closes[252] - 1.0
            signed = [1 if r > 0 else -1 for r in (r1, r3, r12)]
            signal = sum(signed) / 3.0
            if signal == 0:
                continue
            annual_vol = math.sqrt(sum(returns[i] * returns[i] for i in range(60)) / 60.0) * math.sqrt(252)
            if annual_vol < 0.05 or annual_vol > 0.80:
                continue
            confidence = abs(signal)
            score = confidence / annual_vol
            candidates.append((score, symbol, 1 if signal > 0 else -1, annual_vol))

        self.counts["eligible"] += len(candidates)
        sleeve_best = {}
        for item in sorted(candidates, key=lambda x: x[0], reverse=True):
            sleeve = self.sleeve_of[item[1]]
            if sleeve not in sleeve_best:
                sleeve_best[sleeve] = item
        selected = list(sleeve_best.values())
        if not selected or gross_target == 0:
            self.counts["risk_off"] += 1
            for symbol in self.states:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol, tag="trend risk-off")
            return

        inverse_vol_sum = sum(1.0 / item[3] for item in selected)
        targets = {}
        for _, symbol, direction, annual_vol in selected:
            # Volatility parity with a hard per-sleeve cap limits correlation
            # spikes from becoming a single-market bet.
            weight = min(0.20, gross_target * (1.0 / annual_vol) / inverse_vol_sum)
            targets[symbol] = direction * weight
        for symbol in self.states:
            self.set_holdings(symbol, targets.get(symbol, 0.0), tag="multi-horizon trend")
        self.counts["selected"] += len(selected)

    def on_end_of_algorithm(self):
        self.debug("INSTITUTIONAL TREND COUNTS: {}".format(self.counts))
