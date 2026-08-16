from AlgorithmImports import *
from datetime import timedelta
import math


class MultiMarketAIRegimePortfolioV1(QCAlgorithm):
    """Diversified ETF portfolio with per-market online AI risk filters.

    The model is trained only after each 24-hour outcome is observable. It
    uses AI as a risk filter and ranks established trends weekly, limiting
    gross exposure to 50% across at most two independent positions.
    """
    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)

        tickers = ["GLD", "SLV", "USO", "TLT", "SPY", "UUP"]
        self.symbols = {}
        self.states = {}
        for ticker in tickers:
            symbol = self.add_equity(ticker, Resolution.HOUR).symbol
            self.symbols[ticker] = symbol
            self.states[symbol] = {
                # About 20/60 regular trading days at hourly resolution.
                # Direction comes from this slower, more robust trend leg.
                "ema_fast": self.ema(symbol, 140, Resolution.HOUR),
                "ema_slow": self.ema(symbol, 420, Resolution.HOUR),
                "rsi": self.rsi(symbol, 14, MovingAverageType.WILDERS, Resolution.HOUR),
                "atr": self.atr(symbol, 20, MovingAverageType.WILDERS, Resolution.HOUR),
                "closes": RollingWindow[float](80),
                "volumes": RollingWindow[float](30),
                "pending": [], "weights": [0.0] * 6, "bias": 0.0,
                "labels": 0, "feature": None, "price": 0.0,
            }

        self.learning_rate = 0.020
        self.l2 = 0.0005
        self.label_horizon = 24
        self.counts = {"labels": 0, "rebalance_days": 0, "candidates": 0,
                       "selected": 0, "risk_off_days": 0}
        # Rebalance once weekly, avoiding daily prediction-driven turnover.
        self.schedule.on(self.date_rules.week_start(self.symbols["SPY"]),
                         self.time_rules.at(14, 30), self.rebalance)
        # Preserve native hourly data for hourly indicators and model labels.
        self.set_warm_up(timedelta(days=180))

    def on_data(self, data):
        for symbol, state in self.states.items():
            if symbol not in data.bars:
                continue
            bar = data.bars[symbol]
            price = float(bar.close)
            if price <= 0:
                continue
            self._train_matured(state, price)
            if self._ready(state):
                feature = self._features(state, bar)
                if feature is not None:
                    state["feature"] = feature
                    state["price"] = price
                    state["pending"].append({"price": price, "feature": feature, "age": 0})
            state["closes"].add(price)
            state["volumes"].add(float(bar.volume))

    def rebalance(self):
        if self.is_warming_up:
            return
        self.counts["rebalance_days"] += 1
        ranked = []
        for symbol, state in self.states.items():
            if state["labels"] < 1000 or state["feature"] is None:
                continue
            probability = self._predict(state, state["feature"])
            # AI is a risk filter, not the directional engine. Long-only ETF
            # exposure avoids the poor short-side behavior of the V1 model.
            if probability < 0.55:
                continue
            direction = 1
            atr = float(state["atr"].current.value)
            trend = float(state["ema_fast"].current.value - state["ema_slow"].current.value)
            if atr <= 0 or trend <= 0.50 * atr:
                continue
            atr_pct = atr / max(state["price"], 1e-8)
            if atr_pct < 0.002 or atr_pct > 0.040:
                continue
            confidence = abs(probability - 0.5) * 2.0
            strength = min(abs(trend) / atr, 3.0) / 3.0
            # Inverse volatility makes one high-volatility ETF unable to
            # dominate the portfolio.
            score = confidence * strength / atr_pct
            ranked.append((score, symbol, direction, atr_pct))

        self.counts["candidates"] += len(ranked)
        ranked.sort(key=lambda x: x[0], reverse=True)
        chosen = ranked[:2]
        if not chosen:
            self.counts["risk_off_days"] += 1
            for symbol in self.states:
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol, tag="AI risk-off")
            return

        total_inverse_vol = sum(1.0 / row[3] for row in chosen)
        targets = {}
        for _, symbol, direction, atr_pct in chosen:
            weight = min(0.30, 0.50 * (1.0 / atr_pct) / total_inverse_vol)
            targets[symbol] = direction * weight
        for symbol in self.states:
            self.set_holdings(symbol, targets.get(symbol, 0.0), tag="AI regime rebalance")
        self.counts["selected"] += len(chosen)

    def _ready(self, state):
        return state["ema_slow"].is_ready and state["rsi"].is_ready and state["atr"].is_ready and state["closes"].count >= 24 and state["volumes"].count >= 20

    def _features(self, state, bar):
        closes = state["closes"]
        if closes[0] <= 0 or closes[5] <= 0:
            return None
        price = float(bar.close)
        atr = float(state["atr"].current.value)
        atr_pct = max(atr / price, 1e-6)
        ret_1 = (price / closes[0] - 1.0) / atr_pct
        ret_6 = (price / closes[5] - 1.0) / atr_pct
        trend = float(state["ema_fast"].current.value - state["ema_slow"].current.value) / max(atr, 1e-8)
        rsi = (float(state["rsi"].current.value) - 50.0) / 15.0
        rng = float(bar.high - bar.low) / max(atr, 1e-8) - 1.0
        volumes = state["volumes"]
        avg_vol = sum(volumes[i] for i in range(20)) / 20.0
        volume = float(bar.volume) / max(avg_vol, 1.0) - 1.0
        return [self._clip(x, -3.0, 3.0) for x in (ret_1, ret_6, trend, rsi, rng, volume)]

    def _train_matured(self, state, current_price):
        remaining = []
        for item in state["pending"]:
            item["age"] += 1
            if item["age"] < self.label_horizon:
                remaining.append(item)
                continue
            label = 1.0 if current_price > item["price"] else 0.0
            prediction = self._predict(state, item["feature"])
            error = label - prediction
            for i, value in enumerate(item["feature"]):
                state["weights"][i] += self.learning_rate * (error * value - self.l2 * state["weights"][i])
            state["bias"] += self.learning_rate * error
            state["labels"] += 1
            self.counts["labels"] += 1
        state["pending"] = remaining

    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

    def _predict(self, state, feature):
        score = self._clip(state["bias"] + sum(w * x for w, x in zip(state["weights"], feature)), -20.0, 20.0)
        return 1.0 / (1.0 + math.exp(-score))

    def on_end_of_algorithm(self):
        self.debug("MULTI AI COUNTS: {}".format(self.counts))
