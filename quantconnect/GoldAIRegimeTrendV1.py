from AlgorithmImports import *
from datetime import timedelta
import math


class GoldAIRegimeTrendV1(QCAlgorithm):
    """MGC hourly trend strategy with a leakage-free online ML risk filter.

    The model is deliberately small and transparent: it learns the probability
    that gold closes higher 12 hours later from information available at the
    current hourly close.  It is not a price oracle; trades require both model
    confidence and a compatible trend/volatility regime.
    """
    def initialize(self):
        self.set_start_date(2012, 1, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK)
        self.settings.seed_initial_prices = True

        self.mgc = self.add_future(
            Futures.Metals.MICRO_GOLD, Resolution.MINUTE,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0)
        self.mgc.set_filter(0, 60)
        symbol = self.mgc.symbol
        self.ema_fast = self.ema(symbol, 20, Resolution.HOUR)
        self.ema_slow = self.ema(symbol, 60, Resolution.HOUR)
        self.rsi = self.rsi(symbol, 14, MovingAverageType.WILDERS, Resolution.HOUR)
        self.atr = self.atr(symbol, 20, MovingAverageType.WILDERS, Resolution.HOUR)
        self.closes = RollingWindow[float](80)
        self.volumes = RollingWindow[float](30)
        self.consolidate(symbol, timedelta(hours=1), self.on_hour)

        self.weights = [0.0] * 6
        self.bias = 0.0
        self.learning_rate = 0.025
        self.l2 = 0.0005
        self.pending = []
        self.labels_seen = 0
        self.position = None
        self.day = None
        self.trades_today = 0
        self.counts = {"labels": 0, "model_signals": 0, "trend_rejected": 0,
                       "risk_rejected": 0, "entries": 0, "exits": 0}
        self.set_warm_up(timedelta(days=120))

    def on_hour(self, bar):
        price = float(bar.close)
        self._train_matured_labels(price)
        if self.is_warming_up or not self._ready():
            self.closes.add(price)
            self.volumes.add(float(bar.volume))
            return

        if self.day != self.time.date():
            self.day = self.time.date()
            self.trades_today = 0

        self._manage(price)
        feature = self._features(bar)
        probability = self._predict(feature)
        # Store today's feature only after its prediction.  It cannot affect
        # the model until the 12-hour outcome exists.
        self.pending.append({"price": price, "feature": feature, "age": 0})

        if self.position is None and self.trades_today < 2 and self.labels_seen >= 300:
            direction = 1 if probability >= 0.60 else -1 if probability <= 0.40 else 0
            if direction != 0:
                self.counts["model_signals"] += 1
                trend = float(self.ema_fast.current.value - self.ema_slow.current.value)
                atr = float(self.atr.current.value)
                # Model decides confidence; price structure vetoes trades
                # against the prevailing hourly trend.
                if (direction > 0 and trend > 0.20 * atr) or (direction < 0 and trend < -0.20 * atr):
                    self._enter(direction, price, atr, probability)
                else:
                    self.counts["trend_rejected"] += 1

        self.closes.add(price)
        self.volumes.add(float(bar.volume))

    def _ready(self):
        return self.ema_slow.is_ready and self.rsi.is_ready and self.atr.is_ready and self.closes.count >= 24 and self.volumes.count >= 20

    def _features(self, bar):
        price = float(bar.close)
        atr_pct = max(float(self.atr.current.value) / max(price, 1e-8), 1e-6)
        ret_1 = (price / self.closes[0] - 1.0) / atr_pct
        ret_6 = (price / self.closes[5] - 1.0) / atr_pct
        trend = float(self.ema_fast.current.value - self.ema_slow.current.value) / max(float(self.atr.current.value), 1e-8)
        rsi = (float(self.rsi.current.value) - 50.0) / 15.0
        rng = float(bar.high - bar.low) / max(float(self.atr.current.value), 1e-8)
        avg_vol = sum(self.volumes[i] for i in range(20)) / 20.0
        volume = float(bar.volume) / max(avg_vol, 1.0) - 1.0
        return [self._clip(x, -3.0, 3.0) for x in (ret_1, ret_6, trend, rsi, rng - 1.0, volume)]

    def _train_matured_labels(self, current_price):
        surviving = []
        for item in self.pending:
            item["age"] += 1
            if item["age"] < 12:
                surviving.append(item)
                continue
            label = 1.0 if current_price > item["price"] else 0.0
            prediction = self._predict(item["feature"])
            error = label - prediction
            for i in range(len(self.weights)):
                self.weights[i] += self.learning_rate * (error * item["feature"][i] - self.l2 * self.weights[i])
            self.bias += self.learning_rate * error
            self.labels_seen += 1
            self.counts["labels"] += 1
        self.pending = surviving

    def _predict(self, feature):
        score = self.bias + sum(w * x for w, x in zip(self.weights, feature))
        score = self._clip(score, -20.0, 20.0)
        return 1.0 / (1.0 + math.exp(-score))

    def _enter(self, direction, price, atr, probability):
        mapped = self.mgc.mapped
        if mapped is None or mapped not in self.securities or not self.securities[mapped].is_tradable:
            return
        stop_distance = 1.7 * atr
        multiplier = float(self.securities[mapped].symbol_properties.contract_multiplier)
        if stop_distance * multiplier > self.portfolio.total_portfolio_value * .018:
            self.counts["risk_rejected"] += 1
            return
        self.market_order(mapped, direction, tag="AI regime trend p={:.2f}".format(probability))
        self.position = {"symbol": mapped, "direction": direction, "entry": price,
                         "stop": price - direction * stop_distance,
                         "target": price + direction * 2.2 * stop_distance, "hours": 0}
        self.trades_today += 1
        self.counts["entries"] += 1

    def _manage(self, price):
        if self.position is None:
            return
        p = self.position
        if not self.portfolio[p["symbol"]].invested:
            self.position = None
            return
        p["hours"] += 1
        stopped = price <= p["stop"] if p["direction"] > 0 else price >= p["stop"]
        targeted = price >= p["target"] if p["direction"] > 0 else price <= p["target"]
        if stopped or targeted or p["hours"] >= 18:
            self.liquidate(p["symbol"], tag="AI exit")
            self.position = None
            self.counts["exits"] += 1

    @staticmethod
    def _clip(value, low, high):
        return max(low, min(high, value))

    def on_end_of_algorithm(self):
        self.debug("AI COUNTS: {}; weights={}".format(self.counts, [round(w, 3) for w in self.weights]))
