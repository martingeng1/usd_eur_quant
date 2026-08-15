from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class GoldMacroFeatures(PythonData):
    URL = "https://raw.githubusercontent.com/martingeng1/usd_eur_quant/codex/quantconnect-macro-features/data/quantconnect_gold_features.csv"
    FIELDS = ["cot_net_z", "cot_change_z", "gld_flow_z", "usd_return_20d", "real_yield_change_20d", "vix_z"]
    def get_source(self, config, date, is_live_mode): return SubscriptionDataSource(self.URL, SubscriptionTransportMedium.REMOTE_FILE, FileFormat.CSV)
    def reader(self, config, line, date, is_live_mode):
        if not line.strip() or line.startswith("date,"): return None
        try:
            p = line.strip().split(","); values = [float(x) for x in p[1:]]
            if len(values) != 6 or any(math.isnan(x) for x in values): return None
            x = GoldMacroFeatures(); x.symbol = config.symbol; x.time = datetime.strptime(p[0], "%Y-%m-%d"); x.end_time = x.time + timedelta(days=1); x.value = values[0]
            for k, v in zip(self.FIELDS, values): x[k] = v
            return x
        except (ValueError, IndexError): return None


class GoldMacroStrengthBreakoutV6(QCAlgorithm):
    """V6: a stronger, independent subset of V5 macro-breakout opportunities."""
    def initialize(self):
        self.set_start_date(2012, 1, 1); self.set_end_date(2025, 12, 31); self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK); self.settings.seed_initial_prices = True
        self.mgc = self.add_future(Futures.Metals.MICRO_GOLD, Resolution.MINUTE, extended_market_hours=True,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST, data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO, contract_depth_offset=0)
        self.mgc.set_filter(0, 90)
        self.feature_symbol = self.add_data(GoldMacroFeatures, "GOLD_FEATURES", Resolution.DAILY).symbol
        self.ema20 = self.ema(self.mgc.symbol, 20, Resolution.DAILY); self.ema60 = self.ema(self.mgc.symbol, 60, Resolution.DAILY)
        self.atr20 = self.atr(self.mgc.symbol, 20, MovingAverageType.SIMPLE, Resolution.DAILY)
        self.bars = RollingWindow[TradeBar](56); self.features = None; self.position = None
        self.counts = {"no_features": 0, "macro": 0, "trend_strength": 0, "breakout_strength": 0, "entered": 0}
        self.set_warm_up(timedelta(days=300)); self.consolidate(self.mgc.symbol, timedelta(days=1), self.on_daily)

    def on_data(self, data):
        if self.feature_symbol in data:
            p = data[self.feature_symbol]; self.features = {k: float(p[k]) for k in GoldMacroFeatures.FIELDS}

    def on_daily(self, bar):
        if self.is_warming_up or not self.atr20.is_ready or self.bars.count < 55:
            self.bars.add(bar); return
        if self.position is None and self._has_mgc(): self._liquidate_mgc("untracked MGC exposure")
        self._manage(bar)
        if self.position is not None or self._has_mgc(): self.bars.add(bar); return
        if self.features is None:
            self.counts["no_features"] += 1; self.bars.add(bar); return
        direction = self._macro_direction()
        if direction == 0:
            self.counts["macro"] += 1; self.bars.add(bar); return
        atr = float(self.atr20.current.value); spread = float(self.ema20.current.value - self.ema60.current.value)
        if (direction > 0 and spread < .5 * atr) or (direction < 0 and spread > -.5 * atr):
            self.counts["trend_strength"] += 1; self.bars.add(bar); return
        high20 = max(float(self.bars[i].high) for i in range(20)); low20 = min(float(self.bars[i].low) for i in range(20))
        # A close only marginally over the channel is a noisy breakout. Require
        # an ATR-scaled excess instead of fitting a price-specific threshold.
        strong = bar.close > high20 + .25 * atr if direction > 0 else bar.close < low20 - .25 * atr
        if not strong:
            self.counts["breakout_strength"] += 1; self.bars.add(bar); return
        self._enter(direction, float(bar.close)); self.bars.add(bar)

    def _macro_direction(self):
        f = self.features
        long_all = all([f["gld_flow_z"] > 0, f["usd_return_20d"] < 0, f["real_yield_change_20d"] < 0, f["vix_z"] < 1.5, f["cot_change_z"] > -1, f["cot_net_z"] < 1.5])
        short_all = all([f["gld_flow_z"] < 0, f["usd_return_20d"] > 0, f["real_yield_change_20d"] > 0, f["vix_z"] < 1.5, f["cot_change_z"] < 1, f["cot_net_z"] > -1.5])
        return 1 if long_all else -1 if short_all else 0

    def _enter(self, direction, continuous_price):
        if self.mgc.mapped is None or not self.securities[self.mgc.mapped].is_tradable: return
        price = float(self.securities[self.mgc.mapped].price)
        if price <= 0: return
        distance = price * 2 * float(self.atr20.current.value) / continuous_price
        multiplier = float(self.securities[self.mgc.mapped].symbol_properties.contract_multiplier)
        qty = min(3, int(math.floor(self.portfolio.total_portfolio_value * .0035 / max(distance * multiplier, 1))))
        if qty < 1: return
        self.market_order(self.mgc.mapped, direction * qty, tag="MGC macro strength breakout")
        self.position = {"symbol": self.mgc.mapped, "direction": direction, "entry": price, "stop": distance, "days": 0}; self.counts["entered"] += 1

    def _manage(self, bar):
        if self.position is None: return
        p = self.position
        if not self.portfolio[p["symbol"]].invested: self.position = None; return
        price = float(self.securities[p["symbol"]].price)
        if price <= 0: return
        p["days"] += 1; stop = p["entry"] - p["direction"] * p["stop"]
        low10 = min(float(self.bars[i].low) for i in range(10)); high10 = max(float(self.bars[i].high) for i in range(10))
        trend_exit = bar.close < low10 if p["direction"] > 0 else bar.close > high10
        stop_exit = price <= stop if p["direction"] > 0 else price >= stop
        if trend_exit or stop_exit or p["days"] >= 10: self.liquidate(p["symbol"], tag="trend/stop/time exit"); self.position = None

    def _has_mgc(self): return any(sec.invested and sym.value.startswith("MGC") for sym, sec in self.securities.items())
    def _liquidate_mgc(self, reason):
        for sym, sec in self.securities.items():
            if sec.invested and sym.value.startswith("MGC"): self.liquidate(sym, tag=reason)
        self.position = None
    def on_end_of_algorithm(self): self.debug(f"V6 FILTER COUNTS: {self.counts}")
