from AlgorithmImports import *
from datetime import datetime, timedelta
import math


class GoldMacroFeatures(PythonData):
    """Point-in-time COT, GLD-flow and FRED features from a public raw CSV URL."""
    # Replace this with your own GitHub *raw* URL after uploading the CSV.
    FEATURES_URL = "https://raw.githubusercontent.com/YOUR_GITHUB_USER/YOUR_REPOSITORY/main/quantconnect_gold_features.csv"
    FIELDS = ["cot_net_z", "cot_change_z", "gld_flow_z", "usd_return_20d",
              "real_yield_change_20d", "vix_z"]

    def get_source(self, config, date, is_live_mode):
        return SubscriptionDataSource(self.FEATURES_URL, SubscriptionTransportMedium.REMOTE_FILE, FileFormat.CSV)

    def reader(self, config, line, date, is_live_mode):
        if not line.strip() or line.startswith("date,"):
            return None
        parts = line.strip().split(",")
        if len(parts) != 7:
            return None
        try:
            values = [float(x) for x in parts[1:]]
            if any(math.isnan(x) for x in values):
                return None
            point = GoldMacroFeatures()
            point.symbol = config.symbol
            point.time = datetime.strptime(parts[0], "%Y-%m-%d")
            # Features are published after their source session; end-time makes
            # them first available on the following daily custom-data update.
            point.end_time = point.time + timedelta(days=1)
            point.value = values[0]
            for field, value in zip(self.FIELDS, values):
                point[field] = value
            return point
        except (ValueError, IndexError):
            return None


class GoldMacroHourlyV4(QCAlgorithm):
    """MGC hourly pullback, gated by lagged COT, GLD and FRED confirmations."""
    def initialize(self):
        self.set_start_date(2012, 1, 1); self.set_end_date(2025, 12, 31); self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK); self.settings.seed_initial_prices = True
        self.mgc = self.add_future(Futures.Metals.MICRO_GOLD, Resolution.MINUTE,
            extended_market_hours=True, data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO, contract_depth_offset=0)
        self.mgc.set_filter(0, 90)
        self.feature_symbol = self.add_data(GoldMacroFeatures, "GOLD_FEATURES", Resolution.DAILY).symbol
        self.daily_fast = self.ema(self.mgc.symbol, 50, Resolution.DAILY)
        self.daily_slow = self.ema(self.mgc.symbol, 200, Resolution.DAILY)
        self.hour_ema = self.ema(self.mgc.symbol, 20, Resolution.HOUR)
        self.hour_std = self.std(self.mgc.symbol, 20, Resolution.HOUR)
        self.rsi2 = self.rsi(self.mgc.symbol, 2, MovingAverageType.WILDERS, Resolution.HOUR)
        self.atr_hour = self.atr(self.mgc.symbol, 20, MovingAverageType.SIMPLE, Resolution.HOUR)
        self.features = None; self.position = None; self.last_trade_date = None
        self.counts = {"no_features": 0, "macro": 0, "regime": 0, "pullback": 0, "entered": 0}
        self.set_warm_up(timedelta(days=300)); self.consolidate(self.mgc.symbol, timedelta(hours=1), self.on_hour)

    def on_data(self, data):
        if self.feature_symbol in data:
            point = data[self.feature_symbol]
            self.features = {field: float(point[field]) for field in GoldMacroFeatures.FIELDS}

    def on_hour(self, bar):
        if self.is_warming_up or not all(x.is_ready for x in [self.daily_fast, self.daily_slow, self.hour_ema, self.hour_std, self.rsi2, self.atr_hour]):
            return
        if self.position is None and self._has_mgc():
            self._liquidate_mgc("untracked MGC exposure"); return
        self._manage()
        if self.position is not None or self._has_mgc() or self.last_trade_date == self.time.date(): return
        if self.features is None:
            self.counts["no_features"] += 1; return
        long_macro, short_macro = self._macro_confirmation()
        if not long_macro and not short_macro:
            self.counts["macro"] += 1; return
        close = float(bar.close); upper = self.hour_ema.current.value + self.hour_std.current.value; lower = self.hour_ema.current.value - self.hour_std.current.value
        direction = 0
        if long_macro and self.daily_fast.current.value > self.daily_slow.current.value and close < lower and self.rsi2.current.value < 10: direction = 1
        elif short_macro and self.daily_fast.current.value < self.daily_slow.current.value and close > upper and self.rsi2.current.value > 90: direction = -1
        else:
            self.counts["pullback"] += 1; return
        self._enter(direction, close)

    def _macro_confirmation(self):
        f = self.features
        # Require 4/5 independent, publication-safe confirmations. COT crowding
        # is a veto, while GLD flow, dollar, real yield and VIX form the score.
        long_score = sum([f["gld_flow_z"] > 0, f["usd_return_20d"] < 0, f["real_yield_change_20d"] < 0, f["vix_z"] < 1.5, f["cot_change_z"] > -1])
        short_score = sum([f["gld_flow_z"] < 0, f["usd_return_20d"] > 0, f["real_yield_change_20d"] > 0, f["vix_z"] < 1.5, f["cot_change_z"] < 1])
        return long_score >= 4 and f["cot_net_z"] < 1.5, short_score >= 4 and f["cot_net_z"] > -1.5

    def _enter(self, direction, continuous_price):
        if self.mgc.mapped is None or not self.securities[self.mgc.mapped].is_tradable: return
        price = float(self.securities[self.mgc.mapped].price)
        if price <= 0: return
        stop = price * 1.5 * float(self.atr_hour.current.value) / continuous_price
        multiplier = float(self.securities[self.mgc.mapped].symbol_properties.contract_multiplier)
        qty = min(3, int(math.floor(self.portfolio.total_portfolio_value * .0035 / max(stop * multiplier, 1))))
        if qty < 1: return
        self.market_order(self.mgc.mapped, direction * qty, tag="MGC macro/COT/GLD pullback")
        self.position = {"symbol": self.mgc.mapped, "direction": direction, "entry": price, "stop": stop, "time": self.time}
        self.last_trade_date = self.time.date(); self.counts["entered"] += 1

    def _manage(self):
        if self.position is None: return
        p = self.position
        if not self.portfolio[p["symbol"]].invested:
            self.position = None; return
        price, cont = float(self.securities[p["symbol"]].price), float(self.securities[self.mgc.symbol].price)
        if price <= 0 or cont <= 0: return
        stop = p["entry"] - p["direction"] * p["stop"]
        mean_exit = cont >= self.hour_ema.current.value if p["direction"] > 0 else cont <= self.hour_ema.current.value
        stop_exit = price <= stop if p["direction"] > 0 else price >= stop
        if mean_exit or stop_exit or self.time - p["time"] >= timedelta(hours=23):
            self.liquidate(p["symbol"], tag="mean/stop/time exit"); self.position = None

    def _has_mgc(self):
        return any(sec.invested and sym.value.startswith("MGC") for sym, sec in self.securities.items())
    def _liquidate_mgc(self, reason):
        for sym, sec in self.securities.items():
            if sec.invested and sym.value.startswith("MGC"): self.liquidate(sym, tag=reason)
        self.position = None
    def on_end_of_algorithm(self): self.debug(f"V4 FILTER COUNTS: {self.counts}")
