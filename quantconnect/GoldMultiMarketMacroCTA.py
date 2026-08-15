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


class GoldMultiMarketMacroCTA(QCAlgorithm):
    """Seven-market daily trend portfolio with gold macro risk gating."""
    def initialize(self):
        self.set_start_date(2012, 1, 1); self.set_end_date(2025, 12, 31); self.set_cash(250000)
        self.set_time_zone(TimeZones.NEW_YORK); self.settings.seed_initial_prices = True
        specs = {
            "gold": Futures.Metals.GOLD, "silver": Futures.Metals.SILVER,
            "copper": Futures.Metals.COPPER, "crude": Futures.Energy.CRUDE_OIL_WTI,
            "equity": Futures.Indices.SP_500_E_MINI, "rates": Futures.Financials.Y_10_TREASURY_NOTE,
            "euro": Futures.Currencies.EUR,
        }
        self.markets = {}; self.states = {}
        self.roots = {"gold": "GC", "silver": "SI", "copper": "HG", "crude": "CL", "equity": "ES", "rates": "ZN", "euro": "6E"}
        for name, ticker in specs.items():
            future = self.add_future(ticker, Resolution.MINUTE, extended_market_hours=True,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO, contract_depth_offset=0)
            future.set_filter(0, 90); self.markets[name] = future
            state = {"ema20": self.ema(future.symbol, 20, Resolution.DAILY),
                     "ema60": self.ema(future.symbol, 60, Resolution.DAILY),
                     "atr": self.atr(future.symbol, 20, MovingAverageType.SIMPLE, Resolution.DAILY),
                     "bars": RollingWindow[TradeBar](56), "position": None}
            self.states[name] = state
            self.consolidate(future.symbol, timedelta(days=1), lambda bar, n=name: self.on_daily(n, bar))
        self.feature_symbol = self.add_data(GoldMacroFeatures, "GOLD_FEATURES", Resolution.DAILY).symbol
        self.features = None; self.counts = {"no_macro": 0, "signals": 0, "entries": 0, "exits": 0}
        self.set_warm_up(timedelta(days=300))

    def on_data(self, data):
        if self.feature_symbol in data:
            p = data[self.feature_symbol]; self.features = {k: float(p[k]) for k in GoldMacroFeatures.FIELDS}

    def on_daily(self, name, bar):
        s = self.states[name]; bars = s["bars"]
        if self.is_warming_up or not s["atr"].is_ready or bars.count < 55:
            bars.add(bar); return
        self._manage(name, bar)
        if s["position"] is not None or self._market_exposure(name):
            bars.add(bar); return
        if self.features is None:
            self.counts["no_macro"] += 1; bars.add(bar); return
        macro = self._macro_direction()
        # Commodities/equities follow the gold macro direction. Rates/euro are
        # allowed independently; diversification is the risk-control layer.
        if name in ("gold", "silver", "copper", "crude", "equity") and macro == 0:
            bars.add(bar); return
        direction = 1 if bar.close > max(float(bars[i].high) for i in range(20)) else -1 if bar.close < min(float(bars[i].low) for i in range(20)) else 0
        if direction == 0 or s["ema20"].current.value <= s["ema60"].current.value and direction > 0 or s["ema20"].current.value >= s["ema60"].current.value and direction < 0:
            bars.add(bar); return
        if name in ("gold", "silver", "copper", "crude", "equity") and direction != macro:
            bars.add(bar); return
        self._enter(name, direction, float(bar.close)); bars.add(bar)

    def _macro_direction(self):
        f = self.features
        long_score = sum([f["gld_flow_z"] > 0, f["usd_return_20d"] < 0, f["real_yield_change_20d"] < 0, f["vix_z"] < 1.5, f["cot_change_z"] > -1])
        short_score = sum([f["gld_flow_z"] < 0, f["usd_return_20d"] > 0, f["real_yield_change_20d"] > 0, f["vix_z"] < 1.5, f["cot_change_z"] < 1])
        return 1 if long_score >= 4 and f["cot_net_z"] < 1.5 else -1 if short_score >= 4 and f["cot_net_z"] > -1.5 else 0

    def _enter(self, name, direction, continuous_price):
        future = self.markets[name]; mapped = future.mapped
        if mapped is None or not self.securities[mapped].is_tradable: return
        price = float(self.securities[mapped].price); atr_pct = float(self.states[name]["atr"].current.value) / continuous_price
        if price <= 0: return
        distance = price * 2 * atr_pct; multiplier = float(self.securities[mapped].symbol_properties.contract_multiplier)
        # Equal per-market risk, with one contract maximum per market.
        qty = min(1, int(math.floor(self.portfolio.total_portfolio_value * .0012 / max(distance * multiplier, 1))))
        if qty < 1: return
        self.market_order(mapped, direction * qty, tag=f"{name} macro trend")
        self.states[name]["position"] = {"symbol": mapped, "direction": direction, "entry": price, "stop": distance, "days": 0}; self.counts["entries"] += 1

    def _manage(self, name, bar):
        s = self.states[name]; p = s["position"]
        if p is None: return
        if not self.portfolio[p["symbol"]].invested: s["position"] = None; return
        price = float(self.securities[p["symbol"]].price); p["days"] += 1
        stop = p["entry"] - p["direction"] * p["stop"]
        broken = bar.close < min(float(s["bars"][i].low) for i in range(10)) if p["direction"] > 0 else bar.close > max(float(s["bars"][i].high) for i in range(10))
        stopped = price <= stop if p["direction"] > 0 else price >= stop
        if broken or stopped or p["days"] >= 10:
            self.liquidate(p["symbol"], tag="portfolio trend exit"); s["position"] = None; self.counts["exits"] += 1

    def _market_exposure(self, name):
        root = self.roots[name]
        return any(sec.invested and sym.value.replace("/", "").startswith(root) for sym, sec in self.securities.items())
    def on_end_of_algorithm(self): self.debug(f"CTA COUNTS: {self.counts}")
