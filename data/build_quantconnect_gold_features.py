"""Build a point-in-time daily feature file for QuantConnect Object Store.

Inputs have distinct publication clocks.  COT is already delayed to its first
tradable Monday by fetch_cot_gold.py; GLD and FRED inputs are delayed one more
trading day here.  No same-day close is exposed to an intraday strategy.
"""
from pathlib import Path
import pandas as pd


DATA = Path(__file__).resolve().parent


def read(name):
    x = pd.read_csv(DATA / name, parse_dates=["date"])
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return x.set_index("date").sort_index()


def zscore(x, window=252):
    return (x - x.rolling(window, min_periods=60).mean()) / x.rolling(window, min_periods=60).std()


def main():
    cot, gld, fred = read("gold_cot_weekly.csv"), read("gld_holdings_daily.csv"), read("fred_gold_macro_lag1d.csv")
    # FRED already stores a t-1 snapshot. Its non-null business-day calendar is
    # the output calendar; holiday gaps are forward-filled only after a lag.
    index = fred.dropna(how="all").index
    out = pd.DataFrame(index=index)
    cot_net = cot.speculator_net_pct_oi.reindex(index).ffill()
    out["cot_net_z"] = zscore(cot_net)
    out["cot_change_z"] = zscore(cot_net.diff(5))
    # GLD holdings are end-of-day fund data; shift before alignment.
    gld_flow = gld.gld_5d_flow_pct.shift(1).reindex(index).ffill()
    out["gld_flow_z"] = zscore(gld_flow)
    usd = fred.broad_usd_index.reindex(index).ffill()
    real_yield = fred.real_10y_yield.reindex(index).ffill()
    vix = fred.vix.reindex(index).ffill()
    out["usd_return_20d"] = usd.pct_change(20)
    out["real_yield_change_20d"] = real_yield.diff(20)
    out["vix_z"] = zscore(vix)
    # Data only becomes visible to LEAN at the following day's custom-data bar.
    # Do not fill missing early-history values with zero; the algorithm skips them.
    out.index.name = "date"
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    path = DATA / "quantconnect_gold_features.csv"
    out.to_csv(path, float_format="%.10g")
    print(f"wrote {path.name}: {len(out)} rows {out.index.min().date()}..{out.index.max().date()}")
    print(out.notna().sum().to_string())


if __name__ == "__main__":
    main()
