"""
Annual Performance Report for AUD/USD Quant Strategy
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import config
from data.fetch_data import load_data
from ensemble.ensemble_engine import compute_ensemble_signal
from backtest.backtest_engine import event_driven_backtest

# Load data
df = load_data(interval="1d")
df.index = pd.to_datetime(df.index, utc=True)

# Generate signals with ML
ensemble = compute_ensemble_signal(df, use_ml=True)
signal = ensemble["ensemble_signal"]

# Run event-driven backtest
results = event_driven_backtest(
    df, signal,
    initial_capital=config.INITIAL_CAPITAL,
    spread=config.BACKTEST_SPREAD,
    commission=config.BACKTEST_COMMISSION,
    slippage=config.BACKTEST_SLIPPAGE,
)

equity = results["equity_curve"]
trades = results["trades"]

# Count trades/wins per year
trade_count_by_year = {}
win_count_by_year = {}
for t in trades:
    et = pd.Timestamp(t.get("exit_time"))
    y = et.year
    trade_count_by_year[y] = trade_count_by_year.get(y, 0) + 1
    if t.get("pnl", 0) > 0:
        win_count_by_year[y] = win_count_by_year.get(y, 0) + 1

# Annual equity
equity_yearly = equity.resample("YE").last()
equity_yearly.index = equity_yearly.index.year

# Benchmark
bh_returns = df["close"].pct_change().fillna(0)
bh_equity = (1 + bh_returns).cumprod() * config.INITIAL_CAPITAL
bh_yearly = bh_equity.resample("YE").last()
bh_yearly.index = bh_yearly.index.year

# Print report
print()
print("=" * 85)
print("  AUD/USD Quant Strategy - Annual Performance Breakdown (2015-2025)")
print("  Strategy: Ensemble (Trend + Reversion + Momentum + XGBoost ML)")
print("  Mode: Event-Driven with Trailing Stop & Risk Management")
print("=" * 85)
print()
print("  STRATEGY PERFORMANCE")
print("  " + "-" * 80)
print(f"  {'Year':<8} {'End Equity':>16} {'Return %':>12} {'Trades':>9} {'Wins':>7} {'Win %':>9} {'B&H %':>12}")
print("  " + "-" * 80)

prev_eq = config.INITIAL_CAPITAL
prev_bh = config.INITIAL_CAPITAL
total_trades = 0
total_wins = 0

for year in range(2015, 2027):
    if year in equity_yearly.index:
        eq = equity_yearly[year]
        ret = (eq / prev_eq - 1) * 100
        t_y = trade_count_by_year.get(year, 0)
        w_y = win_count_by_year.get(year, 0)
        wp = (w_y / t_y * 100) if t_y > 0 else 0
        total_trades += t_y
        total_wins += w_y

        # Benchmark
        bh = bh_yearly[year] if year in bh_yearly.index else prev_bh
        bh_ret = (bh / prev_bh - 1) * 100 if isinstance(prev_bh, (int, float, np.integer, np.floating)) else (float(bh) / float(prev_bh) - 1) * 100

        eq_str = f"${int(eq):,}"
        ret_str = f"{ret:+.1f}%"
        t_str = f"{t_y}"
        w_str = f"{w_y}"
        wp_str = f"{wp:.1f}%"
        bh_str = f"{bh_ret:+.1f}%"

        # Determine color indicator
        marks = "+" if ret > 0 else "-" if ret < 0 else " "

        print(f"  {marks} {year:<5} {eq_str:>15}  {ret_str:>10}  {t_str:>7}  {w_str:>5}  {wp_str:>7}  {bh_str:>10}")

        prev_eq = float(eq)
        prev_bh = float(bh)
    else:
        print(f"    {year:<5} {'-':>15}  {'-':>10}  {'-':>7}  {'-':>5}  {'-':>7}  {'-':>10}")

print("  " + "-" * 80)
total_ret = results["stats"]["total_return"] * 100
cagr = results["stats"]["cagr"] * 100
max_dd = results["stats"]["max_drawdown"] * 100
sharpe = results["stats"]["sharpe_ratio"]
total_win_pct = total_wins / total_trades * 100 if total_trades > 0 else 0

bh_total = float((bh_equity.iloc[-1] / config.INITIAL_CAPITAL - 1) * 100)

print(f"  {'TOTAL':<8} {'$' + format(int(results['stats']['final_equity']), ','):>15}  {total_ret:>+10.1f}%  {total_trades:>7}  {total_wins:>5}  {total_win_pct:>7.1f}%  {bh_total:>+10.1f}%")

print()
print("  KEY METRICS")
print("  " + "-" * 80)
metrics = [
    (f"Initial Capital", f"${config.INITIAL_CAPITAL:,}"),
    (f"Final Equity", f"${results['stats']['final_equity']:,.0f}"),
    (f"Total Return", f"{total_ret:.1f}%"),
    (f"Annualized CAGR", f"{cagr:.2f}%"),
    (f"Max Drawdown", f"{max_dd:.2f}%"),
    (f"Sharpe Ratio", f"{sharpe:.2f}"),
    (f"Total Trades", f"{total_trades}"),
    (f"Win Rate", f"{total_win_pct:.1f}%"),
    (f"Profit Factor", f"{results['stats']['profit_factor']:.2f}"),
    (f"Calmar Ratio", f"{results['stats']['calmar_ratio']:.2f}"),
]
for name, val in metrics:
    print(f"  {name:<20} {val:>15}")

print("=" * 85)
print()