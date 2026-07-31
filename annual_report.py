"""
Annual Performance Report - generates year-by-year returns and drawdowns
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
df = load_data(interval='1d')
df.index = pd.to_datetime(df.index, utc=True)

# Generate signals with ML
from strategies.ml_model import compute_ml_signal, get_ml_confidence
ensemble = compute_ensemble_signal(df, use_ml=True)
signal = ensemble['ensemble_signal']
signal_strength = ensemble.get('signal_strength', None)

# Run event-driven backtest
results = event_driven_backtest(
    df, signal,
    initial_capital=config.INITIAL_CAPITAL,
    spread=config.BACKTEST_SPREAD,
    commission=config.BACKTEST_COMMISSION,
    slippage=config.BACKTEST_SLIPPAGE,
    signal_strength=signal_strength,
)

equity = results['equity_curve']
trades = results['trades']

# Ensure index is datetime
equity.index = pd.to_datetime(equity.index)

# Daily resampled equity for drawdown calc
equity_daily = equity.resample('D').last().ffill()

# Resample to yearly
equity_yearly = equity.resample('YE').last()
equity_yearly.index = equity_yearly.index.year

# Trades per year
trade_count_by_year = {}
win_count_by_year = {}
pnl_by_year = {}
for t in trades:
    et = pd.Timestamp(t.get('exit_time'))
    y = et.year
    trade_count_by_year[y] = trade_count_by_year.get(y, 0) + 1
    pnl = t.get('pnl', 0)
    pnl_by_year[y] = pnl_by_year.get(y, 0) + pnl
    if pnl > 0:
        win_count_by_year[y] = win_count_by_year.get(y, 0) + 1

# Benchmark
bh_returns = df['close'].pct_change().fillna(0)
bh_equity = (1 + bh_returns).cumprod() * config.INITIAL_CAPITAL
bh_yearly = bh_equity.resample('YE').last()
bh_yearly.index = bh_yearly.index.year

# Print report
sep = '=' * 115
print()
print(sep)
print('  AUD/USD Quant Strategy v6 - Annual Performance Breakdown (2015-2026)')
print('  Target: 40%+ CAGR, Drawdown under 15%')
print(sep)
print()
header = f'  {"Year":<6} {"Start Equity":>15} {"End Equity":>16} {"Return":>10} {"Max DD":>9} {"Trades":>7} {"Wins":>6} {"Win%":>7} {"P/L($)":>14} {"B/H%":>9}'
print(header)
print('  ' + '-' * 112)

prev_eq = config.INITIAL_CAPITAL
prev_bh = config.INITIAL_CAPITAL
total_trades = 0
total_wins = 0
total_pnl = 0

prev_eq_for_dd = config.INITIAL_CAPITAL  # track equity at start of each year for DD calc

for year in range(2015, 2027):
    if year in equity_yearly.index:
        eq = equity_yearly[year]
        ret = (eq / prev_eq - 1) * 100
        t_y = trade_count_by_year.get(year, 0)
        w_y = win_count_by_year.get(year, 0)
        p_y = pnl_by_year.get(year, 0)
        wp = (w_y / t_y * 100) if t_y > 0 else 0

        total_trades += t_y
        total_wins += w_y
        total_pnl += p_y

        # Yearly max DD - use daily equity within the year
        year_mask = equity_daily.index.year == year
        year_eq_d = equity_daily[year_mask].copy()
        if len(year_eq_d) > 1:
            # The running max for DD within a year starts from this year's first value, but we carry forward
            # from last year's end. Build a combined series including the year start point.
            year_dd = (year_eq_d - year_eq_d.cummax()) / year_eq_d.cummax()
            max_dd_year_val = float(year_dd.min() * 100)
        else:
            max_dd_year_val = 0.0

        # Benchmark
        bh = bh_yearly[year] if year in bh_yearly.index else prev_bh
        bh_ret = (float(bh) / prev_bh - 1) * 100 if prev_bh != 0 else 0

        mark = '+' if ret > 0 else '-' if ret < 0 else ' '
        print(f'  {mark} {year:<4} ${prev_eq:>14,.0f}  ${float(eq):>14,.0f}  {ret:>+8.1f}%  {max_dd_year_val:>6.2f}%  {t_y:>5}  {w_y:>4}  {wp:>5.1f}%  ${float(p_y):>12,.0f}  {bh_ret:>+8.1f}%')

        prev_eq = float(eq)
        prev_bh = float(bh)
    else:
        print(f'    {year:<4} {"-":>15}  {"-":>15}  {"-":>9}  {"-":>8}  {"-":>5}  {"-":>4}  {"-":>5}  {"-":>13}  {"-":>9}')

# Totals
total_ret_pt = results['stats']['total_return'] * 100
cagr_pt = results['stats']['cagr'] * 100
max_dd_pt = results['stats']['max_drawdown'] * 100
sharpe = results['stats']['sharpe_ratio']
total_win_pct = total_wins / total_trades * 100 if total_trades > 0 else 0
bh_total_pt = float((bh_equity.iloc[-1] / config.INITIAL_CAPITAL - 1) * 100)

print('  ' + '-' * 112)
final_eq = results['stats'].get('final_equity', 0)
print(f'  TOTAL  $100,000  ${final_eq:>14,.0f}  {total_ret_pt:>+8.1f}%  {max_dd_pt:>6.2f}%  {total_trades:>5}  {total_wins:>4}  {total_win_pct:>5.1f}%  ${total_pnl:>12,.0f}  {bh_total_pt:>+8.1f}%')

print()
print('  KEY METRICS')
print('  ' + '-' * 112)

stats = results['stats']
metrics_table = [
    ('Initial Capital', f"${config.INITIAL_CAPITAL:,.0f}"),
    ('Final Equity', f"${stats['final_equity']:,.0f}"),
    ('Total Return', f"{total_ret_pt:.1f}%"),
    ('Annualized CAGR', f"{cagr_pt:.2f}%  [Target: 40%+]"),
    ('Max Drawdown', f"{max_dd_pt:.2f}%  [Target: under 15%]"),
    ('Sharpe Ratio', f"{sharpe:.2f}"),
    ('Calmar Ratio', f"{stats.get('calmar_ratio', 0):.2f}"),
    ('Total Trades', f"{total_trades}"),
    ('Win Rate', f"{total_win_pct:.1f}%"),
    ('Avg Win', f"${stats['avg_win']:,.0f}"),
    ('Avg Loss', f"${stats['avg_loss']:,.0f}"),
    ('Profit Factor', f"{stats['profit_factor']:.2f}"),
]
for name, val in metrics_table:
    print(f'  {name:<25} {val:>30}')

print(sep)
print()