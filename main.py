"""
AUD/USD 量化交易系统 — 主入口
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

import config
from data.fetch_data import fetch_audusd, load_data
from ensemble.ensemble_engine import compute_ensemble_signal
from backtest.backtest_engine import vectorized_backtest, event_driven_backtest


def print_header():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║        AUD/USD 量化交易系统 v2.0                   ║
    ║        Multi-Strategy Ensemble Trading Engine     ║
    ╚══════════════════════════════════════════════════╝
    """)


def print_stats(stats, label="绩效报告"):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    metrics = [
        ("总交易次数", stats.get("total_trades", 0), "次"),
        ("最终资金", stats.get("final_equity", 0), "USD"),
        ("总收益率", stats.get("total_return", 0) * 100, "%"),
        ("年化收益率 (CAGR)", stats.get("cagr", 0) * 100, "%"),
        ("夏普比率", stats.get("sharpe_ratio", 0), ""),
        ("最大回撤", stats.get("max_drawdown", 0) * 100, "%"),
        ("胜率", stats.get("win_rate", 0) * 100, "%"),
        ("平均盈利", stats.get("avg_win", 0), "USD"),
        ("平均亏损", stats.get("avg_loss", 0), "USD"),
        ("盈亏比", stats.get("profit_factor", 0), ""),
        ("期望值", stats.get("expectancy", 0), "USD/笔"),
        ("卡玛比率", stats.get("calmar_ratio", 0), ""),
        ("最大回撤持续(根)", stats.get("max_drawdown_duration", 0), "K线"),
    ]

    for name, value, unit in metrics:
        if isinstance(value, float) and abs(value) < 1000:
            print(f"  {name:<20} {value:>10.4f}{unit}")
        else:
            print(f"  {name:<20} {value:>10}{unit}")
    print(f"{'='*60}\n")


def print_trade_summary(trades, label="交易记录"):
    if not trades:
        print(f"\n  [{label}] 无交易记录")
        return
    print(f"\n{'='*60}")
    print(f"  {label} (共 {len(trades)} 笔)")
    print(f"{'='*60}")
    print(f"  {'时间':<22} {'方向':<8} {'入场':>10} {'出场':>10} {'盈亏':>10} {'原因'}")
    print(f"  {'-'*60}")
    for t in trades[-10:]:
        entry_str = str(t.get("entry_time", ""))[:19]
        direction = t.get("direction", "?")
        entry_p = t.get("entry_price", 0)
        exit_p = t.get("exit_price", 0)
        pnl = t.get("pnl", 0)
        reason = t.get("reason", "signal")
        print(f"  {entry_str:<22} {direction:<8} {entry_p:>10.5f} {exit_p:>10.5f} {pnl:>+10.5f} {reason}")
    print(f"{'='*60}\n")


def plot_results(results, save_path=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[图表] matplotlib 未安装，跳过绘图。pip install matplotlib")
        return

    equity = results["equity_curve"].copy()
    returns = results["returns"].copy()
    trades = results.get("trades", [])
    stats = results.get("stats", {})

    # 确保 datetime index
    equity.index = pd.to_datetime(equity.index, utc=True)
    returns.index = pd.to_datetime(returns.index, utc=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("AUD/USD Quant Trading System - Backtest Results", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, color="#2196F3", linewidth=1.5, label="Strategy Equity")
    ax1.axhline(y=config.INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    if trades:
        for t in trades:
            color = "green" if t.get("pnl", 0) > 0 else "red"
            marker = "^" if t.get("pnl", 0) > 0 else "v"
            exit_time = t.get("exit_time")
            if exit_time is not None:
                try:
                    exit_ts = pd.Timestamp(exit_time).tz_localize(None)
                    if exit_ts in equity.index:
                        ax1.scatter(exit_ts, equity.loc[exit_ts], c=color, marker=marker, s=50, alpha=0.7, zorder=5)
                except Exception:
                    pass
    ax1.set_ylabel("Equity (USD)", fontsize=11)
    ax1.set_title(f"Equity Curve | Return: {stats.get('total_return', 0)*100:.2f}% | Sharpe: {stats.get('sharpe_ratio', 0):.2f} | Win: {stats.get('win_rate', 0)*100:.1f}%")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    ax2 = axes[1]
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_title(f"Drawdown | Max DD: {stats.get('max_drawdown', 0)*100:.2f}%")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    ax3 = axes[2]
    daily_ret = returns.resample("D").sum() * 100
    ax3.bar(daily_ret.index, daily_ret.values, color=["green" if r > 0 else "red" for r in daily_ret.values], alpha=0.6, width=1)
    ax3.set_ylabel("Daily Return (%)", fontsize=11)
    ax3.set_title("Daily Return Distribution")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[图表] 已保存到 {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AUD/USD 量化交易系统")
    parser.add_argument("--train-ml", action="store_true")
    parser.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--interval", default="1h")
    args = parser.parse_args()

    print_header()

    print("\n[1/5] 获取 AUD/USD 数据...")
    try:
        df = load_data(interval=args.interval)
    except FileNotFoundError:
        df = fetch_audusd(start=config.DATA_START, end=config.DATA_END, interval=args.interval)
    print(f"  数据范围: {df.index[0]} -> {df.index[-1]}")
    print(f"  数据量: {len(df)} 条")
    print(f"  价格范围: {df['close'].min():.5f} - {df['close'].max():.5f}")

    use_ml = False
    if args.train_ml:
        print("\n[2/5] 训练 ML 模型...")
        try:
            from strategies.ml_model import train_ml_model
            model, feature_cols = train_ml_model(df)
            if model is not None:
                use_ml = True
        except ImportError:
            print("  XGBoost 不可用，跳过")

    print("\n[3/5] 生成多策略集成信号...")
    ensemble_result = compute_ensemble_signal(df, use_ml=use_ml)
    signal = ensemble_result["ensemble_signal"]
    long_pct = (signal == 1).sum() / len(signal) * 100
    short_pct = (signal == -1).sum() / len(signal) * 100
    flat_pct = (signal == 0).sum() / len(signal) * 100
    print(f"  多头: {long_pct:.1f}% | 空头: {short_pct:.1f}% | 空仓: {flat_pct:.1f}%")

    print("\n[4/5] 执行回测...")
    print("\n  --- 向量化回测 ---")
    vec_results = vectorized_backtest(df, signal, initial_capital=config.INITIAL_CAPITAL,
                                       spread=config.BACKTEST_SPREAD, commission=config.BACKTEST_COMMISSION,
                                       slippage=config.BACKTEST_SLIPPAGE)
    print_stats(vec_results["stats"], "向量化回测绩效")
    print_trade_summary(vec_results["trades"], "向量化交易记录")

    print("  --- 事件驱动回测 (含风险管理 v3 + 动态仓位 + 熔断) ---")
    signal_strength = ensemble_result.get("signal_strength", None)
    ev_results = event_driven_backtest(df, signal, initial_capital=config.INITIAL_CAPITAL,
                                         spread=config.BACKTEST_SPREAD, commission=config.BACKTEST_COMMISSION,
                                         slippage=config.BACKTEST_SLIPPAGE,
                                         signal_strength=signal_strength)
    print_stats(ev_results["stats"], "事件驱动回测绩效 (含风险控制 v3)")
    print_trade_summary(ev_results["trades"], "事件驱动交易记录")

    # 显示风控统计
    long_pct_sig = (signal == 1).sum() / len(signal) * 100
    short_pct_sig = (signal == -1).sum() / len(signal) * 100
    flat_pct_sig = (signal == 0).sum() / len(signal) * 100
    print(f"\n  信号过滤后: 多头{long_pct_sig:.1f}% | 空头{short_pct_sig:.1f}% | 空仓{flat_pct_sig:.1f}%")
    if signal_strength is not None:
        avg_strength = signal_strength[signal != 0].mean()
        print(f"  平均信号强度: {avg_strength:.3f}")

    print("  --- 基准对比 (买入持有 AUD/USD) ---")
    bh_returns = df["close"].pct_change().fillna(0)
    bh_equity = (1 + bh_returns).cumprod() * config.INITIAL_CAPITAL
    bh_equity.index = pd.to_datetime(bh_equity.index, utc=True)
    bh_total_return = (bh_equity.iloc[-1] - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL
    running_max = bh_equity.cummax()
    bh_max_dd = (bh_equity - running_max).min() / running_max
    trading_days = len(bh_equity)
    bh_cagr = (bh_equity.iloc[-1] / config.INITIAL_CAPITAL) ** (1 / max(trading_days / 252, 0.01)) - 1
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(252) if bh_returns.std() > 0 else 0
    bh_stats = {"total_return": float(bh_total_return), "cagr": float(bh_cagr), "sharpe_ratio": float(bh_sharpe), "max_drawdown": float(bh_max_dd.values[0] if hasattr(bh_max_dd, 'values') else bh_max_dd), "total_trades": 0, "final_equity": float(bh_equity.iloc[-1]), "win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "expectancy": 0, "calmar_ratio": 0}
    print_stats(bh_stats, "买入持有基准绩效")

    if args.plot:
        print("\n[5/5] 生成回测图表...")
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.png")
        plot_results(ev_results, save_path=plot_path)

    print("\n" + "=" * 60)
    print("  Final Comparison")
    print("=" * 60)
    strat_ret = ev_results["stats"].get("total_return", 0) * 100
    strat_sharpe = ev_results["stats"].get("sharpe_ratio", 0)
    strat_dd = ev_results["stats"].get("max_drawdown", 0) * 100
    strat_wr = ev_results["stats"].get("win_rate", 0) * 100
    strat_pf = ev_results["stats"].get("profit_factor", 0)
    print(f"  {'Metric':<20} {'Strategy':>15} {'Buy&Hold':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Total Return':<20} {strat_ret:>14.2f}% {bh_total_return*100:>14.2f}%")
    print(f"  {'Sharpe Ratio':<20} {strat_sharpe:>15.4f} {bh_sharpe:>15.4f}")
    bh_dd_val = float(bh_max_dd.values[0] if hasattr(bh_max_dd, 'values') else bh_max_dd)
    print(f"  {'Max Drawdown':<20} {strat_dd:>14.2f}% {bh_dd_val*100:>14.2f}%")
    print(f"  {'Win Rate':<20} {strat_wr:>14.2f}% {'N/A':>15}")
    print(f"  {'Profit Factor':<20} {strat_pf:>15.4f} {'N/A':>15}")
    print("=" * 60)
    print(f"\n  Initial Capital: ${config.INITIAL_CAPITAL:,.0f}")
    print(f"  Strategy Final:  ${ev_results['stats'].get('final_equity', 0):,.0f}")
    print(f"  Buy&Hold Final:  ${bh_equity.iloc[-1]:,.0f}")
    print(f"  Excess Return:  ${ev_results['stats'].get('final_equity', 0) - bh_equity.iloc[-1]:,.0f}\n")


if __name__ == "__main__":
    main()