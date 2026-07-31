"""
数据获取模块 — 从 Yahoo Finance 获取 AUD/USD 历史数据
"""
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 项目根
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_audusd(start="2015-01-01", end=None, interval="1h", save_csv=True):
    """
    下载 AUD/USD 历史数据

    参数
    ----
    start, end : str
        起始/结束日期
    interval : str
        K线周期: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk
    save_csv : bool
        是否保存到本地 CSV

    返回
    ----
    pd.DataFrame : 包含 OHLCV 的数据
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"[数据] 下载 AUD/USD 数据: {start} → {end}, 周期={interval}")
    ticker = yf.Ticker("AUDUSD=X")

    df = ticker.history(start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError("下载数据为空，请检查日期范围或网络连接。")

    # 清理列名
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "datetime"

    print(f"[数据] 获取到 {len(df)} 条 {interval} K线")

    if save_csv:
        csv_path = os.path.join(BASE_DIR, "data", f"audusd_{interval}.csv")
        df.to_csv(csv_path)
        print(f"[数据] 已保存到 {csv_path}")

    return df


def load_data(interval="1d", csv_only=False, max_age_hours=None):
    """
    加载数据。

    max_age_hours 为 None 时保持原有的缓存优先行为（用于回测）。
    实盘调用应传入最大缓存年龄；缓存过期时会重新下载，下载失败则
    抛出异常，避免使用陈旧行情生成订单。
    """
    csv_path = os.path.join(BASE_DIR, "data", f"audusd_{interval}.csv")

    if os.path.exists(csv_path):
        cache_is_fresh = True
        if max_age_hours is not None:
            modified = datetime.fromtimestamp(os.path.getmtime(csv_path))
            cache_is_fresh = datetime.now() - modified <= timedelta(hours=max_age_hours)
        if cache_is_fresh:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            print(f"[数据] 从缓存加载: {csv_path} ({len(df)} 条)")
            return df
        print(f"[数据] 缓存已过期，重新下载: {csv_path}")

    if csv_only:
        raise FileNotFoundError(f"未找到 {csv_path}，请先运行 fetch_data")

    from config import DATA_START, DATA_END, DATA_START_1H

    # Yahoo 1h 数据只有最近 730 天，日线数据可以追溯到 2015
    if interval in ("1h", "4h", "30m", "15m", "5m", "1m"):
        start = DATA_START_1H
    else:
        start = DATA_START
    # 实盘刷新使用今天作为 yfinance 的排他性结束日期，只获取已经
    # 完成的日线；普通回测仍尊重配置中的固定 DATA_END。
    end = None if max_age_hours is not None else DATA_END
    return fetch_audusd(start=start, end=end, interval=interval)
