from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import yaml
from src.alpha.breakout_alpha import asian_breakout_events, label_horizons
from src.alpha.fake_breakout_alpha import fake_breakout_events
from src.backtest.costs import CostModel
from src.data.loader import load_ohlcv
from src.data.validator import quality_dict
from src.features.breakout import add_asian_range
from src.features.session import add_session_flags
from src.features.trend import add_trend_state
from src.features.volatility import add_atr
from src.regime.detector import detect_regime
from src.research.robustness import robustness_score
from src.research.statistics import benjamini_hochberg, summarize_returns


@dataclass
class ScanResult:
    observations: pd.DataFrame
    summary: pd.DataFrame
    quality: dict[str, object]


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_features(data_path: str | Path, root: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(root)
    frame = load_ohlcv(data_path)
    source_quality = quality_dict(frame)
    research = _yaml(root / "config" / "research.yaml")
    frame = frame.loc[frame.index >= pd.Timestamp(research["target_start"], tz="UTC")].copy()
    quality = {"source": source_quality, "analysis": quality_dict(frame)}
    sessions = _yaml(root / "config" / "sessions.yaml")["sessions"]
    frame = add_atr(frame)
    frame = add_trend_state(frame)
    frame = add_session_flags(frame, sessions)
    frame = add_asian_range(frame)
    frame = asian_breakout_events(frame)
    frame = fake_breakout_events(frame)
    frame = label_horizons(frame)
    frame["regime"] = detect_regime(frame)
    return frame, quality


def scan_session_alpha(data_path: str | Path, root: str | Path) -> ScanResult:
    root = Path(root)
    frame, quality = build_features(data_path, root)
    costs = _yaml(root / "config" / "costs.yaml")
    event_rows = []
    events = frame.loc[frame.breakout_event].copy()
    for horizon in (1, 2, 4):
        raw = events[f"forward_{horizon}h"]
        for scenario, values in costs.items():
            cost = CostModel(**values).round_trip_return_cost
            net = raw - cost
            stats = summarize_returns(net)
            recent = net.loc[net.index >= net.index.max() - pd.Timedelta(days=1825)]
            recent_mean = float(recent.mean()) if len(recent) else float("nan")
            score, status = robustness_score(int(stats["n"]), float(stats.get("sharpe", float("nan"))), float(stats.get("p_value", 1.0)), recent_mean, float(stats.get("mean", 0.0)))
            event_rows.append({"alpha_id": "asian_range_breakout", "horizon_hours": horizon, "cost_scenario": scenario, **stats, "recent_mean": recent_mean, "robustness_score": score, "status": status, "gross_mean": float(raw.mean())})
    # Session conditional statistics, one complete session's return predicted by prior Asian direction.
    session_rows = []
    session_returns: dict[str, pd.Series] = {}
    for name in ["asia", "london_open", "london_morning", "london_ny_overlap", "ny_morning", "ny_afternoon"]:
        flag = f"session_{name}"
        selected = frame.loc[frame[flag]].copy()
        if selected.empty:
            continue
        day = pd.Series(selected.index.tz_convert("UTC").date, index=selected.index)
        per_day = selected.close.groupby(day).agg(["first", "last"])
        r = per_day["last"] / per_day["first"] - 1
        session_returns[name] = r
        stats = summarize_returns(r)
        session_rows.append({"alpha_id": f"session_{name}", "horizon_hours": 0, "cost_scenario": "unlevered", **stats, "recent_mean": float(r.tail(252).mean()), "robustness_score": None, "status": "DESCRIPTIVE", "gross_mean": float(r.mean())})
    # Pre-registered conditional probabilities: these are descriptive tests, not trading rules.
    if {"asia", "london_morning", "ny_morning"}.issubset(session_returns):
        daily = pd.concat(session_returns, axis=1).dropna()
        conditions = {
            "P(NY up | London up > 0.5%)": (daily["london_morning"] > 0.005, daily["ny_morning"] > 0),
            "P(NY continuation | Asia up)": (daily["asia"] > 0, daily["ny_morning"] > 0),
            "P(NY reversal | London down < -0.5%)": (daily["london_morning"] < -0.005, daily["ny_morning"] > 0),
        }
        for label, (condition, outcome) in conditions.items():
            n = int(condition.sum())
            probability = float(outcome[condition].mean()) if n else float("nan")
            session_rows.append({"alpha_id": label, "horizon_hours": 0, "cost_scenario": "descriptive", "n": n,
                                 "win_rate": probability, "mean": probability - 0.5 if probability == probability else float("nan"),
                                 "median": float("nan"), "std": float("nan"), "profit_factor": float("nan"), "sharpe": float("nan"),
                                 "sortino": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "max_drawdown": float("nan"),
                                 "recent_mean": float("nan"), "robustness_score": None, "status": "DESCRIPTIVE", "gross_mean": probability})
    summary = pd.DataFrame(event_rows + session_rows)
    summary["fdr_q_value"] = benjamini_hochberg(summary["p_value"])
    observations = events[["close", "asia_high", "asia_low", "asia_range", "atr", "trend_state", "regime", "breakout_direction", "fake_break_event", "forward_1h", "forward_2h", "forward_4h", "mfe_1h", "mae_1h"]].copy()
    return ScanResult(observations=observations, summary=summary, quality=quality)


def write_scan(result: ScanResult, output_root: str | Path) -> None:
    output = Path(output_root)
    (output / "alpha_scan").mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)
    result.observations.to_csv(output / "alpha_scan" / "asian_breakout_observations.csv")
    result.summary.to_csv(output / "alpha_scan" / "session_alpha_summary.csv", index=False)
    (output / "reports" / "data_quality.json").write_text(json.dumps(result.quality, indent=2), encoding="utf-8")
    report = ["# Session Alpha Scan", "", "All results use the actual loaded data range. Values are descriptive and cost-aware, not a claim of deployable alpha.", "", "```csv", result.summary.to_csv(index=False), "```"]
    (output / "reports" / "session_alpha_report.md").write_text("\n".join(report), encoding="utf-8")
