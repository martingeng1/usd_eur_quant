from __future__ import annotations

def robustness_score(n: int, oos_sharpe: float, p_value: float, recent_mean: float, historical_mean: float) -> tuple[float, str]:
    """Transparent 0-100 heuristic, not a performance forecast."""
    score = min(25, n / 8) + max(0, min(35, (oos_sharpe if oos_sharpe == oos_sharpe else -2) * 15 + 15))
    score += 20 if p_value < 0.05 else 8 if p_value < 0.10 else 0
    score += 20 if historical_mean > 0 and recent_mean > 0 else 0
    if historical_mean <= 0:
        return round(score, 1), "REJECT"
    status = "STRONG" if score >= 70 else "PROMISING" if score >= 45 else "WEAK" if score >= 25 else "REJECT"
    if recent_mean <= 0:
        status = "DECAYED"
    return round(score, 1), status
