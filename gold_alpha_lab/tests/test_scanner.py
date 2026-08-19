import numpy as np
import pandas as pd
from pathlib import Path
from src.research.scanner import scan_session_alpha


def test_scanner_runs_on_realistic_hourly_fixture(tmp_path: Path) -> None:
    idx = pd.date_range("2023-01-01", periods=600, freq="h", tz="UTC")
    p = pd.Series(1800 + np.arange(600), index=idx, dtype=float)
    data = pd.DataFrame({"datetime": idx.astype(str), "open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1})
    csv = tmp_path / "x.csv"; data.to_csv(csv, index=False)
    root = Path(__file__).resolve().parents[1]
    result = scan_session_alpha(csv, root)
    assert result.quality["analysis"]["rows"] == 600
    assert "asian_range_breakout" in set(result.summary.alpha_id)
