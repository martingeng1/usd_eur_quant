from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research.scanner import scan_session_alpha, write_scan


def main() -> None:
    source = ROOT / "data" / "raw" / "xauusd_source.csv"
    if not source.exists():
        raise SystemExit("No project-local raw data. Run: python scripts/download_data.py")
    result = scan_session_alpha(source, ROOT)
    write_scan(result, ROOT / "output")
    print("ACTUAL DATA QUALITY:", result.quality)
    print(result.summary.to_string(index=False))
    print("Wrote CSV and Markdown reports under output/reports.")


if __name__ == "__main__":
    main()
