from __future__ import annotations
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.downloader import copy_first_local_source
from src.data.loader import load_ohlcv
from src.data.validator import quality_dict


def main() -> None:
    assets = yaml.safe_load((ROOT / "config" / "assets.yaml").read_text(encoding="utf-8"))
    candidates = [(ROOT / item).resolve() for item in assets["XAUUSD"]["source_candidates"]]
    destination = ROOT / "data" / "raw" / "xauusd_source.csv"
    saved = copy_first_local_source(candidates, destination)
    print("Copied real local gold source into project data/raw.")
    print(quality_dict(load_ohlcv(saved)))


if __name__ == "__main__":
    main()
