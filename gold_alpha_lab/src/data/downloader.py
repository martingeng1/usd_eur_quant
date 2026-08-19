from __future__ import annotations
import shutil
from pathlib import Path


def copy_first_local_source(candidates: list[str | Path], destination: str | Path) -> Path:
    destination = Path(destination)
    for candidate in candidates:
        source = Path(candidate)
        if source.exists() and source.stat().st_size > 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return destination
    raise FileNotFoundError("No real local gold source found; no synthetic fallback is permitted")
