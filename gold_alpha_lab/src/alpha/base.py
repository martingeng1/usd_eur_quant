from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AlphaSignal:
    alpha_id: str
    timestamp: object
    direction: int
    entry_price: float
    horizon_bars: int
    confidence: float = 1.0
