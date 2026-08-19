from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CostModel:
    spread_bps: float
    slippage_bps: float
    commission_bps: float
    financing_bps_per_day: float = 0.0

    @property
    def round_trip_return_cost(self) -> float:
        return 2 * (self.spread_bps + self.slippage_bps + self.commission_bps) / 10_000
