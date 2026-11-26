"""Deterministic timestamp-major input for the report operator comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatorWorkload:
    """Rows arrive in complete timestamp groups with one row per entity."""

    rows: int
    entities: int
    chunk_ticks: int

    def __post_init__(self) -> None:
        if not (0 < self.entities <= self.rows and self.chunk_ticks > 0):
            raise ValueError("rows, entities, and chunk_ticks must be positive")
        if self.rows % self.entities:
            raise ValueError("operator workload must contain complete timestamp groups")

    @property
    def chunk_rows(self) -> int:
        return self.entities * self.chunk_ticks


def columns(rows: int, entities: int) -> dict[str, np.ndarray]:
    """Reconstruct the same logical values in both Python environments."""
    sequence = np.arange(rows, dtype=np.uint64)
    ticks = (sequence // entities).astype(np.int64)
    symbols = np.asarray([f"S{index:05d}" for index in range(entities)])
    x = 32.0 + ((sequence * 37) % 8191) / 64.0
    y = 1.0 + ((sequence * 17) % 4049) / 128.0
    return {
        "sequence": sequence,
        "tick": ticks,
        "symbol": symbols[sequence % entities],
        "x": x,
        "y": y,
        "unit": (1.0 + (sequence % 63)) / 64.0,
        "positive": 1.0 + (sequence % 97) / 64.0,
        "signed": (sequence % 61) / 16.0 - 1.875,
        "flag": (sequence % 79) < 41,
        "fill": np.where(sequence % 29 == 0, np.nan, x),
    }
