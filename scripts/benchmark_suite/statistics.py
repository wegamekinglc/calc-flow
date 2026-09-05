"""Deterministic paired-median intervals; no resampling or optional dependencies."""

from __future__ import annotations

import math
import statistics


def _median_ranks(count: int) -> tuple[int, float]:
    """Choose the tightest symmetric order interval with at least 95% coverage.

    For iid observations, coverage is P(k <= Bin(n, 0.5) <= n-k).
    Use integer tail counts so rank selection has no floating-point endpoint.
    NIST TN 2119 section 5.3 gives the order-statistic construction.
    """
    outcomes = 2**count
    tail = 0
    selected = None
    for rank in range(1, count // 2 + 1):
        tail += math.comb(count, rank - 1)
        if 40 * tail > outcomes:  # Each tail must be at most 2.5%.
            break
        selected = (rank, 1 - 2 * tail / outcomes)
    if selected is None:
        raise ValueError("too few pairs for a finite 95% median interval")
    return selected


def paired_round(baseline: list[float], candidate: list[float]) -> dict:
    """Summarize aligned, already-validated positive timing observations.

    The interval concerns median per-pair percentage change, not the ratio of
    aggregate medians. It assumes independent pairs with a common change
    distribution; alternating execution does not prove those assumptions.
    """
    changes = [
        100 * (right / left - 1)
        for left, right in zip(baseline, candidate, strict=True)
    ]
    if any(not math.isfinite(value) for value in changes):
        raise ValueError("derived paired changes are nonfinite")
    ordered = sorted(changes)
    rank, coverage = _median_ranks(len(ordered))
    return {
        "median": statistics.median(ordered),
        "low": ordered[rank - 1],
        "high": ordered[-rank],
        "coverage": coverage,
    }
