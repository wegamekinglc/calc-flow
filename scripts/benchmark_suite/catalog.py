"""Single source of truth for CI dimensions and supported comparisons."""

from __future__ import annotations

ROW_SCALES = tuple(10**power for power in range(1, 8))
LEGACY_SCALES = ("overhead", "small", "standard")
SQL_CASES = ("projection", "filter", "group_by", "join", "sma20", "dual_sma")
ROLLING_CASES = SQL_CASES[-2:]
CAPABILITIES = {
    "calc-flow-sql": SQL_CASES,
    "datafusion": SQL_CASES,
    "polars": SQL_CASES,
    "calc-flow-stream": ROLLING_CASES,
    "ta-lib": ROLLING_CASES,
}
THREADS = 32
BATCH_ROWS = 64_000
CONTRACT = "calc-flow-benchmark-suite-v2"
STREAM_SCOPE = "ready-enqueue-to-arrow"


def engine_cases(rows: int | None = None) -> list[dict]:
    sizes = ROW_SCALES if rows is None else (rows,)
    return [
        {
            "id": f"engines/{size}/{backend}/{scenario}",
            "family": "engines",
            "backend": backend,
            "scenario": scenario,
            "rows": size,
            "scope": (
                STREAM_SCOPE if backend == "calc-flow-stream" else "execute-to-arrow"
            ),
        }
        for size in sizes
        for backend, scenarios in CAPABILITIES.items()
        for scenario in scenarios
    ]


def warm_cases(history: int) -> list[dict]:
    increments = (1, 4, 16, 64, 640, 6_400, 64_000) if history == 1_000_000 else (64,)
    return [
        {
            "id": f"warm/{history}/{append}/{scenario}",
            "family": "warm",
            "backend": "calc-flow-stream",
            "scenario": scenario,
            "rows": append,
            "history_rows": history,
            "entities": 1,
            "scope": "warm-enqueue-to-arrow",
        }
        for append in increments
        for scenario in ROLLING_CASES
    ]


def shards() -> list[dict]:
    return [
        *(
            {"id": f"python-{scale}", "family": "python", "scale": scale}
            for scale in LEGACY_SCALES
        ),
        *(
            {"id": f"{family}-{rows}", "family": family, "rows": rows}
            for family in ("engines", "warm")
            for rows in ROW_SCALES
        ),
        *(
            {"id": family, "family": family}
            for family in ("rust", "studio", "frontend", "lifecycle")
        ),
    ]


def get_shard(identifier: str) -> dict:
    for shard in shards():
        if shard["id"] == identifier:
            return shard
    raise ValueError(f"unknown benchmark shard: {identifier!r}")


def shard_cases(shard: dict) -> list[dict]:
    if shard["family"] == "engines":
        return engine_cases(shard["rows"])
    if shard["family"] == "warm":
        return warm_cases(shard["rows"])
    raise ValueError("legacy benchmark cases are discovered from their native runners")
