"""Single source of truth for CI dimensions and supported comparisons."""

from __future__ import annotations

import ast
from pathlib import Path

ROW_SCALES = tuple(10**power for power in range(1, 8))
LEGACY_SCALES = ("overhead", "small", "standard")
SQL_CASES = ("projection", "filter", "group_by", "join", "sma20", "dual_sma")
ROLLING_CASES = SQL_CASES[-2:]
# The native streaming column covers every SQL scenario except `join`: the
# bounded inner stream join emits one output stream message per matched row,
# so the 10,000,000-row engine scale cannot complete inside the suite budget.
# Revisit after the join operator gains batched output emission.
STREAM_CASES = ("projection", "filter", "group_by", "sma20", "dual_sma")
CAPABILITIES = {
    "calc-flow-sql": SQL_CASES,
    "datafusion": SQL_CASES,
    "polars": SQL_CASES,
    "calc-flow-stream": STREAM_CASES,
    "ta-lib": ROLLING_CASES,
}
THREADS = 32
BATCH_ROWS = 64_000
CONTRACT = "calc-flow-benchmark-suite-v3"
STREAM_SCOPE = "ready-enqueue-to-arrow"


def comparison_kind(case: dict, baseline_ids: frozenset[str] | None) -> str:
    """Classify one measured case against the baseline catalog membership.

    A case the baseline catalog never declared has no paired reference: the
    baseline wheel is functionally the candidate's engine for it, so gating
    the pair only measures same-runner noise. ``None`` means the baseline
    source is unavailable and every paired case keeps the interleaved gate.
    """

    if not case["backend"].startswith("calc-flow"):
        return "external"
    if baseline_ids is None:
        return "interleaved"
    return "interleaved" if case["id"] in baseline_ids else "new"


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


def _baseline_catalog_constants(
    catalog_path: Path,
) -> dict[str, tuple[str, ...]] | None:
    """Read the baseline catalog's declarative tuples without executing code.

    Only literal string-tuple assignments are accepted; anything else fails
    closed so an unparseable baseline keeps every paired case gated.
    """

    tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    constants: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _declarative_tuple(node.value, constants)
        if value is not None:
            constants[target.id] = value
    required = ("ROW_SCALES", "SQL_CASES", "ROLLING_CASES")
    if any(name not in constants for name in required):
        return None
    return constants


def _declarative_tuple(
    node: ast.expr, constants: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Accept a literal string tuple or the catalog's derived forms."""

    try:
        literal = ast.literal_eval(node)
    except ValueError:
        literal = None
    if isinstance(literal, tuple) and all(isinstance(item, str) for item in literal):
        return literal
    sliced = _sliced_tuple(node, constants)
    if sliced is not None:
        return sliced
    return _powers_of_ten_tuple(node)


def _sliced_tuple(
    node: ast.expr, constants: dict[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Match ``KNOWN_TUPLE[lower:upper]`` over already-parsed constants."""

    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    source = constants.get(node.value.id)
    if source is None:
        return None
    part = node.slice
    if not isinstance(part, ast.Slice):
        return None
    bounds = []
    for component in (part.lower, part.upper, part.step):
        if component is None:
            bounds.append(None)
            continue
        try:
            bounds.append(ast.literal_eval(component))
        except ValueError:
            return None
    lower, upper, step = bounds
    return tuple(source[lower:upper:step])


def _powers_of_ten_tuple(node: ast.expr) -> tuple[str, ...] | None:
    """Match ``tuple(10**power for power in range(start, stop))`` exactly."""

    generator = _generator_argument(node)
    if generator is None:
        return None
    power = generator.elt
    if not isinstance(power, ast.BinOp) or not isinstance(power.op, ast.Pow):
        return None
    if not isinstance(power.left, ast.Constant) or power.left.value != 10:
        return None
    if not isinstance(power.right, ast.Name):
        return None
    comprehension = generator.generators[0]
    if power.right.id != comprehension.target.id:
        return None
    bounds = _constant_range_bounds(comprehension.iter)
    if bounds is None:
        return None
    start, stop = bounds
    return tuple(str(10**exponent) for exponent in range(start, stop))


def _generator_argument(node: ast.expr) -> ast.GeneratorExp | None:
    """Return the sole generator argument of a ``tuple(...)`` call."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "tuple" or len(node.args) != 1 or node.keywords:
        return None
    generator = node.args[0]
    if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
        return None
    if generator.generators[0].ifs or generator.generators[0].is_async:
        return None
    return generator


def _constant_range_bounds(node: ast.expr) -> tuple[int, int] | None:
    """Match ``range(constant, constant)`` exactly."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "range" or node.keywords or len(node.args) != 2:
        return None
    try:
        start, stop = (ast.literal_eval(argument) for argument in node.args)
    except ValueError:
        return None
    if isinstance(start, int) and isinstance(stop, int):
        return (start, stop)
    return None


def _baseline_engine_ids(constants: dict[str, tuple[str, ...]]) -> frozenset[str]:
    sql, rolling = constants["SQL_CASES"], constants["ROLLING_CASES"]
    stream = constants.get("STREAM_CASES", rolling)
    columns = (
        ("calc-flow-sql", sql),
        ("datafusion", sql),
        ("polars", sql),
        ("calc-flow-stream", stream),
        ("ta-lib", rolling),
    )
    return frozenset(
        f"engines/{rows}/{backend}/{scenario}"
        for rows in constants["ROW_SCALES"]
        for backend, scenarios in columns
        for scenario in scenarios
    )


def _baseline_warm_ids(constants: dict[str, tuple[str, ...]]) -> frozenset[str]:
    scales = constants["ROW_SCALES"]
    dense = "1000000" in scales
    ids = set()
    for scale in scales:
        appends = (
            (1, 4, 16, 64, 640, 6_400, 64_000)
            if dense and scale == "1000000"
            else (64,)
        )
        for append in appends:
            for scenario in constants["ROLLING_CASES"]:
                ids.add(f"warm/{scale}/{append}/{scenario}")
    return frozenset(ids)


def baseline_case_ids(
    baseline_source: Path | None, shard: dict
) -> frozenset[str] | None:
    """Resolve the baseline catalog's case ids for one shard family."""

    if baseline_source is None:
        return None
    catalog_path = baseline_source / "scripts" / "benchmark_suite" / "catalog.py"
    if not catalog_path.is_file():
        return None
    constants = _baseline_catalog_constants(catalog_path)
    if constants is None:
        return None
    if shard["family"] == "engines":
        return _baseline_engine_ids(constants)
    if shard["family"] == "warm":
        return _baseline_warm_ids(constants)
    return None
