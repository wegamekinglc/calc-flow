"""Single source of truth for CI dimensions and supported comparisons."""

from __future__ import annotations

import ast
from pathlib import Path

ROW_SCALES = tuple(10**power for power in range(1, 8))
LEGACY_SCALES = ("overhead", "small", "standard")
SQL_CASES = ("projection", "filter", "group_by", "join", "sma20", "dual_sma")
ROLLING_CASES = SQL_CASES[-2:]
# Keep this a literal tuple: the suite resolves baseline case ids by parsing
# the baseline catalog's declarative forms, and derived assignments fail
# closed to degraded gating.
STREAM_CASES = ("projection", "filter", "group_by", "join", "sma20", "dual_sma")
CAPABILITIES = {
    "calc-flow-sql": SQL_CASES,
    "datafusion": SQL_CASES,
    "polars": SQL_CASES,
    "calc-flow-stream": STREAM_CASES,
    "ta-lib": ROLLING_CASES,
}
# The native-stream join column carries evidence only through the 100k tier
# (user-directed pacing constraint, DAL-290, 2026-09-20): per-sample
# performance above that scale (≈5 s at 1M and ≈200 s at 10M on the dev
# machine, because the join retains one state row per matched input row for
# the whole run) would slow the whole suite's cadence. Larger tiers stay
# unsupported in the catalog rather than being measured to fill the column;
# see docs/benchmark-suite.md.
STREAM_JOIN_MAX_ROWS = 100_000
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


def _measured_stream_join(
    backend: str, scenario: str, size: int, cap: int | None
) -> bool:
    """Whether one native-stream join scale stays in the measured catalog.

    ``cap is None`` means the catalog declared no cap, so every declared
    stream-join case is measured.
    """

    if cap is None:
        return True
    return not (backend == "calc-flow-stream" and scenario == "join" and size > cap)


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
        if _measured_stream_join(backend, scenario, size, STREAM_JOIN_MAX_ROWS)
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
) -> dict[str, tuple[str, ...] | int] | None:
    """Read the baseline catalog's declarative constants without executing code.

    Only literal string-tuple and integer assignments are accepted, and every
    required constant must resolve to a string tuple; anything else fails
    closed so an unparseable baseline keeps every paired case gated.
    """

    tree = ast.parse(catalog_path.read_text(encoding="utf-8"))
    constants: dict[str, tuple[str, ...] | int] = {}
    for node in tree.body:
        assigned = _assigned_constant(node, constants)
        if assigned is not None:
            constants[assigned[0]] = assigned[1]
    required = ("ROW_SCALES", "SQL_CASES", "ROLLING_CASES")
    if any(not isinstance(constants.get(name), tuple) for name in required):
        return None
    return constants


def _assigned_constant(
    node: ast.stmt, constants: dict[str, tuple[str, ...]]
) -> tuple[str, tuple[str, ...] | int] | None:
    """Return one top-level ``NAME = literal`` assignment, else ``None``."""

    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    value: tuple[str, ...] | int | None = _declarative_tuple(node.value, constants)
    if value is None:
        value = _declarative_int(node.value)
    return None if value is None else (target.id, value)


def _declarative_int(node: ast.expr) -> int | None:
    """Accept a literal integer assignment."""

    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


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
    power = _ten_to_name_power(generator.elt)
    if power is None:
        return None
    comprehension = generator.generators[0]
    if power.id != comprehension.target.id:
        return None
    bounds = _constant_range_bounds(comprehension.iter)
    if bounds is None:
        return None
    start, stop = bounds
    return tuple(str(10**exponent) for exponent in range(start, stop))


def _ten_to_name_power(node: ast.expr) -> ast.Name | None:
    """Match ``10 ** name`` and return the exponent variable."""

    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Pow):
        return None
    if not isinstance(node.left, ast.Constant) or node.left.value != 10:
        return None
    if not isinstance(node.right, ast.Name):
        return None
    return node.right


def _named_call(node: ast.expr, name: str, arity: int) -> ast.Call | None:
    """Match a bare ``name`` call with exactly ``arity`` positional arguments."""

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != name or node.keywords or len(node.args) != arity:
        return None
    return node


def _generator_argument(node: ast.expr) -> ast.GeneratorExp | None:
    """Return the sole generator argument of a ``tuple(...)`` call."""

    call = _named_call(node, "tuple", 1)
    if call is None:
        return None
    generator = call.args[0]
    if not isinstance(generator, ast.GeneratorExp) or len(generator.generators) != 1:
        return None
    comprehension = generator.generators[0]
    if comprehension.ifs or comprehension.is_async:
        return None
    return generator


def _constant_range_bounds(node: ast.expr) -> tuple[int, int] | None:
    """Match ``range(constant, constant)`` exactly."""

    call = _named_call(node, "range", 2)
    if call is None:
        return None
    try:
        start = ast.literal_eval(call.args[0])
        stop = ast.literal_eval(call.args[1])
    except ValueError:
        return None
    if isinstance(start, int) and isinstance(stop, int):
        return (start, stop)
    return None


def _baseline_engine_ids(constants: dict[str, tuple[str, ...] | int]) -> frozenset[str]:
    sql, rolling = constants["SQL_CASES"], constants["ROLLING_CASES"]
    stream = constants.get("STREAM_CASES", rolling)
    join_cap = constants.get("STREAM_JOIN_MAX_ROWS")
    cap = join_cap if type(join_cap) is int else None
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
        if _measured_stream_join(backend, scenario, int(rows), cap)
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
