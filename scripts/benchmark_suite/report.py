"""Complete Markdown tables and explicitly scoped regression decisions."""

from __future__ import annotations

import math
import statistics
from collections import Counter

from scripts.benchmark_suite.catalog import CAPABILITIES, SQL_CASES, STREAM_SCOPE

THRESHOLD_PERCENT = 5.0
ROUNDS = 2
SAMPLES = 10


def validate_shards(expected: list[str], received: list[str]) -> None:
    if Counter(expected) != Counter(received):
        missing = sorted((Counter(expected) - Counter(received)).elements())
        extra = sorted((Counter(received) - Counter(expected)).elements())
        raise ValueError(
            f"benchmark shards missing={missing}, duplicate/unexpected={extra}"
        )


def checked_samples(rounds: list[list[float]], *, minimum: int = 1) -> list[float]:
    if len(rounds) != ROUNDS or any(len(values) < minimum for values in rounds):
        raise ValueError(
            f"expected two complete rounds with at least {minimum} samples"
        )
    samples = [value for values in rounds for value in values]
    _validate_observations(samples)
    return samples


def _validate_observations(samples: list[float]) -> None:
    if any(type(value) not in (int, float) for value in samples):
        raise ValueError("benchmark samples must be numeric")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("benchmark samples must be finite and positive")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _verdict(kind: str, changes: list[float]) -> str:
    # Protect the exact +5% endpoint from ratio/subtraction roundoff.
    threshold = THRESHOLD_PERCENT + 1e-12
    slow = [change > threshold for change in changes]
    if kind == "suite-blocks":
        return "informational-slowdown" if all(slow) else "informational"
    if all(slow):
        return "regression"
    if any(slow):
        return "inconclusive"
    if all(change < -threshold for change in changes):
        return "improved"
    return "no-confirmed-regression"


def _head_statistics(case: dict, head: list[float]) -> dict:
    median = statistics.median(head)
    return {
        "head_p50": median,
        "head_p95": _percentile(head, 0.95),
        "head_min": min(head),
        "head_max": max(head),
        "rows_per_second": case["rows"] / median if case.get("rows") else None,
        "samples": len(head),
        "base_p50": None,
        "change_percent": None,
        "round_changes": [],
    }


def comparison(case: dict) -> dict:
    if case.get("correctness") is not True:
        raise ValueError(f"{case['id']}: correctness was not established")
    kind = case["comparison"]
    minimum = SAMPLES if kind == "interleaved" else 1
    head = checked_samples(case["candidate"], minimum=minimum)
    result = _head_statistics(case, head)
    if not case["baseline"]:
        labels = {"external": "external-reference", "new": "new-coverage"}
        if kind not in labels:
            raise ValueError("missing baseline is not an implicit performance pass")
        return _checked_statistics({**result, "verdict": labels[kind]})
    if kind not in ("interleaved", "suite-blocks"):
        raise ValueError("unexpected historical comparison kind")
    return _historical_statistics(case, result, minimum)


def _historical_statistics(case: dict, result: dict, minimum: int) -> dict:
    kind = case["comparison"]
    base = checked_samples(case["baseline"], minimum=minimum)
    if kind == "interleaved" and [len(r) for r in case["baseline"]] != [
        len(r) for r in case["candidate"]
    ]:
        raise ValueError("paired sample counts differ")
    changes = [
        100 * (min(right) / min(left) - 1)
        for left, right in zip(case["baseline"], case["candidate"], strict=True)
    ]
    return _checked_statistics(
        {
            **result,
            "base_p50": statistics.median(base),
            "change_percent": 100 * (result["head_p50"] / statistics.median(base) - 1),
            "round_changes": changes,
            "verdict": _verdict(kind, changes),
        }
    )


def _checked_statistics(result: dict) -> dict:
    values = [value for value in result.values() if type(value) in (int, float)]
    if any(not math.isfinite(value) for value in [*values, *result["round_changes"]]):
        raise ValueError("derived benchmark statistics are nonfinite")
    return result


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def table(headers: list[str], rows: list[list[object]]) -> str:
    cells = [[_cell(value) for value in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) + " |"
        for row in cells
    ]
    lines.insert(1, "|" + "|".join("-" * (width + 2) for width in widths) + "|")
    return "\n".join(lines)


def _milliseconds(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.6f}"


def _result_row(case: dict) -> list[object]:
    dimensions = f"N={case.get('rows', '—')}"
    if "history_rows" in case:
        dimensions += f", H0={case['history_rows']}"
    prefix = [case["id"], dimensions, case["scope"]]
    if case["status"] != "ok":
        return [
            *prefix,
            *("—" for _ in range(6)),
            "error",
        ]
    result = comparison(case)
    changes = ", ".join(f"{value:+.2f}%" for value in result["round_changes"]) or "—"
    change = result["change_percent"]
    throughput = result["rows_per_second"]
    return [
        *prefix,
        _milliseconds(result["base_p50"]),
        _milliseconds(result["head_p50"]),
        _milliseconds(result["head_p95"]),
        f"{throughput:,.0f}" if throughput is not None else "—",
        f"{change:+.2f}%" if change is not None else "—",
        changes,
        result["verdict"],
    ]


def render_report(cases: list[dict], errors: list[str]) -> str:
    rows = []
    metrics = []
    failures = list(errors)
    for case in sorted(cases, key=lambda item: item["id"]):
        target = metrics if case.get("kind") == "metric" else rows
        formatted, problems = _format_case(case)
        target.append(formatted)
        failures.extend(problems)
    header = [
        "Case",
        "Dimensions",
        "Timed scope",
        "Base P50 ms",
        "Head P50 ms",
        "P95 ms",
        "Rows/s",
        "Change",
        "Round min changes",
        "Result",
    ]
    parts = [
        "# Complete benchmark results",
        "",
        f"{len(rows) + len(metrics)} result rows; positive changes mean slower. "
        "No top-N filtering.",
        "",
        "Historical engine/warm gates use two rounds of ten interleaved samples; "
        "both round minima must exceed +5% to fail. This is a repeat-confirmed "
        "best-of-N gate, not a confidence interval or a claim of equivalence.",
        "Existing suite-block comparisons and cross-library references "
        "are informational.",
        "",
        table(header, rows),
    ]
    if metrics:
        parts.extend(
            [
                "",
                "## Allocation metrics",
                "",
                table(["Case", "Metric", "Base", "Head", "Delta", "Result"], metrics),
            ]
        )
    cross_library = _cross_library_table(cases)
    if cross_library:
        parts.extend(
            [
                "",
                "## Cross-library comparison",
                "",
                "Head P50 milliseconds, same input and materialized Arrow output. "
                "Native streaming starts with empty rolling state on a ready runner "
                "and excludes runner startup and shutdown; enqueue, tasks/channels, "
                "rolling, watermarks and Arrow materialization remain timed. "
                "Other columns "
                "use their declared execute-to-Arrow boundaries. This is not a "
                "kernel-only ranking. Ten-row SMA cases have no full-window outputs.",
                "",
                cross_library,
            ]
        )
    if failures:
        parts.extend(
            [
                "",
                "## Incomplete or invalid evidence",
                "",
                *[f"- {_cell(e)}" for e in dict.fromkeys(failures)],
            ]
        )
    return "\n".join(parts) + "\n"


def _format_case(case: dict) -> tuple[list[object], list[str]]:
    formatter = _metric_row if case.get("kind") == "metric" else _result_row
    failures = []
    if case["status"] != "ok":
        failures.append(f"{case['id']}: {case.get('error', 'failed')}")
    try:
        return formatter(case), failures
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{case.get('id')}: {error}")
        return formatter({**case, "status": "error", "error": str(error)}), failures


def _metric_row(case: dict) -> list[object]:
    if case["status"] != "ok":
        return [case["id"], case.get("metric", "unknown"), "—", "—", "—", "error"]
    base, head = case["baseline_value"], case["candidate_value"]
    validate_metric(case)
    return [
        case["id"],
        case["metric"],
        f"{base:g}",
        f"{head:g}",
        f"{head - base:+g}",
        "informational",
    ]


def validate_metric(case: dict) -> None:
    if case.get("correctness") is not True:
        raise ValueError("allocation correctness was not established")
    for value in (case["baseline_value"], case["candidate_value"]):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid allocation metric")


def _reference_cell(case: dict | None) -> str:
    if case is None:
        return "missing"
    if case["status"] != "ok":
        return "error"
    if case.get("backend") == "calc-flow-stream" and case.get("scope") != STREAM_SCOPE:
        return "invalid scope"
    try:
        return _milliseconds(comparison(case)["head_p50"])
    except (KeyError, TypeError, ValueError):
        return "invalid"


def _cross_library_table(cases: list[dict]) -> str:
    selected = [case for case in cases if case.get("family") == "engines"]
    if not selected:
        return ""
    index = {
        (case["rows"], case["scenario"], case["backend"]): case for case in selected
    }
    rows = [
        _cross_library_row(size, scenario, index)
        for size in sorted({case["rows"] for case in selected})
        for scenario in SQL_CASES
    ]
    return table(
        [
            "Rows",
            "Scenario",
            "Native stream (ready)",
            "Calc Flow SQL",
            "DataFusion",
            "Polars",
            "TA-Lib",
        ],
        rows,
    )


def _cross_library_row(size: int, scenario: str, index: dict) -> list[object]:
    backends = ("calc-flow-stream", "calc-flow-sql", "datafusion", "polars", "ta-lib")
    return [
        size,
        scenario,
        *(
            "unsupported"
            if scenario not in CAPABILITIES[backend]
            else _reference_cell(index.get((size, scenario, backend)))
            for backend in backends
        ),
    ]
