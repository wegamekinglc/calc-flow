"""Verify the causal status and entire output of each public static Join run."""

from __future__ import annotations

CASE_ID = "engines/1000000/calc-flow-stream-1t/join"
ROWS = 1_000_000


def validate_static_sample(sample: dict) -> None:
    expected = {"passed": True, "rows": ROWS, "finite_rows": ROWS, "max_abs_error": 0.0}
    if sample["correctness"] != expected:
        raise ValueError("static Join expected one million exact finite products")
    evidence = sample["stream_evidence"]
    status = evidence["lookup_join"]
    if status["emitted_match_rows"] != ROWS:
        raise ValueError("post-quote status precedes the complete result")
    if status["state_limit_failures"] or status["match_limit_failures"]:
        raise ValueError("static Join reported a limit failure")
    if status["left"]["retained_rows"] or status["left"]["evicted_rows"]:
        raise ValueError("sealed Join retained or evicted a quote")
    accepted = status["right"]["watermark_micros"]
    if accepted is None or accepted < evidence["sealed_dimension_watermark_micros"]:
        raise ValueError("right dimension was not sealed")

    operator = evidence["join_operator"]
    if (operator["input_rows"], operator["fanned_out_rows"], operator["errors"]) != (
        ROWS + 64,
        ROWS,
        0,
    ):
        raise ValueError("incomplete final Join operator counters")
    if not 16 * ROWS <= operator["fanned_out_bytes"] < 24 * ROWS:
        raise ValueError("Join emitted a wider intermediate than sequence and product")
