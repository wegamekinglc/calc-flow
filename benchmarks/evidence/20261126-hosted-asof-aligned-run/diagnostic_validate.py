"""Validate native finalize clocks and output counters from a single public request."""

from __future__ import annotations

FIELDS = (
    "matching_ns",
    "arrow_output_ns",
    "left_commit_ns",
    "right_eviction_ns",
    "emit_ns",
    "residual_ns",
)
TICKETS = "admitted_gather_tickets_match_arrow_left_right_emit_residual"


def counts(recs, pid):
    if not recs or len({r["operator"] for r in recs}) != 1:
        raise ValueError("expected exactly one instrumented ASOF operator")
    last = {"emitted_left_rows": 0, "matched_rows": 0, "evicted_right_rows": 0}
    parts = (
        "arrow_bookkeeping_ns",
        "arrow_charge_ns",
        "arrow_columns_ns",
        "arrow_assembly_ns",
    )
    for r in recs:
        if (
            any(type(r[k]) is not int or r[k] < 0 for k in parts)
            or sum(r[k] for k in parts) != r["arrow_output_ns"]
        ):
            raise ValueError("Arrow subphases do not close")
        if (
            r["schema"] != "calc-flow.asof-watermark-exclusive.v1"
            or r["pid"] != pid
            or r["success"] is not True
        ):
            raise ValueError("wrong process, trace schema or failed finalize callback")
        times = [r[k] for k in FIELDS]
        if (
            any(type(n) is not int or n < 0 for n in times)
            or sum(times) != r["wall_ns"]
        ):
            raise ValueError("exclusive callback times do not close")
        tickets = r[TICKETS]
        if len(tickets) != len(FIELDS) or any(
            type(n) is not int or n < 0 for n in tickets
        ):
            raise ValueError("invalid gather-service ticket inventory")
        for delta, gauge in (
            ("emitted_delta", "emitted_left_rows"),
            ("matched_delta", "matched_rows"),
            ("evicted_delta", "evicted_right_rows"),
        ):
            diff = r["after"][gauge] - r["before"][gauge]
            if diff != r[delta] or diff < 0 or r["before"][gauge] != last[gauge]:
                raise ValueError("noncontiguous ASOF counter across callbacks")
            last[gauge] = r["after"][gauge]
    if any(last[k] != 10_000_000 for k in ("emitted_left_rows", "matched_rows")):
        raise ValueError("ASOF trace did not match and emit every expected row")
    if not 0 <= last["evicted_right_rows"] <= 10_000_000:
        raise ValueError("evictions exceed admitted reference rows")
    groups = {
        "all": recs,
        "emitting": [r for r in recs if r["emitted_delta"]],
        "other": [r for r in recs if not r["emitted_delta"]],
    }

    def aggregate(rs):
        return {
            **{
                k: sum(r[k] for r in rs)
                for k in (
                    "wall_ns",
                    *FIELDS,
                    *parts,
                    "emitted_delta",
                    "matched_delta",
                    "evicted_delta",
                )
            },
            "callbacks": len(rs),
            "tickets": [sum(r[TICKETS][i] for r in rs) for i in range(len(FIELDS))],
        }

    return {k: aggregate(rs) for k, rs in groups.items()}
