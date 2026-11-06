"""One fixed CASE projection study using the maintained paired collector."""

from __future__ import annotations

from dataclasses import replace

from scripts.benchmark_suite import sum_columnar_keys as shared

SLUG = "sum-case-projection-20261106"
BASELINE = "dc8393d2f33771706b32492e9c4fa5436c6baf12"
CASES = (
    {**shared.CASES[0], "reason": "dense stock CASE with mixed buy/sell masks"},
    {**shared.CASES[1], "reason": "same dense CASE path with four physical cores"},
    {
        **shared.CASES[2],
        "reason": "sparse updates with alternating uniform buy/sell business batches",
    },
    *(
        {
            **shared.CASES[3],
            "id": f"uniform-{side}-stream-1cpu",
            "fixture": f"uniform-{side}",
            "reason": f"pure {side} CASE mask on seed and every business row",
        }
        for side in ("buy", "sell")
    ),
)
CONTROLS = {
    f"uniform-{side}": {
        **shared.CONTROL_FIXTURES["distinct"],
        "side": side,
        "selector": f"pure uniform {side} CASE mask; no admission threshold",
    }
    for side in ("buy", "sell")
}
CONTRACT = replace(
    shared.COLUMNAR,
    slug=SLUG,
    baseline=BASELINE,
    cases=CASES,
    controls=CONTROLS,
    tools=(
        *shared.TOOL_FILES,
        "scripts/benchmark_suite/sum_case_projection.py",
        "scripts/test_sum_case_projection.py",
    ),
    report_lines=(
        "Five fixed cases use200formal observations and20warmups in two rounds",
        "of ten alternating pairs per case. Both dense1 profiles use six jobs",
        "per product, excluding the first; five validated processing windows",
        "are required. No case, sample, favorable retry or threshold is added.",
        "One600-second budget includes controller setup and live global quiet",
        "guard, fixtures, preflight, warmups, startup, full oracles, resources,",
        "profiles/capture/decoding, statistics, publication, postseals and cleanup.",
        "Builds are separate. At least one dense target needs the exact maintained",
        "improved verdict, with complete evidence and no confirmed regression on",
        "any case. Inconclusive is not equivalence, speedup or confirmed regression.",
        "The fresh baseline is publicdc8393d2. Old1fc and experimental rankedP",
        "products are incompatible; compatible dependency caches do not replace",
        "fresh core/example compilation or public source and binary proof.",
        "Both products use identical common657 instrumentation from public841.",
        "Dense has10k zero seed rows plus20x100k source rows, selecting80k trades",
        "per business batch. Sparse has10k seed groups plus20x100 source rows,",
        "selecting80 trades/batch; those business batches alternate uniform sides.",
        "Uniform-buy and uniform-sell reuse the maintained one-cycle all-trade",
        "recipe:10k seeds plus20x10k business rows. Each forces side on the seed",
        "AND all business records, preserving quantity, other column values/types,",
        "exact side Field, schema metadata and caller-owned input buffers.",
        "The two pure masks are separately paired, so one branch cannot hide the",
        "other's regression in an alternating workload. Both retain21 complete",
        "independent arithmetic/schema/input/snapshot checks per job.",
        "DF target/input partitions remain1, min_rows_per_partition1, batch8192,",
        "checkpoints off. Dense4 uses CPUs0/2/4/6 withTokio4; others CPU0/Tokio1.",
        "The controller uses CPU31 and external helper pools stay1. Host load",
        "and power outside the WSL guest remain unobservable.",
        "The timer starts after IPC preload and covers graph planning, execution,",
        "retained full outputs, natural end and cleanup. Oracles are outside that",
        "timer but inside the total budget. Whole-process peak RSS includes input",
        "preload and retained outputs, not operator-only heap or reservation fees.",
        "Full CPU stacks and exact clock/loss evidence are retained. Inclusive",
        "shares overlap and are not phase wall times; missing/inlined frames or",
        "zero matched samples do not establish zero runtime cost or a route hit.",
        "Mixed CASE with an all-trade filter, finite, nullable/UInt64/generic/UDF,",
        "checkpoint and tight-budget latency remain unmeasured. Focused source",
        "semantics and allocation tests are separate from these timings.",
        "[Summary](summary.json), [formal samples](samples.csv),",
        "[complete raw evidence](complete-evidence.json.gz),",
        "[profiles](profile-summary.json), [targets](profile-targets.json) and",
        "[manifest](evidence-manifest.json) retain receipts, intervals and seals.",
    ),
)


def validate_protocol(protocol):
    return shared.validate_protocol(protocol, CONTRACT)


def main():
    shared.main(contract=CONTRACT)


if __name__ == "__main__":
    main()
