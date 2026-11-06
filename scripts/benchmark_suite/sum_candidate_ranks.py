"""One frozen candidate/rank/slot study using the maintained SUM collector."""

from __future__ import annotations

from dataclasses import replace

from scripts.benchmark_suite import sum_columnar_keys as shared

SLUG = "sum-candidate-ranks-20261106"
BASELINE = "1fcafde11a247b1be275d3cb6ab41a59476d6832"
CASES = (
    {**shared.CASES[0], "reason": "dense candidate/rank/slot target"},
    {**shared.CASES[1], "reason": "same dense container path with four physical cores"},
    {**shared.CASES[2], "reason": "few touched slots over10k retained groups"},
    {
        **shared.CASES[3],
        "reason": "all-distinct within batch container overhead control",
    },
    {
        **shared.CASES[4],
        "id": "repeat2-stream-1cpu",
        "reason": "N2G low-repetition control, not an enablement threshold",
    },
)
CONTROLS = {
    name: {**value, "selector": "cardinality control; no columnar selector"}
    for name, value in shared.CONTROL_FIXTURES.items()
}
CONTRACT = replace(
    shared.COLUMNAR,
    slug=SLUG,
    baseline=BASELINE,
    cases=CASES,
    controls=CONTROLS,
    tools=(
        *shared.TOOL_FILES,
        "scripts/benchmark_suite/sum_candidate_ranks.py",
        "scripts/test_sum_candidate_ranks.py",
    ),
    report_lines=(
        "Five fixed candidate/rank/slot cases use200formal observations and20warmups",
        "under one600-second budget including preflight, fixture/oracle setup,",
        "profiles/decoding, resources/statistics, publication/postseals and cleanup.",
        "Builds are separate. No cases, samples or favorable retries are added.",
        "At least one dense target needs the exact maintained improved verdict,",
        "with no confirmed regression in any case and all evidence complete.",
        "Inconclusive is uncertainty, not equivalence or a confirmed speedup.",
        "The freshly compiled baseline is public1fcafde1 including Arrow59.3.2;",
        "older22f/P74/columnar experiment products are not compatible baselines.",
        "Both frozen build trees use common657 instrumentation from public841.",
        "DF target/input partitions remain1, min_rows_per_partition1, checkpoints off.",
        "Dense1/controls use CPU0/Tokio1; dense4 uses CPUs0/2/4/6 andTokio4.",
        "The controller uses CPU31; external helper pools stay1. WSL host load",
        "and power are unobservable; resource fingerprints and guards are retained.",
        "Dense has10k zero seed rows plus20x100k source rows, filtered to80k trades",
        "per business batch. Sparse has10k seeds plus20x100 rows/80trades each.",
        "Distinct preserves10k seeds and20x10k all-trade business rows (one/key).",
        "Repeat2 preserves10k seeds and20x20k all-trade rows (two/key,N2G).",
        "The old edge IPC recipe is reused as low-repetition control, with no",
        "columnar route or new enablement threshold implied by these cases.",
        "Every native sample retains21 complete value/schema/hash-checked snapshots.",
        "Peak process RSS includes inputs and retained outputs, not operator heap.",
        "Common657 inputs are nonnull. UInt64/nullable/generic, finite, checkpoint",
        "and tight-budget latency remain unmeasured; focused tests are separate proof.",
        "Both dense1 profiles require five processing windows after excluding each",
        "first job. All timestamps and loss/unwind diagnostics are preserved.",
        "Full stacks and maintained candidate/hash/drop targets describe CPU samples.",
        "Inclusive shares overlap and are not wall-time phases. Missing/inlined",
        "frames do not establish zero cost or a separately verified runtime path hit.",
        "[Summary](summary.json), [all formal samples](samples.csv),",
        "[complete raw evidence](complete-evidence.json.gz),",
        "[profiles](profile-summary.json), [targets](profile-targets.json) and",
        "[manifest](evidence-manifest.json) retain all receipts, intervals and hashes.",
    ),
)


def validate_protocol(protocol):
    return shared.validate_protocol(protocol, CONTRACT)


def main():
    shared.main(contract=CONTRACT)


if __name__ == "__main__":
    main()
