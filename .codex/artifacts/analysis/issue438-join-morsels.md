# Issue 438: bounded Join probe and gather

Baseline: `main@40cddf67cdddbc49f9febb74f6c5463f08251b6b`, fetched at task
start. Prerequisites #429, #431 and #436 are merged. Branch:
`feature/dal-345-join-morsels`. No shared identity or checkpoint layout change.

## Measurement scope fixed before implementation

First measure the existing observed parallel-work submission, result installation
and cleanup path: 100 warm empty submissions at one unit and at four units,
using a four-worker isolated instance of the existing service. Compare the
median to the proposed 50 microsecond boundary. This is a development-profile
dispatch diagnostic, not a sealed release performance claim. Keep the existing
pool unless evidence warrants a separate implementation; do not add Rayon first.

After implementation, compare the existing split path with fused morsels in
the same candidate test executable, using two rounds of ten adjacent alternating
AB/BA pairs for the 64k fanout case and a 1,024-row fallback control. Keep full
output oracles outside timing, and collect the maintained PhaseClock wall,
process CPU and per-thread scheduler runtime. The baseline switch only selects
the existing split implementation; these observations do not constitute exact
main/candidate sealed release acceptance. All five cases (two empty unit counts,
two paired workloads and one projected 64k workload) share a 600-second collection
budget including fixtures, warmup, oracle and cleanup; compilation is separate.
No full-suite or R16 claim is planned.

The declared CPU budget is the actor plus at most four native workers (five CPU
seconds per wall second). The 1,024-row control is serial and retains the
existing two-CPU-seconds-per-wall-second diagnostic rejection rule. A result
outside these budgets cannot support acceptance. Process-clock quantization is
reported rather than treated as zero CPU work.

## Implementation boundary

Pre-implementation empty-task result (100 warm samples each): one unit median
45.185 microseconds / p95 81.509; four units median 87.893 / p95 156.437.
The four-unit roundtrip exceeds 50 microseconds, while one unit's median is
below it. Keep the existing pool and 320,000 estimated-run-visit gate; this does
not justify parallelizing thin one-to-one or small batches. No second pool is
introduced. Fixed extra funding caps: 512 MiB scratch per batch and 16 MiB
result funding per unit (at most 64 MiB total). These are optional acceleration
ceilings inside the unchanged runtime pool, not larger runtime budgets.
The process clock has 10ms resolution (one-unit aggregate quantized to zero);
surviving per-thread scheduler runtime was also recorded. Raw observations and
thread identities are in the [resource receipt](issue438-join-morsels-resources.json).

Use deterministic contiguous incoming ranges, at most four units. Keep the
existing small-input and estimated-probe-cost gates. Count/limit precedes state
preparation; complete state and expiration admission precede fused output scratch
and result admission. Each worker then probes, preflights and gathers its range.
Every range settles before the first emission. Preserve sequential publication
after successful output and funded owners through cancellation and cleanup.

Unsupported output shapes and optional-credit refusals retain the split or
serial implementation. ASOF and public bindings remain unchanged.

The deferred snapshot's control reservation is carved out of the existing pair
allowance, with at least the worst-case three-usize-per-match selection storage
left separately funded. Otherwise deferral declines. The split and deferred
paths reserve exactly the same total at the end of state/expiration preparation;
the dedicated finite-pool ledger regression verifies that equality and complete
refund. The pending dictionary already shares its key/ID scratch reservation.

## Verified source and checks

Final Rust source: `e5987c9cb1ebe826834a980178451481fde8d751`.
`f49f89ef6f954d38bb2dacf03a322bdf1f4fb1a7` introduces the kernel; the final
source commit adds the state-admission budget fix and its ledger regression.
The subsequent results commit changes only this report and the JSON receipt.

All Rust builds used Rust 1.88.0, `CARGO_BUILD_JOBS=2`,
`CARGO_PROFILE_DEV_DEBUG=0` and `CARGO_PROFILE_TEST_DEBUG=0`; Cargo.lock was
unchanged. Exact source, lockfile and executable hashes are in the receipt.

Actual focused commands and results on final Rust source:

```bash
cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic \
  --lib test_morsel_deferred_snapshot_keeps_split_state_admission_budget \
  -- --test-threads=1
target/debug/deps/calc_flow-abab16b2e36df81b operator::join:: --test-threads=1
cargo test --locked -p calc-flow --test stream_join_properties \
  --test stream_join_state --test stream_join_spec -- --test-threads=1
cargo clippy --locked -p calc-flow --lib --tests --benches --all-features -- -D warnings
cargo fmt --all --check
git diff HEAD --check
```

- Dedicated budget ledger: one passed. Full Join module on the same executable:
  **396 passed**, two ignored (the explicit paired diagnostic and existing broad
  4M-row experiment). Property/state/spec integration: **20 passed**.
- **30 gather lifecycle tests passed**, including generation retirement,
  abandonment, deadline, launch/join failures and real ownership refunds. This
  evidence is reused: pool code, these tests and their dependencies were
  unchanged by the later Join-only fixes. The empty diagnostic was run explicitly.
- Final Clippy, format and whitespace checks passed. Lizard 1.23.0 receipt:
  **90 changed/new functions, maximum CCN 8**, with changed Rust file hashes.
- New tests verify worker-local gather before settlement, uneven sliced input
  ranges on both sides, literal tie/row order, retained projection, checkpoint
  restore, input immutability, match/state/row/sequence/counter preflight,
  scratch/result/worker refusal, cancellation/deadline/abandonment/reset,
  worker panic, and output funding through a blocked/failing collector.

Before implementation the new worker-gather regression observed no fill units
instead of four: the old cost estimate counted distinct keys only. Correcting
that estimate exposed the separate fill/gather path. The final regression
requires actual column take work on each probe worker before it settles.

Full workspace, cross-platform and total 90% coverage checks remain CI gates.
No binding, schema, dependency or checkpoint format changed.

## Paired development diagnostic

Executed the same final-source test executable with
`test_morsel_paired_resources --ignored --nocapture --test-threads=1`.
The foreground process had a 595-second timeout inside the 600-second total
collection allowance and finished in **137.61 seconds**, including fixtures,
all row oracles and owned cleanup. The earlier empty diagnostic took 0.05 seconds.
No samples were retried or added. Builds were separate.

Each case has two rounds of ten alternating AB/BA pairs: **40 observations per
case, 120 total**, all with complete row oracles outside handler timing. The
64k cases emit 512k matches against eight unordered opposite rows with ties;
the incoming batch has two sliced records. Full output has six columns;
projection selects right status and narrows retained left payload. The 1,024-row
control stays on the serial path in both modes. The switch selects the existing
split path inside the same candidate, including common cost-gate/range fixes;
it does not represent unmodified main. Timing includes fresh-job native worker
startup; it excludes dimension preload and output-oracle time. No CPU affinity
reservation was made; observed affinity was CPUs 0–31.

Medians across both rounds, milliseconds:

| Case                 | Mode  | Wall     | Process CPU | All thread runtime | Native thread runtime |
| -------------------- | ----- | -------- | ----------- | ------------------ | --------------------- |
| 64k full             | split | 1,945.98 | 2,100       | 2,095.25           | 197.53                |
| 64k full             | fused | 1,071.06 | 2,520       | 2,514.29           | 1,945.58              |
| 64k projected        | split | 1,950.31 | 2,110       | 2,106.17           | 205.56                |
| 64k projected        | fused | 1,087.75 | 2,545       | 2,544.73           | 1,966.71              |
| 1,024 serial control | split | 18.78    | 20          | 19.46              | 0.04                  |
| 1,024 serial control | fused | 18.71    | 20          | 19.09              | 0.02                  |

Fused handler process CPU is about **20% higher** for full output and **21%
higher** for projection. The lower wall observations do not erase that cost.
Aggregate process CPU/wall ratios are 2.35/2.34 for fused full/projected, below
the predeclared five-CPU budget; split is 1.08. Serial controls are 0.97–1.04,
below two. Per-thread totals corroborate these ratios. Every individual sample
also stays within its declared CPU budget; no threshold was relaxed.

These are unoptimized development observations with a 10ms process CPU clock,
not sealed release main/candidate speedup, equivalence or final acceptance.
The serial control isolates dispatch eligibility inside this candidate and
does not establish a small-batch regression verdict against unmodified main.
The 50µs full callback target, release crossover and all eight formal targets
remain unmeasured here and belong to #443/R16. Raw per-round samples, actual
thread identities/runtime, CPU budget checks and build provenance are retained
in the resource receipt.
