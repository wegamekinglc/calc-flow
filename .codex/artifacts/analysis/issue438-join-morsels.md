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
surviving per-thread scheduler runtime was also recorded. Raw observations:
`target/issue438-roundtrip.log` (runtime-local verification log).

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
