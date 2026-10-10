# Stream Join probe cost gate and arena verdict evidence

Branch `fix/stream-join-probe-cost-gate` versus main `82d44293`. All
timings are paired AB/BA between sealed release wheels, one fresh worker
per sample, with the suite's full-row oracle asserted on every sample,
on the shared WSL2 host (absolute values carry window noise; paired
deltas are the claims).

## Contents of this branch

1. **Parallel-probe cost gate.** The owned parallel probe engaged at
   8,192+ admitted rows regardless of shape; two gather submissions per
   batch cost ~20 ms in a live runtime while a one-to-one lookup needs
   ~8 ms of serial probe per 64k rows, regressing the flagship suite
   case by +209% at 1M (158 → 497 ms, CI [+209%, ...] two rounds).
   Capture now sums the resolved key slots' opposite run lengths and
   declines to the serial native path below 320,000 estimated visits.
   Result: 439.1 → 156.7 ms p50 at 1M (−62.8%), restoring the
   pre-regression level.
2. **Arena layout algebra (J-P1).** `ArenaLayout::measure` stopped
   walking every row cell: fixed-width key columns size in O(1), string
   columns run one tight length loop, and charge/bytes/row layout are
   derived algebraically from the same per-cell formula. Charge stays
   byte-identical (funding/allocation contract tests green).
3. **Typed-primary experiment, measured and rejected.** The borrowed-key
   typed route (streaming hash, typed equals, no frame materialization)
   was promoted behind a null-safe gate with all four path-selection
   pins rewritten. Correctness held: every key type matched SQL in both
   directions, a new row-by-row framing test proved byte-identical
   frames, 209 join tests green. Two independent measurements rejected
   it: same-window 100k A/B typed 25.06 ms vs arena 16.31 ms (typed
   +32.9% slower), and the post-flip profile — `BorrowedKey::hash` 19.8%
   + `equals` 9.1% of all samples versus 14.6% for the arena's one-pass
   `push_bytes`. The arena stays primary; the typed route remains the
   null-bearing fallback; the framing-equivalence test pins both to the
   canonical V1 encoder.

## Final measurements

| Comparison | Case | Result |
|---|---|---|
| main 82d44293 vs branch | join 1M, 8 pairs | 321.2 → 216.3 ms p50, −32.7% (main is the regressed head; branch restores and adds J-P1) |
| branch arena vs typed (same window) | join 100k, 6 pairs | 16.31 vs 25.06 ms, arena −32.9% |
| pre-regression c7cf6ae6 vs branch | join 1M | level within window noise (branch = fix + J-P1; the regression, not serial speed, was the delta) |

Post-optimization profile (4 kHz userspace sampler, 30 timed samples,
ELF-symbolized): `process_data` is 70.7% of samples; top operator leaves
`append_granted_rows` 4.2%, take kernels 4.9%, `ArenaLayout::measure`
2.8% (from 5.1%), `window_by_id` 1.3%, payload drops 2.0%. No single
interning kernel dominates the probe any more.

## Known diagnostic

`cargo clippy --tests` reports one span-less `large_stack_arrays` error
on this branch. Bisection: it appears exactly when the slim-probe
regression test is present — in any module, minimized to three asserts,
with the hook removed or present — while main carries the identical
8,192-row batch pattern elsewhere cleanly, and `--lib` alone is clean.
Recorded for CI adjudication; not silenced with an allow.
