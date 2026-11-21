# Checkpoint v4 measurement protocol

Status: registered on 2026-11-16 for P3.1. This document is a pre-registration,
not performance evidence. No build or measurement was run for this registration.
P3.7 runs the four cases after P2.7 and the P3 checkpoint runner exist. A narrow
`scripts/profile_checkpoint_v4.py` driver is part of that later measurement step;
this step deliberately does not add an executable placeholder or production
checkpoint stubs.

## Reused maintained components

- `scripts/profile_warm_stream.py` supplies paired fresh-process execution,
  build/binary identity checks, budget stopping, seeded bootstrap summaries and
  Markdown/JSON reporting. Its current `native` subcommand is **not** an M1/M2
  substitute: its two arms are recorder absent/present, whereas P3 compares
  checkpoint on/off. The P3 driver imports the paired utilities without changing
  P2.1's workloads or arms.
- The P2.1 `1x1-dual` and `6400x64-dual` fixtures supply deterministic 64-bit
  prices, timestamps, `difference(mean(5), mean(20))` and the numeric oracle.
  P3 adds the exact column projection and checkpoint-on/off arms without changing
  rolling arithmetic or the P2.1 registration.
- `benchmarks/engine_stream.py` supplies replayable sources, ordinary sinks and
  managed-runner ownership. `benchmarks/checkpoint_cycles.py` supplies owned
  cleanup and checkpoint-evidence checks. Its finite `run_cycle` is not an M3/M4
  substitute because terminal-checkpoint-only timing is forbidden.
- `benchmarks/sustained_join.py` provides the cutoff/settlement pattern only;
  its Join workload and oracle are not reused for Rolling.

The driver emits raw per-round samples below this directory and refuses to
overwrite existing evidence. Builds, host suitability and build times are
reported separately. Timed runs use release binaries, `TOKIO_WORKER_THREADS=4`,
one helper thread per BLAS pool, and no overlapping owned build or test.

## Fixed cases and budgets

The four allocations sum to **600 s**, including fixture creation, startup,
warm-up, coverage validation, Arrow/oracle checks, restore checks, process
startup and cleanup. They are per-case ceilings, not permission to exceed the
combined budget. Exhaustion, an unsuitable host or missing coverage stops the
run; there are no added cases, samples or retries.

- **M1 — 40 s.** Direct sync Rolling `1x1-dual`, exact projection and a
  4,096-row warm history; checkpoint on with no epoch versus off. Two rounds of
  ten adjacent AB/BA pairs; each arm observation is the median of 30 dispatches.
  Gate: median paired `on/off - 1` ≤ 0.03.
- **M2 — 50 s.** M1 at `6400x64-dual`, with the recorder absent in both arms.
  Two rounds of ten adjacent AB/BA pairs; each arm observation is the median of
  30 dispatches. Gate: median paired `on/off - 1` ≤ 0.03.
- **M3 — 250 s.** Replayable-source Rolling runner with exact projection, 64
  entities and 6,400-row batches; 1 s checkpoints versus off. Two rounds of ten
  AB/BA pairs, one warm-up per arm/round and 5 s measured per arm. Gate: median
  paired `on_rows_s/off_rows_s` ≥ 0.90.
- **M4 — 260 s.** M3 with a 100 ms checkpoint interval and the same fixed
  counts. Gate: median paired `on_rows_s/off_rows_s` ≥ 0.75.

M3 and M4 each require 200 s of measured arm time; their remaining allocations
cover the fixed warm-ups and all non-timed validation and cleanup. M1 uses the
same recorder configuration in both arms and does not add a recorder comparison.
M2 explicitly disables the recorder in both arms.

## Exact P3.7 commands

Run from the repository root after sealing the release evidence binary at
`target/checkpoint-v4/checkpoint-v4-evidence`. These commands define the later
driver's CLI and are not claimed to run at P3.1. Each report embeds the source
commit, binary SHA-256, host and environment identity. Raw round files go under
the corresponding `benchmarks/checkpoint-v4/m*-raw/` directory.

```bash
TOKIO_WORKER_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/profile_checkpoint_v4.py direct \
  --binary target/checkpoint-v4/checkpoint-v4-evidence \
  --case 1x1-dual --history-rows 4096 --rounds 2 --pairs 10 --dispatches 30 \
  --budget-seconds 40 --work-dir benchmarks/checkpoint-v4/m1-raw \
  --output benchmarks/checkpoint-v4/m1-p3.7.json

TOKIO_WORKER_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/profile_checkpoint_v4.py direct \
  --binary target/checkpoint-v4/checkpoint-v4-evidence \
  --case 6400x64-dual --history-rows 4096 --recorder absent \
  --rounds 2 --pairs 10 --dispatches 30 --budget-seconds 50 \
  --work-dir benchmarks/checkpoint-v4/m2-raw \
  --output benchmarks/checkpoint-v4/m2-p3.7.json

TOKIO_WORKER_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/profile_checkpoint_v4.py runner \
  --binary target/checkpoint-v4/checkpoint-v4-evidence \
  --entities 64 --batch-rows 6400 --checkpoint-ms 1000 \
  --rounds 2 --pairs 10 --seconds 5 --warmups-per-arm-round 1 \
  --minimum-completed-nonterminal-epochs 4 --budget-seconds 250 \
  --work-dir benchmarks/checkpoint-v4/m3-raw \
  --output benchmarks/checkpoint-v4/m3-p3.7.json

TOKIO_WORKER_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/profile_checkpoint_v4.py runner \
  --binary target/checkpoint-v4/checkpoint-v4-evidence \
  --entities 64 --batch-rows 6400 --checkpoint-ms 100 \
  --rounds 2 --pairs 10 --seconds 5 --warmups-per-arm-round 1 \
  --minimum-completed-nonterminal-epochs 40 --budget-seconds 260 \
  --work-dir benchmarks/checkpoint-v4/m4-raw \
  --output benchmarks/checkpoint-v4/m4-p3.7.json
```

The driver constructs the checkpoint-off reference arm internally. Both arms
come from the same sealed candidate binary; checkpoint-off is always the ratio
reference. No parent binary or additional control arm is run. P2.1 evidence at
`benchmarks/rolling-kernel-v2/` is informational revision control only.

## Validation and statistics

- Every graph computes `difference(mean(5), mean(20))` followed by the exact
  column projection, with deterministic timestamps and 64-bit prices. Every arm
  validates its complete emitted prefix against the P2 numeric oracle.
- M3/M4 count only input rows whose expected output reached the ordinary sink by
  the 5 s cutoff. Admission stops at cutoff; settle/cancel runs outside the
  timer and inside the case budget.
- An on sample must contain at least four completed nonterminal epochs in M3 or
  forty in M4. An off sample must prove zero checkpoint objects. Missing
  coverage or failed cleanup invalidates the sample rather than triggering a
  retry.
- Record accepted rows, emitted rows, completed nonterminal epochs,
  freeze-to-forward latency, persistence latency, edge high-water marks,
  checkpoint reservation/actual bytes and recovery validation time for every
  arm.
- Pair metrics use adjacent AB/BA observations only. For M1/M2, one pair metric
  is `on_median_30/off_median_30 - 1`; for M3/M4 it is
  `on_rows_s/off_rows_s`. Report each round's median paired metric and seeded
  bootstrap 95% interval (seed 771031, 20,000 resamples), plus the pooled median
  over all 20 pairs. Do not substitute independent P50s or best samples.
- If either per-round interval crosses that case's gate, the case is
  `inconclusive` and does not establish the gate. A missed gate stops with the
  stage/resource breakdown for a decision. There is no expanded measurement set.

Build identities, all raw samples and the final JSON/Markdown reports are
committed under `benchmarks/checkpoint-v4/` by P3.7. This registration itself
asserts no timing, throughput or gate result.
