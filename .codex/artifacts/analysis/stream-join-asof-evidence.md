# Stream Join and ASOF evidence foundation

Issue: [#363](https://github.com/wegamekinglc/calc-flow/issues/363).
Base: `main@49d346df`. Branch: `feature/stream-join-asof-evidence`.

This deliverable covers the issue's first suggested PR: Phase 0.1, 0.5 and
0.7. Operator representations, Join status and checkpoint layouts belong to
later deliverables.

## Changes

- ASOF lockstep waits for cumulative sink row delivery instead of polling
  `job.status()`. The sink signals after Arrow conversion and table capture.
  Every accepted left row produces one result only after native finality.
  This workload therefore uses delivery as the next-pair gate; it does not
  wait for post-delivery operator status updates.
- The stream scope advances from v4 to v5. A declared baseline scope mismatch
  removes native stream cases from comparable baseline ids, making them new
  coverage. SQL and warm-append gates remain independent. Missing or
  unparsable catalog information retains the existing degraded gating rule.
- A separate `polars-1t` backend covers the same seven workloads and seven
  row scales as `polars`. Fresh workers set `POLARS_MAX_THREADS` before
  importing Polars. Measurement and aggregation require the actual pool
  size for the selected case. A direct 1-thread EngineCase rejects a larger
  pool instead of mislabeling a sample.
- The guide documents 22 shards, 48 engine cases per tier, 336 total engine
  cases, the reopened stream tiers, and the v5 timing boundary. Static Join
  status polling remains explicitly documented for Phase 0.2.

## Test-first evidence

Observed expected failures before each implementation:

1. Sink cumulative-delivery wait was missing; the lockstep integration probe
   reached `job.status()` and raised its polling assertion.
2. A v4 baseline was incorrectly classified as an interleaved v5 reference;
   the catalog still advertised v4.
3. The 1-thread backend and report column were absent, worker environment
   overrides were unsupported, and measurement/aggregation rejected a
   genuine 1-thread pool.
4. A direct 1-thread reference running with 32 threads did not raise.

## Local verification

Commands use the repository's managed `.venv/bin/python` and
`PYTHONPATH=target/issue363-python` with an absolute path from the worktree.
That site contains current Python sources and the native module extracted
from the investigation's branch wheel. There are no Rust or binding changes.
The first integration attempt used an older September Studio native build;
it was interrupted and replaced with the investigation wheel before the
functional integration results below.

- `pytest benchmarks/test_engine_stream.py -q`: 10 passed.
- `pytest scripts/test_benchmark_suite.py scripts/test_benchmark_measure.py
  scripts/test_benchmark_process.py scripts/test_benchmark_aggregate.py -q`:
  69 passed and 53 subtests passed.
- `pytest benchmarks/test_engine_comparison.py -q -k single_thread`:
  3 passed, including independent processes for 10 and 101 rows across all
  seven Polars workloads.
- Selected native ASOF tests: 4 passed, covering 320k-row multi-batch output,
  delayed watermarks on either side, and 1,024-row batches with status reads
  forbidden.
- Ruff check and format checks passed for the 12 touched Python modules.
- `git diff --check` and generated-contract drift checks passed.

No throughput gain is claimed. Sealed release measurements, the small-batch
performance target, full CI and coverage gates remain unverified.

## Specialist review

The final specialist review approved this implementation slice after two
documentation corrections: `window_sum` uses ten-second tumbling windows,
and engine comparisons share the candidate harness while warm comparisons
load each revision's adapters. No code blockers remain. This approval does
not establish performance acceptance or green required CI checks.

## Next deliverable

Phase 0.2 adds observable per-side Join progress and updates the static
fixture. Phase 0.6 supplies the order/recovery/complexity safety net before
J1. D1-D6 remain assumptions from the issue's recommendations, not recorded
public-contract approvals.
