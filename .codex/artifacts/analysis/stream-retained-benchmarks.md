# Retained stream benchmark coverage

## Design checkpoint

Issue #363 FR6 and FR7 in the reviewed acceleration specification govern this
delivery. It changes benchmark adapters and evidence, with no Rust runtime,
Python public API, or Studio contract changes.

- Interval input uses 64 keys, one tick per key per second, exact-eighth
  prices, and inclusive five-second bounds in both directions. Every backend
  projects both physical row identities and the same product. The independent
  pair oracle retains boundary matches and duplicate-key multiplicity.
- Native interval execution retains both inputs and advances their watermarks.
  SQL/DataFusion and Polars use eleven equivalent integer-second offset
  equality probes for this declared grid fixture. The algorithm is explicit
  in case identity; it is not a measurement of arbitrary non-grid interval SQL.
- The interval catalog has an explicit 1M-row cap for every backend. A 10M
  fixture would emit roughly 110M rows, so reports mark that tier unsupported.
- Native Join, interval Join, ASOF, and projection add 1,024-row throughput
  cases. The existing 64,000-row throughput cases remain.
- Checkpoint duration cases run only at 100k and 1M rows with both batch sizes.
  Their lifecycle scope includes a declared 100 ms delay, durable epoch
  acknowledgement, controlled cancellation, manifest recovery, and restart.
  Initial plan compilation and runner readiness precede timing. These cases
  are shown separately from throughput references.
- An immutable event-log source restores the exact next data position with
  stable cursors and legally replays its equal accepted watermark. A prefix
  stops feeding before the durable cut; accepted output is retained and
  combined with the resumed suffix without deduplication. Ordinary delivery
  stays at-least-once.
- Case and raw sample evidence attest batch rows, checkpoint interval, replay
  mode, source mode/binding IDs, workload and scope. Checkpoint evidence needs
  an acknowledged nonterminal epoch, a nonterminal input position, and a
  successful recovery oracle. Configuring a timer alone cannot pass.

## TDD status

Before implementation, `benchmarks/test_retained_stream.py` recorded the
unsupported scenario argument, absent interval/reference and checkpoint
catalog cases, missing configurable batch size, and missing owned checkpoint
operation helper. `scripts/test_retained_benchmark_suite.py` recorded its
missing raw-evidence validator. Each selected red run completed in under one
second with no native calculation or build.

Additional focused red tests exposed static/interval fixture misclassification,
missing lookup no-left-state proof, a delayed final watermark leaving eviction
unobserved, variant classification despite changed baseline batch/duration
dimensions, replay reopening at the old position, rejection of a valid
checkpoint receipt after a later epoch, and acceptance of lifecycle evidence
shorter than its declared delay. Production fixes followed those failures.

## Verification

The Phase 0.2 dependency was cherry-picked as `0e16fb94`, equivalent to the
reviewed `3505903d`. Its one merge conflict preserved the sealed dimension plus
one input-batch limit. Static setup is now selected explicitly, so single-batch
interval input never enters the dimension preload path.

Focused checks used Python 3.13 and the managed functional native installation
under `target/issue363-join-progress/python`. This native build is **debug** and
is used only for correctness, not performance evidence.

- 21 retained-adapter tests passed across scoped red/green invocations:
  inclusive pair oracle, native/SQL/DataFusion/Polars references, on-time
  out-of-order input, delayed eviction frontiers, cursor seek/tampering/reopen,
  owned failure cancellation, checkpoint/restart at 10 and 4,097 rows for all
  four scenarios, a later-epoch receipt race, and a 100k-row interval run with
  more than 1M oracle-checked output rows and bounded retained state.
- `benchmarks/test_engine_stream.py`: 14 passed.
- `benchmarks/test_engine_comparison.py -k 'single_thread or
  asof_small_batches'`: 4 passed, including fresh-process Polars 1T execution
  of all eight reference workloads and actual 1,024-row ASOF configuration.
- `scripts/test_benchmark_suite.py scripts/test_benchmark_measure.py
  scripts/test_benchmark_aggregate.py scripts/test_retained_benchmark_suite.py`:
  79 passed and 54 subtests passed. These include original warm-up/sample and
  aggregation evidence rejection, inventory, per-scope classification, report
  separation, actual checkpoint receipts, and documented table alignment.
- Ruff check and format check passed on the 11 affected Python files.
- Generated project/OpenAPI/TypeScript contracts have no additional drift;
  `git diff --check` passed.

No throughput gain is claimed for these new scopes. Sealed release pair
measurements, 1M interval execution/cost validation, full scheduled matrix, and
CI gates remain unverified. Initial ready setup is excluded; checkpoint
lifecycle timing includes the declared delay, acknowledgement, cancellation,
manifest read, fresh plan compilation and runner restart. Final frontier
observation, EOF, terminal checkpoint and cleanup remain outside the timer.
Ordinary delivery is at-least-once, with a controlled cut and no deduplication
used to establish the recovery oracle.

Final specialist review identified malformed variant evidence crashing report
rendering. Four focused tests reproduced failures for both variant types with
incorrect results and nonfinite samples. Shared guarded P50 formatting now
retains invalid rows and the complete evidence-failure report; all eight
affected report tests and scoped Ruff/format checks passed. Final specialist
re-review approved the correction and the complete Phase 0.3/0.4 delivery.
