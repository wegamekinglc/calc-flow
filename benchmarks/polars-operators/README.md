# Polars operator coverage: DAL-394

81 of 84 added cells completed: two rounds of ten samples per cell, 1,620 accepted measured checks and 162 warmup checks passed. Maximum absolute oracle error: 0. Total collection elapsed 1804.0 seconds, including the stopped initial path and final process cleanup. The 1,800-second budget was exhausted; no further collection was started. The missing cells are 10M/1T unique64, cs_mean and window_sum. Full inventory validation therefore fails, and this is an incomplete delivery pending an additional measurement budget.

The addition covers cumulative average, 64/256-row argmax, 64-row distinct
count, exact-time cross-section mean and ten-second tumbling sum. Both Polars
references now support every scenario in the existing matrix; no newly
requested cell requires an unsupported marker. The existing 1M interval-join
cap is unchanged. This task changes benchmark adapters and evidence only.

## Measurement plan

Measure only the 84 added cells: six operators, seven original row scales
(10 through 10M), and the existing 1T/32T references. Baseline catalog:
`1085c651` (main at task start). Candidate: the clean implementation commit
recorded in `results.json`. These are external-reference coverage measurements,
with no baseline timing or speedup/regression verdict.

Use the maintained collector: two rounds, ten samples per round, one warmup
and a fresh process per round, with a fixed 1,800-second total collection
budget including wheel installation, process startup, fixtures, oracles,
warmup, validation and cleanup. No native build is needed: reuse the sealed
DAL-389 release wheel only after checking that the candidate's complete native
source-input hash and Cargo lock hash match the original build. Preserve its
original revision and wheel/native hashes separately from the new harness
revision and hash. No timing observations from earlier tasks are reused.

## Semantics and timing

Every engine receives the same immutable Arrow fixture: up to 64 entities,
exact-eighth prices, UTC microsecond event times, unique sequence values and
64,000-row Arrow chunks. All new values are Float64. Every warmup and sample
is checked against the maintained independent NumPy/table oracles at
`rtol=atol=1e-10`. Tests additionally exercise repeated maxima, repeated values
leaving the distinct window, temporal/sequence ordering, negative timestamps,
exact microsecond bin boundaries and input immutability.

Average uses per-symbol cumulative sum/count ordered by event time and sequence.
Argmax returns the row lag of the oldest tied maximum in a partial/full frame.
Per-symbol rolling maxima supply the lookup price; forward ASOF from the frame's
starting row selects the earliest row with that price, which must lie in the
frame. This avoids expanding all rolling-group members. Unique count uses
`n_unique` in a native integer-index partial/full 64-row frame. CS mean
groups exact event times, and window sum groups symbol and UTC timestamps
truncated to epoch-anchored half-open ten-second bins. See
[suite semantics](../../docs/benchmark-suite.md#inputs-correctness-and-timing-boundaries).

The timer includes `plan.collect(engine="streaming").to_arrow()`, including
lazy execution, ordering, rolling groups, aggregation and Arrow output.
Input conversion and lazy plan construction, fixtures, warmup, process/IPC
overhead and correctness checks stay outside individual samples. Polars may
execute unsupported streaming operations in its in-memory engine; the report
records the requested collection engine, not a claim that every operator has
bounded streaming memory. All calculations use Polars native expressions.

## Evidence and environment

[Accepted results](results.json) identify the measurement cohort of every cell.
[Initial collection](initial-results.json) and [completion collection](completion-results.json)
preserve their original inventories, raw samples, environments and source identities.
[Validation](validation.json) seals the accepted results and records the unchanged
plan hashes; [build](build.json) preserves the reused wheel identity;
[host](host.json) records the machine and exact Python executable. The issue
attachment includes the verification script, worker requests, warmups,
termination record and collection logs.

Initial harness: `a6428dabb7d20fcbf2e4d8e7ab7cada802548fb4`; completion harness:
`1f0f6417ca8479b17c247613da77b63999373b49`. The first collection stopped after
911.3 seconds during the 10M/32T argmax256 warmup,
after observing about 21 GiB RSS in the rolling-group expression. All 25
completed initial argmax cells were excluded from the accepted results;
the final index-lookup expression was measured for all 28 argmax cells.
The 49 complete non-argmax cells were retained after verifying identical
unoptimized lazy plans for every unchanged operator. The completion collection
measured those 28 argmax cells and four of the seven previously unmeasured cells.
Its 880-second timeout expired while preparing 10M/1T unique64; settling the
owned process took total elapsed time to 892.7 seconds.
The remaining three cells are explicitly unmeasured. No partial round or
superseded argmax timing contributes to the table.

2026-10-21; Intel Core i9-13900HX, 16 exposed physical cores / 32 logical CPUs,
31.2 GiB RAM; Linux 5.15.167.4 WSL2. Python 3.13.9, Polars 2.0.0,
NumPy 2.5.2 and PyArrow 24.0.0. Both thread configurations use separate
processes with the requested Polars pool verified as 1 or 32; OMP, OpenBLAS
and MKL pools are 1. This shared host supplies informational external-reference
timings; these results establish coverage, with no paired version verdict.

The release wheel was built at `23edd3a39ad5b8ab9c493387b3e0ba9d22641442` with
Rust/Cargo 1.94.0, release optimization and CP313/CP313. Both measured harness
revisions have the identical complete native source-input and Cargo lock hashes.
Native SHA-256: `13e801f24536b27553bc9bff8897f363f8f023da5ef97981f67f3c4c52b30122`.

## Results

All values below are milliseconds. P50 and P95 use the maintained suite's
statistics over both complete rounds for each cell.

| Input rows | Scenario   | Polars 32T P50 | Polars 32T P95 | Polars 1T P50 | Polars 1T P95 |
|------------|------------|----------------|----------------|---------------|---------------|
| 10         | average    | 4.624          | 5.731          | 0.472         | 0.687         |
| 10         | argmax64   | 29.851         | 37.125         | 2.439         | 2.716         |
| 10         | argmax256  | 25.704         | 34.834         | 1.902         | 2.935         |
| 10         | unique64   | 7.033          | 8.084          | 1.070         | 1.442         |
| 10         | cs_mean    | 6.490          | 8.559          | 0.452         | 0.597         |
| 10         | window_sum | 5.910          | 6.747          | 0.641         | 1.017         |
| 100        | average    | 5.816          | 9.953          | 0.510         | 0.660         |
| 100        | argmax64   | 31.130         | 36.358         | 2.742         | 3.294         |
| 100        | argmax256  | 34.976         | 40.820         | 2.715         | 3.693         |
| 100        | unique64   | 16.931         | 19.099         | 0.887         | 1.505         |
| 100        | cs_mean    | 6.287          | 7.429          | 0.416         | 0.699         |
| 100        | window_sum | 6.703          | 7.232          | 0.616         | 0.857         |
| 1,000      | average    | 6.263          | 7.195          | 0.731         | 1.280         |
| 1,000      | argmax64   | 40.446         | 44.356         | 3.867         | 5.742         |
| 1,000      | argmax256  | 44.443         | 62.042         | 3.786         | 9.345         |
| 1,000      | unique64   | 22.753         | 25.242         | 1.725         | 2.077         |
| 1,000      | cs_mean    | 6.623          | 7.497          | 0.439         | 0.649         |
| 1,000      | window_sum | 15.439         | 18.479         | 0.685         | 0.917         |
| 10,000     | average    | 8.205          | 9.286          | 1.595         | 1.879         |
| 10,000     | argmax64   | 68.022         | 78.174         | 11.823        | 29.279        |
| 10,000     | argmax256  | 68.737         | 87.873         | 10.467        | 34.998        |
| 10,000     | unique64   | 27.732         | 33.101         | 8.768         | 9.192         |
| 10,000     | cs_mean    | 6.325          | 7.594          | 0.677         | 1.028         |
| 10,000     | window_sum | 15.583         | 17.149         | 1.098         | 1.350         |
| 100,000    | average    | 11.311         | 13.317         | 8.979         | 9.517         |
| 100,000    | argmax64   | 134.723        | 243.266        | 49.897        | 55.582        |
| 100,000    | argmax256  | 81.109         | 94.380         | 45.864        | 53.453        |
| 100,000    | unique64   | 44.962         | 50.635         | 96.050        | 98.867        |
| 100,000    | cs_mean    | 7.355          | 8.207          | 2.800         | 3.117         |
| 100,000    | window_sum | 17.095         | 19.332         | 4.264         | 4.958         |
| 1,000,000  | average    | 33.150         | 39.661         | 117.982       | 133.260       |
| 1,000,000  | argmax64   | 158.184        | 313.697        | 527.555       | 599.143       |
| 1,000,000  | argmax256  | 138.399        | 158.368        | 529.822       | 604.580       |
| 1,000,000  | unique64   | 137.640        | 168.299        | 990.722       | 1016.510      |
| 1,000,000  | cs_mean    | 15.180         | 19.002         | 21.343        | 24.855        |
| 1,000,000  | window_sum | 29.872         | 40.992         | 38.588        | 42.237        |
| 10,000,000 | average    | 261.099        | 307.840        | 1626.491      | 1802.157      |
| 10,000,000 | argmax64   | 1309.992       | 1547.381       | 6753.822      | 7705.206      |
| 10,000,000 | argmax256  | 1263.417       | 1418.224       | 7233.731      | 15519.624     |
| 10,000,000 | unique64   | 1480.239       | 2361.235       | unmeasured    | unmeasured    |
| 10,000,000 | cs_mean    | 113.136        | 194.279        | unmeasured    | unmeasured    |
| 10,000,000 | window_sum | 121.604        | 136.189        | unmeasured    | unmeasured    |

## Commands and validation

`BENCH_PY` was the prepared DAL-389 Python executable recorded in `host.json`;
the dependency environment was reused read-only. The benchmark dependency lock
is unchanged. The wheel was copied into `target/polars-release` with its original
manifest; `load_release` verified both wheel and native seals before collection.
The baseline catalog was extracted with `git show 1085c651:scripts/benchmark_suite/catalog.py`.

The actual collection commands, with that executable abbreviated, were:

```bash
PYTHONPATH=. "$BENCH_PY" benchmarks/polars-operators/collect.py \
  --release target/polars-release/release.json \
  --baseline-source target/polars-baseline \
  --output target/polars-operator-results --budget-seconds 1800

PYTHONPATH=. "$BENCH_PY" benchmarks/polars-operators/collect.py \
  --release target/polars-release/release.json \
  --baseline-source target/polars-baseline \
  --output target/polars-completion-results --budget-seconds 880 \
  --case-ids-file target/polars-completion-ids.json

PYTHONPATH=. "$BENCH_PY" target/verify_completed_polars.py
# Exit 1: completed artifacts validate, but three required cells are unmeasured.
```

The completion ID list and verifier are included in the issue attachment.
For a fresh reproduction from the final implementation, prepare the locked
benchmark environment and a sealed release wheel with identical native inputs,
then run `collect.py` without `--case-ids-file` to collect all 84 cells directly.
The collector validates selected IDs against the original added-cell inventory.

To finish only the three remaining cells after an additional budget is approved:

```bash
PYTHONPATH=. "$BENCH_PY" benchmarks/polars-operators/collect.py \
  --release target/polars-release/release.json \
  --baseline-source target/polars-baseline \
  --output target/polars-remaining-results --budget-seconds 360 \
  --case-ids-file benchmarks/polars-operators/remaining-case-ids.json
```

This command has not been run. Retain the 81 complete cells; the three remaining
cells still require two complete ten-sample rounds each.

Focused verification used the same prepared Python and the current source:

```bash
PYTHONPATH=. POLARS_MAX_THREADS=32 TOKIO_WORKER_THREADS=32 JAX_PLATFORMS=cpu \
  "$BENCH_PY" -m pytest benchmarks/test_polars_operators.py -q
# Initial implementation: 25 passed.

PYTHONPATH=. POLARS_MAX_THREADS=32 TOKIO_WORKER_THREADS=32 JAX_PLATFORMS=cpu \
  "$BENCH_PY" -m pytest benchmarks/test_engine_comparison.py -q -k polars
# Initial implementation: 32 passed, 181 deselected.

PYTHONPATH=. POLARS_MAX_THREADS=32 TOKIO_WORKER_THREADS=32 JAX_PLATFORMS=cpu \
  "$BENCH_PY" -m pytest benchmarks/test_polars_operators.py -q -k 'argmax or unique'
# Final implementation: 12 passed, 13 deselected.

"$BENCH_PY" -m unittest scripts.test_benchmark_suite \
  scripts.test_benchmark_workflow scripts.test_benchmark_measure \
  scripts.test_benchmark_worker -q
# 88 passed; shared suite/oracle code was unchanged by the argmax replacement.
```

Changed Python files passed Ruff lint/format checks; `git diff --check` passed.
The final artifact verification checks the 84-cell catalog and validates the
81 completed cells' thread counts, collection engines, native seals,
round/sample inventories, raw versus summary samples, recomputed statistics and
every accepted warmup. Its full-inventory gate remains failed for the three
missing cells. All 34 installed package versions match the unchanged lock.
An intermediate overbroad `-k 'argmax or polars_single_thread'` selection picked four
Finance-Python cases that failed because its separate Python 3.9 worker was not
configured; its 36 other tests passed. The focused Polars selection above passes.
No core runtime behavior changed. Full regressions and coverage remain with CI.
