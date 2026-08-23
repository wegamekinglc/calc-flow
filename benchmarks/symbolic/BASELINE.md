# Symbolic Execution Baselines (SCE-01)

Raw paired benchmark results for hand-built calc-flow plans that compute the
workloads the symbolic layer will compile ([GitHub #168], SCE-01 of the
[symbolic computation engine plan]). These numbers record what the current
runtime does; they make no absolute performance claim and gate nothing.

[GitHub #168]: https://github.com/wegamekinglc/calc-flow/issues/168
[symbolic computation engine plan]: ../../docs/superpowers/plans/2026-08-22-symbolic-computation-engine.md

## Recorded runs

| Run   | Artifact                                   | Date          |
| ----- | ------------------------------------------ | ------------- |
| run 1 | `baseline-b9deff7-run1.json`               | 2026-08-23    |
| run 2 | `baseline-b9deff7-run2.json`               | 2026-08-23    |

Both runs executed the identical command against base commit
`b9deff784daa9db35a521e7e765da5c5c9092dca` (`origin/main`):

```bash
uv sync --extra benchmark --extra dev
uv run maturin develop
CALC_FLOW_BENCHMARK_SCALE=standard JAX_PLATFORMS=cpu \
  uv run pytest benchmarks/test_symbolic_baseline.py -q --benchmark-only \
  --benchmark-json=benchmarks/symbolic/baseline-b9deff7-run<N>.json
```

## Environment

| Property         | Value                                                         |
| ---------------- | ------------------------------------------------------------- |
| Machine / host   | `chengli-i9`, WSL2 (`5.15.167.4-microsoft-standard-WSL2`)     |
| CPU              | 13th Gen Intel Core i9-13900HX, 32 logical CPUs               |
| Python           | CPython 3.13.9                                                |
| calc-flow        | 3.0.0                                                         |
| NumPy / pyarrow  | 2.5.2 / 24.0.0                                                |
| JAX              | 0.11.1, `JAX_PLATFORMS=cpu`, x64 disabled                     |
| Scale            | `standard`: 100,000 table rows; stream capped at 50,000 rows  |

The quote workload at `standard` resolves to 99,968 rows over 64 entities in
8 industries (complete 8-member cross sections). The stream workload is
40 entities × 1 row/second with 60-second tumbling windows and 2,500-row
batches, so active operator state per entity is bounded by a 60-row history.

## Scenario results (per-execute mean over pytest-benchmark rounds)

| Scenario                              | Rows in/out     | Run 1 mean | Run 2 mean | Δ      | Rows/s (run 2) |
| ------------------------------------- | --------------- | ---------: | ---------: | -----: | -------------: |
| projection 20 derived columns         | 99,968 → 99,968 | 0.0568 s   | 0.0615 s   | +8.2%  | ≈1,625,000     |
| rolling 20/60-row temporal features   | 99,968 → 99,968 | 1.3572 s   | 1.2978 s   | −4.4%  | ≈77,000        |
| cross-section rank/z-score            | 99,968 → 99,968 | 0.5764 s   | 0.6457 s   | +12.0% | ≈154,800       |
| table_matmul NumPy (20×8 weights)     | 99,968 → 99,968 | 0.1529 s   | 0.1554 s   | +1.6%  | ≈643,300       |
| table_matmul JAX (float32)            | 99,968 → 99,968 | 0.1475 s   | 0.1588 s   | +7.7%  | ≈629,500       |
| stream window checkpoint + recovery   | 25,000 → 440    | 0.1182 s   | 0.1318 s   | +11.5% | —              |

The stream row counts only the 25,000 rows consumed before the checkpoint
(the cancelled remainder is dropped by design); its 440 output rows are the
tumbling windows accumulated from that consumed half.

Supplementary recorded metrics (identical in both runs because they are
deterministic workload properties):

| Metric                                    | Value                          |
| ----------------------------------------- | ------------------------------ |
| DataFusion query count per execute        | 1 (projection, rolling, cross) |
| table_matmul provider calls per execute   | 1 (NumPy and JAX)              |
| Arrow feature-column bytes (NumPy)        | 15,994,880                     |
| Dense matrix bytes (NumPy float64)        | 15,994,880                     |
| Arrow/dense bytes (JAX float32)           | 7,997,440                      |
| Peak process RSS (stream scenario)        | ≈577 MB (includes JAX import)  |

Peak RSS is the process-wide monotonic high-water mark, not a per-scenario
attribution; the stream reading includes the resident JAX runtime.

Stream checkpoint metrics from the measured (non-timed) lifecycle:

| Metric                 | Run 1      | Run 2      |
| ---------------------- | ---------: | ---------: |
| Checkpoint duration    | 23.7 ms    | 23.4 ms    |
| Checkpoint bytes       | 107,141    | 107,141    |
| Recovery duration      | 18.9 ms    | 19.6 ms    |
| Batches before pause   | 10 of 20   | 10 of 20   |

The stream lifecycle boundary: compile the window graph, run to half the
batches, take one durable checkpoint, cancel (dropping the unconsumed half
by design), then start a second runner that restores the checkpointed
operator state and flushes it at drain. `pytest-benchmark` times that whole
lifecycle; checkpoint duration, checkpoint bytes, and recovery duration come
from `perf_counter`/disk scans inside the dedicated measured lifecycle
recorded in each benchmark's `extra_info`.

The recorded `recovery_resumed_batches` (11 in both runs) counts in-flight
source reads discarded by the recovery drain — a timing artifact, not a
stable workload property.

## Noise and confidence

- Run-to-run means differ by 1.6%–12% on this machine. Treat any single
  number as ±12% until more samples exist; the repository-wide guidance
  (compare only matching machine/dependency/workload fingerprints, collect
  ≈20 comparable samples before classifying) applies.
- This host is WSL2 with shared host memory and CPU frequency scaling;
  absolute values will differ from bare-metal CI runners. Only
  same-machine, same-process paired comparisons are meaningful.
- Rounds per scenario are low (3–9) because `max_time=0.5` with slow
  scenarios; the min/mean spread inside each run stays under ~10%.
- Checkpoint bytes are byte-identical across runs (deterministic state
  serialization), which is the strongest reproducibility signal here.

## Comparison method for later SCE phases

1. Re-run the same command on the same machine and process state against
   both `origin/main` (these baselines) and the symbolic-compiler branch.
2. Compare paired means per scenario; the initial row-local regression gate
   is five percent
   ([design](../../docs/superpowers/specs/2026-08-22-symbolic-computation-engine-design.md)).
   Stateful and matrix gates must be set from these baselines before their
   implementations begin, not from absolute targets.
3. For the stream scenario compare `checkpoint_duration_seconds`,
   `checkpoint_bytes`, `recovery_duration_seconds`, and the lifecycle mean
   from `extra_info`; for matmul compare provider calls and Arrow/dense copy
   bytes in addition to the mean.
4. A result is comparable only when scale, seed, entity count, and input
   order match exactly; all are fixed by `benchmarks/symbolic_support.py`.
