# Stream Join hash-native index performance evidence

Branch `feature/stream-join-hash-index-j2a` (commits `b7cdddba`,
`d3bad343`) against `main` at `9db4cd25`. This executes the J2a index
redesign from issue #363: the native probe index becomes a distinct-key
hash dictionary of `u32` ids whose per-key entry lists stay sorted by
`(time, row_id)`, inclusive windows resolve through `partition_point`,
admission probes encode each distinct key once per batch into a reused
buffer, and index eviction releases empty key slots in lockstep with its
funding shrink (per-entry reservation 192 bytes). The gathered output
column path from commit `b7cdddba` was reverted in `d3bad343` after the
paired suite measurement showed a regression; the baseline per-row slice
concatenation stays on Arrow's contiguous-slice fast path.

## Measurement contract

- Machine: i9-13900HX, WSL2, 32 logical CPUs, 31 GB RAM, shared and
  noticeably time-varying load during collection.
- Both sides built as release wheels with the suite recipe
  (`maturin build --release --locked --features pyo3/abi3-py313`) from
  clean worktrees: baseline wheel SHA-256
  `a819b28d66fe03d2a2dc294dc7e14e2517bae9ed5a378d5f1b6c80b682a15687`,
  candidate wheel SHA-256
  `89f6400f0e1f6b557edaaf29cafa2ceea939b519ed5e1d086839609d1c089cd5`.
- Suite numbers use the engine-case adapters, workload, oracle and
  ready-enqueue-to-Arrow timer from `benchmarks/engine_comparison.py`,
  one fresh worker process per sample, alternating AB/BA order.
- These are investigation numbers on a shared host, not sealed
  two-round quiet-room verdicts; every reported sample passed the full
  output oracle.

## Suite results (paired, fresh worker per sample)

`calc-flow-stream join`, candidate `d3bad343`:

| Case      | Rounds | Baseline p50 | Candidate p50 | Median change (pooled) | 95% CI (exact order-statistic) |
|-----------|--------|--------------|---------------|------------------------|--------------------------------|
| 1,000,000 | 2 × 10 | 1,037.8 / 1,065.7 ms | 861.6 / 857.2 ms | −18.67% | [−19.59%, −16.25%] |
| 100,000   | 8 pairs | 114.5 ms | 97.5 ms | −13.59% | — |

The earlier three-way interleaved experiment with the gathered output
path (commit `b7cdddba`) measured +26.07% pooled
(CI [+23.35%, +28.12%]) at 1M and +24.5% at 100k, which motivated the
revert. Stage timing attributed that regression to output column
materialization (materialize ≈ the whole emit stage), while the
operator-level Rust benches had shown the same build faster; the
gathered path loses to Arrow's contiguous-slice concat fast path on this
workload's shape.

## Operator-level Rust benches (`stream_join_perf --quick`)

| Bench                          | main      | candidate | Change |
|--------------------------------|-----------|-----------|--------|
| handler/right_10k_no_match     | 8.78 ms   | 6.45 ms   | −27%   |
| handler/right_10k_one_to_one   | 16.76 ms  | 12.20 ms  | −27%   |
| handler/right_10k_fanout10     | 63.89 ms  | 58.83 ms  | −8%    |
| handler/watermark_evict_10k    | 2.77 ms   | 2.75 ms   | ≈      |
| checkpoint/capture_dirty_20k   | 42.96 ms  | 46.11 ms  | ≈ noise |
| steady_60k                     | 678 µs    | 465 µs    | −31%   |
| restore/full_20000             | 44.5 ms   | 52.3 ms   | ≈ noise |
| restore/full_60000             | 140.6 ms  | 149.7 ms  | ≈ noise |

The pre-revert build additionally measured one_to_one 6.70 ms (−60%) and
fanout10 10.20 ms (−84%) — the gathered output path is a large win at
operator level — but the suite-level regression above wins the decision.

## Verdict against the issue #363 targets

- Stream join 1M single-thread ≤ 60 ms: **not met**. Candidate p50 is
  858 ms on this host (baseline 1,038–1,066 ms under the same
  conditions); the improvement is −18.7% with a confirmed interval, not
  the ~11× required. The remaining cost sits in per-row admission
  (`quantum` stepping, per-row classification), per-row probe encoding,
  and output materialization; none of these are addressed yet.
- The index change is directionally correct and statistically confirmed;
  the gathered output experiment is documented as rejected evidence for
  the next iteration, which needs a take/zero-copy output design that
  preserves the concat fast path for contiguous incoming runs.

Raw sample JSON files are under `target/j2a-perf/engines-1000000-join/`
and `target/j2a-perf/engines-100000-join/` in the worktree.

## Iteration 2: take-based output gather (commit `ad3edbd6` range)

`materialize_output_record` now gathers each output side once per
materialization call: when every matched pair reads rows of one shared
payload chunk, that side's columns are built with one Arrow `take` per
column over the chunk's row offsets; dictionary columns and mixed or
legacy payloads keep the concatenated per-row-slice path (which retains
the unreferenced-dictionary-value contract). A failing test first
recorded 54 per-row slices for a 3x3 native match and passes at zero
after the change.

- Materialization bench (best-of runs): narrow_f100_fast 99.3 → 4.9 ms
  (−95%), wide_f10_fast 14.4 → 3.1 ms, wide_f100_fast 125.0 → 19.9 ms
  (−84%), wide_f100_slow 140.2 → 118.0 ms. The earlier per-pair
  `interleave` build measured 10.5/6.8/45.5/126.3 on the same cases, so
  the take gather also beats the rejected interleave shape.
- Suite `interval_join` 100k paired (6 pairs): baseline 797.4 ms p50 →
  candidate 701.7 ms p50 (−12.87%, all oracles passed).
- Suite `join` 100k paired (8 pairs): −14.63% (110.5 → 95.2 ms).
- Suite `join` 1M paired (2 × 10 pairs): −18.50% pooled,
  CI95 [−20.42%, −16.70%]; candidate p50 906.6 ms — statistically level
  with the index-only build (−18.67%), confirming the 1:1 static case's
  output share is small; fanout shapes collect the gather win.
- Stage attribution on the take build (quiet in-process): per 100k —
  prepare ≈ 40–48 ms (probe ≈ 12 ms; the rest is admission plus the
  owned payload-chunk copy that state-ownership accounting requires),
  emit ≈ 40 ms (take gather is a small share; the remainder awaits the
  edge/sink inside the suite's timed window), commit timer ≈ 18 ms but
  `commit_prepared` is O(1) with empty retention — wall-clock stage
  timers on a busy runtime absorb scheduler preemption and are no
  longer trustworthy at this granularity. The next attribution step
  needs a native sampler profile, and the next operator-local lever is
  vectorized admission (null/late/retain masks, batched row-id
  reservation); the emit-side remainder belongs to the runtime/sink
  path, not the operator.

## Iteration 3: parent-referenced legacy rows (final commit of this branch)

Fine-grained admission timing revealed the decisive fact: the suite's
default runtime never takes the funded copy path at all, because
`can_copy_payload` gates on `serial_owned_sql()` (DataFusion
`Fixed`-parallelism, one target partition) and the production default is
multi-partition. Every suite batch was admitted through the legacy path,
which sliced the input record into one wrapped single-row `RecordBatch`
per row (about 360 ns/row of wrapper allocation), and the take gather
could not engage for those payloads. Loosening the serial gate is a
specialist-review boundary (the funded ownership graph is proven only
for the exact serial plan subset), so instead the legacy path itself
became lean: admitted rows now carry `RowPayload::Rowed { parent:
Arc<RecordBatch>, row }` — one parent clone per batch, one refcount per
row — and `columns()`/`offset()` address the row through the parent.
`shared_chunk_id` now identifies the parent, so the iteration-2 take
gather engages for legacy-admitted sides too. Two funded-copy contract
assertions were widened from `Legacy(_)` to the unfunded legacy family
(`Legacy | Rowed`); observable contracts (per-row IPC bytes, charges,
pool zero, no funded lease) are unchanged.

Results (same paired protocol, oracle asserted on every sample):

| Case                    | main        | candidate   | change                          |
|-------------------------|-------------|-------------|---------------------------------|
| suite join 1M (2 × 10)  | 1,016.9 / 1,035.0 ms | 122.3 / 128.1 ms | **−87.83%**, CI95 [−87.92%, −87.74%] |
| suite join 100k (8)     | 113.1 ms    | 16.2 ms     | −85.49%                         |
| suite interval_join 100k (4) | 664.7 ms | 240.9 ms    | −63.8% (retained-state workload) |

Operator benches: one_to_one 12.2 → 7.38 ms, fanout10 58.8 → 9.42 ms,
evict ≈ level, capture/restore within noise of main.

The #363 single-thread target of ≤ 60 ms at 1M rows is still not met on
this shared host (candidate p50 123.7 ms; this host measures main about
1.6× slower than the historical quiet evidence machine, so the same
build should land near 75–85 ms there). The remaining suite time is now
dominated by the runtime/sink path inside the timed window and the
per-row admission loop itself; both are outside what further operator
index work can remove.

Rejected in this iteration, with reasons:
- Probe count/collect single-pass merge: the count pass over the hash
  index is already cheap and merging would reorder the pairs-credit
  reservation ahead of its bound, perturbing fail-closed funding.
- Vectorized admission masks: superseded — the dominant admission cost
  was per-row wrapper allocation, removed by the Rowed representation.
- Enabling the funded copy path for non-serial runtimes: explicitly
  outside the proven serial subset; specialist review required.

## Iteration 4: batch-mask legacy admission

`admit_legacy_record` now classifies the whole record with batch-level
masks (event-time null mask, key-null mask with a no-null fast path per
column) and one block row-id reservation per record, instead of one
`classify_row` call per row. The frozen precedence is preserved and now
pinned by tests: null event time drops first, a non-null row's timestamp
conversion failure still errors before a null-key drop, lateness last.
The copy path (serial runtimes only) keeps per-row classification.

Paired 1M (2 × 10 pairs): −88.28% pooled, CI95 [−88.76%, −87.50%],
candidate p50 144.0 ms on a noisier evening host where main itself
measured 1,108–1,318 ms; the standalone contribution of this iteration
over iteration 3 is about half a percentage point — the structural value
is the range-local admission shape that J3 shards directly. A zero-copy
contiguous output slice was evaluated and deferred: Arrow sliced arrays
report their full parent buffers in memory accounting, so an unfunded
shared-output path would over-count against the edge byte budget until
a shared-output accounting story exists.
