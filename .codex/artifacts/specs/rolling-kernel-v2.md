# P2 Rolling kernel v2 — specification

Date: 2026-11-15. Source baseline: `main` `39cab2fb` (rolling, array and
benchmark sources are unchanged on `feature/stream-session`). Branch:
`feature/rolling-kernel-v2`. Plan:
[streaming core re-architecture](../../../docs/plans/2026-11-15-streaming-core-rearchitecture.md),
phase P2 (§3.6, §4.2, §4.4, §5, §7.3, §7 gates, §8). Status: draft for
`cf-api-designer` (public removals, §4.6 and §6.3) and `cf-critic` (numeric
contract, §4); then implementation as the PR series in §9.

Paths below are relative to `crates/calc-flow/src/` unless they start with a
repository directory.

## 1. Decisions applied

The plan's decisions, scope and gates are fixed inputs:

- Decision 2: `stable_v1` and `stable_v2` are deleted; §4 is the only rolling
  numeric contract.
- Decision 3: rolling checkpoints written before P2 are not restorable. Old
  layouts and encodings are deleted, not kept readable. Restore from the new
  layout continues bit-identically.
- Decision 4: `unsafe_code = "forbid"` stays; vectorization comes only from
  autovectorizable safe loops.
- §8: every PR starts with a failing test, deletes what it replaces, and
  measures at most five cases in 600 s.

Decisions this spec takes where the plan leaves room:

- **D1** (plan §10.3): the DataFusion AVG-only rewrite keeps DataFusion 54's
  sticky-NaN behavior (`operator/rolling/kernel.rs:3118-3123`); it becomes a
  plan-time NaN policy, not a per-row branch.
- **D2**: `RollingSpec.state_layout_version` keeps its accepted declaration
  values {1, 2, 4, 5, 6} and admission gates (`operator/rolling/compile.rs:22-61`)
  because they are project-document declaration levels, not encodings
  (`docs/symbolic-design.md`, "Writer layout is distinct from the declaration
  fields"). What is deleted is every durable reader/writer keyed by layouts
  1–6. All declarations write one new layout, 7.
- **D3**: the rolling entity-parallel lane path is deleted with the state it
  operates on; large-batch parallelism returns with the P6 morsel pool (§11 R1).
- **D4**: gates G1–G3 are evaluated at the rolling-operator dispatch boundary
  (§10.2). The plan's 6.1 µs and 329 µs figures included the projection; the
  baseline is re-measured at the gate boundary.

## 2. Problem

Per plan §3.6, at `b9a46e12` a warm 1-row dual-SMA append costs ~6 µs of
fixed dispatch and every row costs 33–51 ns. Verified sources of that cost:

- West/Welford add/remove (`array/rolling_mean_math.rs:63-116`) puts two
  divisions per window-row on a loop-carried `mean`. `stable_v2` adds periodic
  rebasing (`rebase_due`, `:145`; `operator/rolling/kernel.rs:3125-3145`).
- `RowConverter` is built per dispatch: entity keys (`kernel.rs:2477`),
  `encode_bounds` (`kernel.rs:2607-2632`) and `last_ordered_identity`
  (`kernel.rs:729-734`).
- Each stage guard takes mutexes and reads the clock
  (`operator/rolling_metrics.rs:340`, `:434`, `stage()` at `:443`).
- `format!` builds the port-validation context on every call
  (`operator/rolling.rs:1588`, `:1728`).
- Every touched entity is deep-cloned per batch (`kernel.rs:1740`). The
  in-place undo path excludes `MeanKernel`, shared row tapes and large batches
  (`kernel/in_place.rs:420-428`) and is skipped when checkpointing is on
  (`operator/rolling/ordered_stream.rs:337-383`). With checkpointing on,
  `RollingHistories` keeps a second copy of each entity's history.
- State is `HashMap<Vec<u8>, usize>` → `Vec<Arc<TypedEntityState>>` →
  `Vec<TypedWindowState>` → `VecDeque<TimedSample>`/`Arc<RowsTape>`
  (`kernel.rs:610-616`), with 24 bytes per sample.
- One `lag` or `delta` output (`crates/calc-flow/rolling-kernels.json`,
  `"complexity": "general"`) moves the whole node to the `ScalarValue`/
  `BTreeMap` row path. String or temporal `min`/`max`, `rank`, `quantile`,
  non-Float64 or Duration scans, and indicator-containing mixed nodes take the
  same path (`kernel.rs:1943-2120`).
- Restore rebuilds sliding accumulators by replaying adds over history
  (`operator/rolling/state_codec.rs:1158-1175`). Output after restore can
  therefore differ in the last bits from an uninterrupted run.

## 3. Goals and non-goals

Goals:

1. One numeric contract for rolling sum, mean, dispersion and pair outputs,
   checked against an exact oracle within declared bounds (§4).
2. Rolling state in an SoA entity slab on a minimal state-table interface
   that P3 can re-implement (§5).
3. Every output kind in the typed plan. Batch atomicity comes from an undo
   log for all group kinds, with checkpointing on or off (§5).
4. A two-tier kernel plus a full-tick tier. Per-dispatch fixed costs are
   removed (§6).
5. One checkpoint layout with bit-identical restore. Layouts 1–6, their codecs
   and `RollingNumericalProfile` are deleted (§7, §8).
6. Gates: G1 native 1-row append ≤ 1 µs; G2 ≤ 4 ns per window-row; G3
   6,400 rows × 64 entities ≤ 100 µs (§10).

Non-goals:

- Projection, `EnvelopeCost`, edge, runner, `StreamSession` or the sync
  operator trait (P1/P7). Snapshot pages, dirty bitmaps, background
  persistence and manifest v4 (P3). P2 keeps the current barrier protocol and
  manifest v3.
- The keyed-state core shared with group by and join (P4). P2's typed routing
  is rolling-local, and P4 may absorb it.
- Sharding, the morsel pool and `x86-64-v3` wheels (P6). No SIMD intrinsics,
  no `unsafe`.
- `operator/window.rs` and SQL compact accumulators (P7, P4). The DataFusion
  COUNT/AVG transitions (`datafusion_rolling/sql_aggregate.rs`),
  `sql_sum_range.rs` and the composite argmax rewrite keep DataFusion's
  operation order and are not changed.
- New output kinds or options. Batch MIN/MAX (2026-11-08 plan C3).

## 4. Numeric contract

### 4.1 Scope by output kind

| Output kind                                            | P2 arithmetic                                          | Guarantee                           |
|--------------------------------------------------------|--------------------------------------------------------|-------------------------------------|
| `sum` over integers                                    | unchanged: exact wide sum, checked narrowing           | exact; overflow fails the callback  |
| `sum` over floats                                      | compensated sliding sum                                | §4.4 sum bound                      |
| `mean`                                                 | compensated sliding sum, divided at readout            | §4.4 mean bound                     |
| `variance`, `stddev`                                   | shifted compensated moments                            | §4.4 dispersion bounds              |
| `covariance`, `correlation`                            | shifted compensated co-moments                         | §4.4 pair bounds                    |
| `difference`                                           | one subtraction of its two leaf readouts               | leaf bounds plus one rounding       |
| `count`, `min`, `max`, `lag`, `delta`                  | unchanged                                              | bitwise equal to baseline           |
| `argmax`, `argmin`, `rank`, `quantile`, `unique_count` | unchanged, typed scans                                 | bitwise equal to baseline           |
| `decay`                                                | unchanged certified incremental / reference fold       | bitwise equal to baseline rule      |
| `ewma`, `cumulative_*`, `rsi`, `atr`, MACD components  | unchanged recurrences                                  | bitwise equal to baseline           |
| array `rolling_mean`, `RollingMeanState`               | same as table `mean`                                   | bitwise equal to table `mean`       |
| DataFusion AVG-only rewrite                            | same as `mean`, sticky NaN (D1)                        | §4.4 mean bound                     |

"Bitwise equal to baseline" means equal to the `39cab2fb` build on the same
ordered rows in an uninterrupted run with no decay cache refusal. Existing
expected values for these kinds must not be regenerated.

### 4.2 Sample classification (unchanged)

- Native policy (table outputs, array `rolling_mean`): null and NaN are not
  valid samples. ±Inf are valid samples. `min_periods` counts valid samples.
- Sticky policy (D1, AVG-only rewrite): NaN is a valid sample. Once one enters
  an entity's partition, `mean` reads NaN for the rest of that partition.
- Readout order is unchanged: the `min_periods` null gate, then the divisor
  null gate (`valid − ddof ≤ 0`), then classification. Both signs of infinity
  read NaN for sum and mean. One sign reads that infinity. Any infinity reads
  NaN for variance, stddev, covariance and correlation.
- Integer inputs are converted to f64 by correctly rounded `as f64`, except
  for `sum` and the numerator of `mean`. Float32 is widened exactly.

### 4.3 Arithmetic

Each window group carries only the accumulator families its readouts need,
chosen at plan time. A readout's bits never depend on which other outputs
share its group.

- **Counters (exact).** `valid`, `finite`, `pos_inf`, `neg_inf`, `nan_seen`
  (sticky policy only) and `since_rebuild` (u32).
- **Sum family.** `(s, e)` over finite valid samples. Each operand `o` is
  applied as `(s, t) = two_sum(s, o); e = e + t`, using Knuth's branch-free
  TwoSum. The correction is never written back into `s`. In a Rows frame where
  one finite sample leaves and one enters, the row applies one paired
  operation: `(δ, δe) = two_sum(x_in, −x_out); (s, t) = two_sum(s, δ);
  e = e + (δe + t)`. The loop-carried chain is therefore one add on `s` and
  one on `e` per row. Every other case applies operands one at a time:
  evictions oldest-first, then the insert.
- **Integer sum.** The existing exact wide accumulator. Integer `mean` is
  `fl(fl(S as f64) / valid)` with exact `S`.
- **Dispersion family.** Shift `k`, then `(ds, de)` for `Σd` and `(qs, qe)` for
  `Σq`, where `d = fl(x − k)` and `q = fl(d·d)`, under the same pairing rule.
  An evicted sample recomputes `d` from its ring value and the same `k`.
- **Pair family.** Shifts `kx`, `ky`, then compensated `Σdx`, `Σdy`,
  `Σdx·dy`, `Σdx²` and `Σdy²` over pairwise-valid positions.
- **Shift choice.** `k` is set by the first finite valid sample inserted into
  an empty group. At each rebuild it is the newest finite valid sample in the
  window.
- **Per-row order.** For each finalized row of an entity and each group:
  evict, insert, `since_rebuild += 1`, rebuild if due, then read out. This
  order is the same in every tier.
- **Rebuild.** A rebuild is due when `since_rebuild` reaches `R`, or when any
  float sum of the group is non-finite. The rebuild resets every float sum to
  `+0.0`, re-chooses `k`, folds the window's finite valid samples
  oldest-to-newest with the single-operand rule, and sets `since_rebuild = 0`.
  For Rows frames, `R = max(W, 64)` with `W` the frame size. For Duration
  frames, `R = max(n_prev, 64)`, where `n_prev` is the number of positions
  retained right after the previous rebuild (64 before the first). Counting
  every finalized entity row, valid or not, makes the cadence independent of
  batch splits.
- **Reset.** When `finite` reaches 0, all float sums reset to `+0.0`, `k`
  becomes unset and `since_rebuild = 0`.
- **Readouts.**
  - Finite sum: `F = s` if `s` is non-finite, else `fl(s + e)`.
  - Infinity term: `I` is +inf, −inf or NaN from `pos_inf` and `neg_inf`,
    and 0 when neither is present.
  - `sum = F` when `I = 0`, else `fl(F + I)`. Sticky NaN reads NaN.
  - `mean = fl(sum / valid)`.
  - `D = fl(ds + de)`, `Q = fl(qs + qe)` and `M = max(0, fl(Q − fl(fl(D·D)/n)))`.
  - `variance = fl(M / (n − ddof))` and `stddev = fl(√variance)`.
  - Pair readouts use `Mx`, `My` and `C = fl(Pxy − fl(fl(Dx·Dy)/n))`, where
    `Pxy` is the readout of the compensated `Σdx·dy`.
    `covariance = fl(C / (n − ddof))`. `correlation` reads null when
    `Mx == 0` or `My == 0`, else `fl(C / fl(fl(√Mx)·fl(√My)))`, with no
    clamping, as today (`kernel.rs:4249-4257`).
- **Constant windows.** When a window has been constant for at least `R`
  positions, `Σd`, `Σq` and `M` are exactly zero. A rebuild inside the
  constant stretch sets `k` to that constant.
- **Decay.** Arithmetic is unchanged (`kernel/decay.rs`). Its summary and
  `since_seed` become persisted slab columns. The per-entity cache, its
  `MemoryReservation` and the quota-reference mode are deleted, so output
  follows the certified-or-reference rule for every batch split and after
  restore.
- **Unchanged recurrences.** EWMA, cumulative outputs and RSI/ATR/MACD keep
  their recurrences. Their state bits are persisted exactly.

### 4.4 Exact oracle and error bounds

**Oracle.** A Python oracle in `python/tests/test_rolling_numeric_contract.py`
uses `fractions.Fraction`, which is exact for every finite f64 and integer.
It replays per-entity frame membership (Rows and Duration), the §4.2 policies,
the rebuild, reset and shift schedule, and the operand sequence since the last
rebuild. From these it computes exact `S`, `n`, `μ`, `Q_K` and co-moments,
the bound terms below, and the exact classification. It asserts classification
equality exactly and value bounds with exact rational comparison.

**Notation.** `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`, `γ_k = k·u/(1 − k·u)`. At position
`p`, `r ≤ p` is the group's last rebuild or reset. `O` is the operand sequence
of one compensated sum since `r`: the rebuild fold, then each insert and each
negated eviction, where a paired row counts as two operands. `m = |O|` and
`A = Σ|o|` over `O` (exact). `A_D`, `A_Q` and `A_P` are `A` for the deviation,
square and product sums. `K` is the shift, `Q_K = Σ(x − K)²` over the
window's finite valid samples, `M = Σ(x − μ)²`, and `n` is the valid count.

**Domain.** The bounds apply to finite windows with no infinity and no sticky
NaN, when `A`, `A_Q`, `A_P` and `n·Q_K` are at most 2¹⁰⁰⁰. Outside the domain,
outputs are deterministic and classified by §4.2 and §4.3 but unbounded.

Declared bounds on `abs(computed − exact)`:

- **sum**: `u·abs(S) + 2γ_m²·A`.
- **mean**: `3u·abs(μ) + 3γ_m²·A/n`, or `3u·abs(μ)` for integer input.
- **variance**:
  `Δv = [12u·Q_K + 3γ_m²·A_Q + 5γ_m²·A_D·√(Q_K/n) + 4m·η] / (n − ddof) + u·v`.
- **stddev**: `min(√Δv, Δv/σ) + u·(σ + √Δv)`, or `√Δv + u·√Δv` when `σ = 0`.
- **covariance**: `[12u·√(Q_Kx·Q_Ky) + 3γ_m²·A_P +
  5γ_m²·(A_Dx·√(Q_Ky/n) + A_Dy·√(Q_Kx/n)) + 4m·η] / (n − ddof) + u·abs(c)`.
- **correlation**: when `ΔMx ≤ Mx/2` and `ΔMy ≤ My/2`,
  `2ΔC/√(Mx·My) + 2·abs(r)·(ΔMx/Mx + ΔMy/My) + 6u·abs(r)`; otherwise only the
  classification is asserted.
- **difference**: `Δa + Δb + u·(abs(d) + Δa + Δb)`, with `Δa` and `Δb` the
  leaf bounds.

`ΔC`, `ΔMx` and `ΔMy` are the bracketed numerators of the covariance and
variance rows, before division by `n − ddof`. The sum term follows
Ogita–Rump–Oishi's `Sum2` bound. The first-order constants come from rounding
`d`, `q`, `D·D/n` and the final subtraction. `cf-critic` verifies the
derivation. A corpus violation is an implementation defect unless the critic
shows the bound itself is wrong.

**Acceptance wording.** For every corpus position, the computed output has the
oracle's classification. A finite output lies within the declared bound of the
exact value. Outputs are bit-identical (`f64::to_bits`) across the
determinism axes in §4.5.

**Corpus** (seeded, ≤ 20,000 rows in total, runs in seconds):

1. `1e9 + N(0, 1e−3)`, W = 20, 10,000 positions.
2. Alternating `±1e16` mixed with unit values.
3. Magnitudes `1e−300..1e300` with random signs, inside the domain.
4. Integers near `2⁵³` and at the i64/u64 extremes for `sum` and `mean`.
5. Null, NaN and ±Inf injections under both policies.
6. Constant stretches of at least `R` positions, including correlation's null
   classification.
7. Duration frames with bursts and gaps.
8. W = 5 over 10,000 positions, crossing many rebuilds.

### 4.5 Determinism

For one declaration and one canonical ordered row sequence per entity, every
output bit and every persisted state bit is independent of:

- Data batch boundaries and watermark cadence;
- finite (batch) versus stream evaluation, and a `StreamSession` push
  sequence;
- tier choice and pre-scan outcome;
- checkpoint and restore at any barrier;
- which other outputs share a group;
- entity count and `EntityId` assignment;
- thread count.

### 4.6 Removed public surface (`cf-api-designer` confirms)

Delete `RollingNumericalProfile`, including its `lib.rs:119` and
`operator/mod.rs:65` exports, and every `numerical_profile` surface:

- the `RollingSpec.numerical_profile` field (`operator/rolling.rs:1345-1349`);
- `RollingMeanOptions::{with_numerical_profile, numerical_profile}`
  (`array/rolling_mean.rs:58-78`);
- array provider IR, options and schemas (`array/provider/ir.rs:112-120`,
  `array/provider/stream/options.rs:98-124`, `array/provider/schema.rs:131`,
  `array/provider/stream/schema.rs:29`);
- the sequence codec profile marker (`array/sequence/codec.rs:47-51`, `:139`);
- the PyO3 parameter (`crates/calc-flow-python/src/array.rs:243-260`) and
  `_native.pyi:127`;
- Python `rolling_mean(..., numerical_profile=...)`
  (`python/calc_flow/_native_array.py:191`), where passing it now raises
  `TypeError`;
- the `python/calc_flow/capabilities.py:755-767` sequence-schema check;
- `profile=` in the symbolic optimizer's explain line
  (`python/calc_flow/symbolic/optimizer.py:632`).

`deny_unknown_fields` rejects project documents and IR options that still
carry the field. Documents that omit it keep their configuration hashes,
because `stable_v1` was never serialized.

Expected `schemas/project-v3.schema.json` diff:

- `$defs.RollingNumericalProfile` (lines 2466–2480) is removed.
- `RollingSpec.properties.numerical_profile` (lines 3709–3712) is removed.
- Description text is regenerated from the reworded `state_layout_version`
  rustdoc (D2).
- No other change.

The array sequence-state mean marker becomes a single new value. Markers 1 and
2 are rejected with an error naming the removed profiles. The EWMA, RSI, ATR
and MACD sequence encodings are unchanged.

## 5. State

### 5.1 Minimal state-table interface (the P3 seam)

This is a new crate-private module `state/table.rs`. It offers only what
rolling needs:

- **Column kinds.**
  - Keyed columns hold one fixed-width value per row (`EntityId`).
  - Ring columns hold per-row rings with a fixed logical capacity for Rows
    retention, or growable rings for Duration retention.
  - A value log holds Arrow-backed values addressed by u32 handles.
  - The entity dictionary maps `EntityId` to canonical key bytes and typed key
    values.
- **Operations.**
  - Row append.
  - Slice access to keyed and ring columns.
  - `begin`, `rollback` and `commit` of an undo-log transaction (§5.4).
  - `snapshot() -> TableSnapshot` and `restore(TableSnapshot)`.
  - `logical_bytes()`, computed from column capacities by arithmetic, never per
    row.
- **P2 implementation.** `snapshot` copies the whole table into owned Arrow
  arrays. P3 replaces only what sits behind `snapshot` and `restore` (frozen
  copy-on-write pages, dirty bitmaps). Kernel code never sees the snapshot
  representation. When checkpointing is off, nothing records dirtiness.
- **No checkpointing branches.** The rolling hot path calls
  `checkpointing()` nowhere. `rg -n 'checkpointing\(\)'` over
  `operator/rolling.rs` and `operator/rolling/` returns no matches.

### 5.2 Entity slab

- **Entities.** A dense `EntityId` (u32, assigned on first appearance) indexes
  every keyed column. Exhausting u32 fails with `Operator`.
- **Rings.** History is one ring per input column (or input pair) and entity,
  shared by every window, lag and scan on that column. Its capacity is the
  largest retention: Rows `max(W, periods + 1)`, or Duration by time. This
  lands the 2026-11-08 multi-window shared history. Duration retention adds
  one i64 event-time ring per entity, shared across columns. Mixed Rows and
  Duration outputs retain by both bounds, as today.
- **Groups.** Accumulator families (§4.3), counters, extrema deques (rings of
  positions), EWMA/cumulative/indicator recurrence state, decay summaries and
  `since_rebuild` are keyed columns. A group stores no history of its own.
- **Logical size.** A Rows-frame Float64 history is at most 8 bytes plus one
  validity bit per retained sample, plus per-entity constants (24 bytes per
  sample today). The state budget (`StateBudget` rows and bytes) checks
  `logical_bytes()` plus pending rows before commit. Rejection leaves state
  bitwise unchanged.
- **Pending rows.** The reorder buffer is a log of retained input batch slices
  with typed `(event_time, EntityId, sequence)` index columns. In-order input
  that the next watermark finalizes is processed from the retained arrays
  without a row-wise copy.

### 5.3 Typed plan for every output kind

`KernelSelection::General`, `KernelComplexity::General`, the `ScalarValue`/
`BTreeMap` row path and `fallback_reason` are deleted. The plan
monomorphizes kernels over (input physical type, frame kind, group family set)
and dispatches once per batch and group, never per row.

| Input physical type                    | Ring storage                     | Kinds                                  |
|----------------------------------------|----------------------------------|----------------------------------------|
| Float64, Float32                       | native-width floats + validity   | all numeric kinds, `lag`, `min`, `max` |
| Int8–Int64, UInt8–UInt64               | i64 / u64 + validity             | all numeric kinds, `lag`, `min`, `max` |
| Boolean, Date32, Date64, timestamp[us] | u8 / i32 / i64 + validity        | `lag`, `count`, `min`, `max`           |
| Utf8, LargeUtf8                        | value-log handles + validity     | `lag`, `count`, `min`, `max`           |
| any other type                         | value-log handles + validity     | `lag`, `count`                         |

- `count` reads only validity.
- `lag` and `delta` read the ring at an offset. `delta` keeps today's
  subtraction semantics per type. Value-log outputs are emitted with
  `interleave`.
- `rank`, `quantile`, `unique_count`, `argmax` and `argmin` run their current
  algorithms over typed ring slices for every numeric type and frame. Decay
  uses its certified incremental form for Float64 Rows frames and a typed
  reference fold elsewhere.
- `difference` reads two leaf readouts. RSI, ATR and MACD use recurrence
  columns that call the unchanged `array::indicators` step functions.
- `rolling-kernels.json`, `operator/rolling/generated_kernel_manifest.rs` and
  `python/calc_flow/symbolic/_generated_rolling_kernels.py` are regenerated
  with `scripts/generate_rolling_kernel_manifest.py`. No primitive is
  `"general"`. `lag` and `delta` get a typed transition, whose name the API
  designer confirms.
- The optimizer explain line (`optimizer.py:623-645`) drops `general`,
  `profile=` and `fallback=`.

### 5.4 Transactions

- One undo-log transaction wraps each state-mutating callback: Data
  buffering, watermark finalization and end-of-input flush. Commit happens
  only after output emission succeeds. Any error, budget rejection or
  cancellation rolls back to the bitwise prior state. This includes the
  output sequence, last identity, pending rows and entity creation.
- Undo records cover:
  - each touched group's keyed row on first touch, detected by a per-entity
    transaction stamp rather than a hash set;
  - ring head and length;
  - overwritten ring slots (at most `min(run, capacity)` per entity);
  - dictionary and row appends (truncated on rollback);
  - value-log and pending-log appends;
  - scalar operator fields.
- Undo cost is bounded by touched entities, overwritten slots and appended
  rows. It is never proportional to untouched entities or untouched history.
- The same path serves every group kind, every batch size and checkpointing
  on or off. Checkpoints occur only between transactions.

## 6. Kernels and fixed costs

### 6.1 Tiers

One routing pass per finalized batch maps rows to `EntityId`s, detects
duplicate identities and picks the tier:

- **Scalar tier** (`rows ≤ T_s`, initially 16): per row, route, then
  slab, then groups, then the output slot. No sort, no pre-scan, no per-row
  allocation.
- **Full-tick tier** (`rows > T_s`, every touched entity appears exactly
  once, touched ≥ `T_f`, initially 16): updates each group's keyed columns
  across entities side by side, so the per-entity chains are independent.
- **Run tier** (otherwise): a stable counting sort by `EntityId` forms
  per-entity runs. Each run keeps its accumulators in locals and splits at
  rebuild positions. A pre-scan per run and input column (Arrow
  `null_count == 0` and all values finite, plus no NaN under the sticky
  policy) selects a branch-free loop. Otherwise the checked loop runs.
  Cancellation is checked every 8,192 rows. Kernels contain no `.await`.
- Thresholds are named constants that P2.6 and P2.7 tune within their
  measurement budget.
- The finite (batch) path, the DataFusion AVG-only route (a per-partition
  single-entity slab) and array `rolling_mean`/`RollingMeanState` call the
  same tier functions.
- Every tier produces bitwise-identical outputs and state (§4.5).
- Outputs are written directly into typed `Vec`s. A validity bitmap is built
  only once a null occurs. No per-row builder `append_option`.

### 6.2 Fixed-cost removals

- **F1.** No `RowConverter` is constructed per dispatch. Keys with several
  columns or non-primitive types use one converter built at plan construction.
- **F2.** Routing persists across batches. Typed key maps are chosen at plan
  time (single Utf8/LargeUtf8/dictionary-Utf8, single integer/temporal/bool,
  or cached converter bytes). Consecutive equal keys reuse the previous id.
  Canonical key bytes are computed only on first appearance.
- **F3.** The last identity is typed `(event_time: i64, EntityId, typed
  sequence)`. The order proof uses typed comparisons, with a fast path when
  event time is strictly increasing.
- **F4.** Endpoint and context strings are precomputed at construction
  (`operator/rolling.rs:1588`, `:1728`). There is no `format!` on success
  paths.
- **F5.** Stage observation is lock-free: no mutex inside stage scopes, at most
  one publication per callback, and no clock reads in the kernel when the
  recorder is absent.
- **F6.** A scalar-tier append has a constant number of allocations,
  independent of history size and of rows up to `T_s`.

### 6.3 Rolling metrics (`cf-api-designer` confirms)

- `RollingMetrics` keeps its three callback records and ten-stage shape
  (`docs/runtime-envelope.md:836-866`). Transitions and readouts count as
  `numeric_update`. Ring growth and log appends count as
  `history_maintenance`. Undo setup and entity creation count as
  `state_preparation`.
- Counters for deleted mechanisms are removed:
  - `copied_entities`;
  - the lane counters and `parallel_fallback_*`;
  - `numeric_general_calls`;
  - `scalar_value_conversions`;
  - `history_charge_*` and `history_rows_materialized`;
  - the decay cache and quota counters.
- New counters: `scalar_tier_calls`, `run_tier_calls`, `full_tick_calls`,
  `prescan_checked_rows` and `rebuilds`.

## 7. Checkpoint

- **Layout.** One writer layout, `ROLLING_STATE_TABLE_LAYOUT_VERSION = 7`
  (crate-private), is reported by `OperatorCheckpointCapability`. It encodes
  the whole table as Arrow IPC `StateSegment`s:
  - the entity dictionary;
  - every keyed column, with raw f64 bits;
  - each ring in logical order with validity;
  - value logs and pending rows;
  - `since_rebuild`, `k`, output sequence, last identity and last epoch;
  - fingerprints: configuration hash, input schema, and the plan with numeric
    contract id `compensated-sliding-1`.
- **Families.** Later P2 PRs may add group families to layout 7. They never
  change existing families' encodings.
- **Restore validation.** Restore checks the layout, fingerprints and column
  inventory, plus the invariants: ring length ≤ capacity, counters equal a
  recount from the rings, `k` is set iff `finite > 0`, and `since_rebuild < R`.
  Any failure fails closed before installing state. A snapshot with layout 1–6
  fails with a checkpoint-mismatch error that names the layout and says to
  start a new lineage (plan §6.2).
- **Bit-identity.** After restore, the operator continues bit-identically
  (§4.5).
- **Snapshot cost.** P2's snapshot is a synchronous whole-table encode in
  `checkpoint(&mut self, epoch)`. The indicator capture, async prepare and
  worker machinery is deleted. P3 moves this behind frozen pages and
  background persistence without kernel changes.

## 8. Deletions (by the end of P2)

- **State, kernels and codecs.**
  - `RollingKernelState`, `TypedEntityState`, `TypedWindowState`,
    `SlidingSampleQueue` and `RowsTape`.
  - The per-batch clone (`kernel.rs:1740`), `kernel/in_place.rs`,
    `kernel/mean.rs` and `kernel/numeric_rows.rs`.
  - The `stable_v2_*` rebase code (`kernel.rs:3112-3144`, `:3335-3361`,
    `:3673-3679`).
  - The general row path and `KernelSelection::General` in
    `operator/rolling.rs` and `kernel.rs`.
  - `operator/rolling/history.rs` (`RollingHistories`) and the deferred-history
    and checkpointing branches in `ordered_stream.rs`.
  - `state_v3.rs`, `state_codec.rs`, `state_codec/` and the legacy
    `checkpoint.rs`.
- **Lanes.** `kernel/entity_parallel.rs`, `kernel/entity_parallel/` and its
  tests. `runtime/streaming/entity_work.rs` and its operator-task, runner and
  supervision plumbing are deleted once rolling is its last consumer (verify
  with `rg entity_work`).
- **Indicators.** `operator/rolling/indicators/`, except the output
  compilation and validation moved into `compile.rs`, and rolling's use of
  `array::RollingWorkers`. The array provider keeps its workers.
- **Shared arithmetic.** In `array/rolling_mean_math.rs`: West add/remove,
  `rebase_due`, `StableFold` and the Kahan `CompensatedSum`. `window.rs`
  keeps its own until P7.
- **Public surface.** `RollingNumericalProfile` and every surface in §4.6.
- **Fixtures.** Rolling fixtures that pin West, `stable_v1` or `stable_v2`
  bits are rebuilt from the oracle.

## 9. PR breakdown

Each PR is independently revertible and red→green. It deletes what it
replaces and runs only its focused checks locally (§12); full regression runs
in CI. Every measurement follows §10.

| PR   | Prefix      | Content                                                         | Gate   |
|------|-------------|-----------------------------------------------------------------|--------|
| P2.1 | `test:`     | harness extensions, baseline registration                       | —      |
| P2.2 | `feat:`     | numeric contract, shared accumulator module, profile removal    | —      |
| P2.3 | `refactor:` | state table, entity slab, undo log, layout 7 for typed kinds    | —      |
| P2.4 | `feat:`     | typed plan for every output kind; delete general path, 1–4      | —      |
| P2.5 | `feat:`     | RSI/ATR/MACD on the slab; delete indicator subsystem, 5–6       | —      |
| P2.6 | `perf:`     | scalar tier and fixed-cost removals                             | G1     |
| P2.7 | `perf:`     | run tier with pre-scan, full-tick tier                          | G2, G3 |

**P2.1: harness (bench and test only).**

- Changes:
  - Add evidence workloads to
    `runtime/streaming/operator_task/performance_evidence.rs`, selected by
    `CALC_FLOW_NATIVE_EVIDENCE_CASES`: `1x1-dual`, `99x1-dual`,
    `6400x64-dual`, `64x64-dual-tick` and `6400x64-dual-lag1`. Dual means
    mean(5) and mean(20). Keep the recorder-absent and recorder-present arms.
  - Add a `native` subcommand to `scripts/profile_warm_stream.py`. It runs two
    prebuilt release evidence binaries alternately and summarizes them with
    `paired_summary`.
  - Add a `--numeric-contract-change` option that skips
    `_validate_pair_digest` (`scripts/profile_warm_stream.py:630-634`) while
    keeping oracle tolerance validation.
- Red: the script test for the missing subcommand, and an unknown case
  selector rejected.
- Measurement: register baseline G1–G3 numbers at this PR's merge commit in
  `benchmarks/rolling-kernel-v2/baseline-<sha>.{json,md}`.

**P2.2: numeric contract.** Needs API designer and critic first.

- Changes:
  - Add `accumulate/`, a crate-private module with two-sum, the families,
    rebuild and readouts.
  - Wire it into the typed kernel, the general row path, array `rolling_mean`
    and `RollingMeanState`, and the AVG-only rewrite (`DataFusionRollingKernel`,
    `operator/rolling.rs:158`, `:254-266`, with D1 as a plan flag).
  - Delete the §4.6 surface and the West/Kahan/`stable_v2` code.
  - Regenerate the schema and the
    `python/tests/fixtures/native_array_capabilities/*.json` fixtures.
  - Update examples and `benches/core.rs`.
- Restore: still rebuilds from history in this PR. Post-restore output stays
  within the bound but is not yet bit-identical; P2.3 delivers identity. The
  changed kernel and numeric fingerprints make older rolling snapshots fail
  closed.
- Red tests:
  - oracle corpus 1 and 2 are expected to exceed the declared mean and
    variance bounds under West (record the observed violation; if none
    occurs, the other red tests carry the PR);
  - a `RollingSpec` with `numerical_profile` is accepted but must be rejected;
  - `{−1e308, +1e308}` mean reads +inf but must read 0;
  - Rust batch-split identity for the new accumulators.
- Measurement (no-regression controls; a CI upper bound below 0.95 stops the
  PR): `1x1-dual` and `6400x64-dual` (evidence), and `benchmarks/test_rolling_kernel.py`
  dual SMA at `small` scale.

**P2.3: slab, undo and layout 7 for currently typed kinds.**

- Changes:
  - Add `state/table.rs` and the slab.
  - Move every currently typed group (numeric, signed, unsigned, extrema
    primitive, pair, EWMA, cumulative, Float64 Rows scans, decay) onto the
    slab, with shared rings and persisted decay summaries.
  - Apply undo transactions everywhere and write layout 7 for typed nodes.
  - The DataFusion AVG route uses a single-entity slab.
  - Delete the AoS typed state, the clone, `in_place.rs`, `mean.rs`,
    `numeric_rows.rs`, lanes and `entity_work`, deferred history,
    `RollingHistories` for typed nodes, and the typed parts of layouts 3/4.
- Red tests:
  - typed restore at a random barrier differs from the uninterrupted run;
  - `logical_bytes` for a W = 20 Float64 entity exceeds 8 bytes plus one bit
    per sample;
  - decay output differs across batch splits when the cache is refused;
  - a forced emission failure with checkpointing on and off leaves the state
    bitwise unchanged, covering mean, decay and shared rings;
  - checkpointing on and off produce identical metrics stage sets.
- Measurement:
  - `1x1-dual`, `99x1-dual` and `6400x64-dual` (evidence);
  - `6400x64-dual` with checkpointing enabled but not triggered (evidence,
    `with_checkpointing(true)`);
  - informational: a 64,000 × 64 dual-SMA runner case with
    `benchmarks/warm_stream.py` on four CPUs, where the baseline uses lanes.

**P2.4: every output kind typed.**

- Changes:
  - Type `lag` and `delta`; `count` of any type; string, bool and temporal
    extrema; `rank` and `quantile`; scans over all numeric types and Duration
    frames; difference; value logs.
  - Delete the general path, `history.rs`, `state_v3.rs`, `state_codec*` and
    `checkpoint.rs` (layouts 1–4).
  - Regenerate the manifests.
- Red tests:
  - each listed shape compiles to the typed plan (selection is reported in
    metrics and explain);
  - restore identity for these shapes;
  - their outputs stay bitwise equal to the existing general-path expected
    values, which are not regenerated.
- Measurement: `6400x64-dual-lag1` and its 1-row variant (evidence), and the
  `6400x64-dual` control.

**P2.5: indicators.**

- Changes:
  - Move RSI, ATR and MACD onto recurrence columns, writing layout 7.
    Declarations 5 and 6 are unchanged.
  - Mixed indicator nodes become typed.
  - Delete the indicator subsystem and the layout 5/6 encodings.
  - Remove the P1 session's layout 5/6 rejection
    (`runtime/streaming/session.rs:696-697`). This requires P1 merged.
- Red tests:
  - RSI/ATR/MACD restore identity;
  - a session admits an RSI + MACD node and matches the runner;
  - an indicator + mean node selects the typed plan;
  - indicator outputs stay bitwise equal to baseline.
- Measurement: `benchmarks/rolling_indicator_comparison.py` RSI(14) and MACD
  at 1 row and 6,400 × 64, against baseline, with the TA-Lib arm as reference
  only. Four cases.

**P2.6: scalar tier and fixed costs (G1).**

- Changes: the scalar tier, F1–F6, and the §6.3 metrics changes.
- Red tests:
  - a `#[cfg(test)]` converter-construction counter is non-zero per dispatch;
  - a test clock observes more than one publication per callback;
  - the allocation count of a 1-row append grows with history (harness
    allocation counter);
  - a forced scalar tier is bitwise equal to the run path.
- Measurement:
  - `1x1-dual` recorder absent (G1);
  - `1x1-dual` recorder present (overhead report);
  - `99x1-dual`;
  - a Python session 1-row push via P1's warm-stream session driver
    (informational, requires P1).

**P2.7: run and full-tick tiers (G2, G3).**

- Changes: the run tier with pre-scan, the full-tick tier, and the shared tier
  functions for finite, DataFusion and array paths.
- Red tests:
  - forced scalar, run and full-tick tiers are bitwise equal, including
    rebuild boundaries inside runs, null/NaN/Inf runs (checked loop) and
    Duration frames;
  - cancellation is observed within 8,192 rows;
  - array `rolling_mean` is bitwise equal to table `mean`.
- Measurement:
  - `6400x64-dual` recorder absent (G3);
  - `6400x64-dual` recorder present (G2 from `numeric_update`);
  - `64x64-dual-tick`;
  - the `1x1-dual` control;
  - `benchmarks/test_rolling_kernel.py` dual SMA (batch path).

## 10. Measurement

### 10.1 Protocol (every PR)

- **Registration.** Before running, register the cases (at most five), the
  baseline, the candidate, the sample counts and the 600 s budget, which
  includes startup, warm-up, correctness checks and cleanup. Report builds
  separately. Stop when the budget is exhausted: no added cases, samples or
  retries (`AGENTS.md`, Verification; `docs/benchmark-suite.md`).
- **Baselines.** Gates use the P2.1 registration. Each PR's no-regression
  controls use its parent commit.
- **Evidence cases.** Release test binaries for both arms run alternately in
  ABBA order for 10 rounds. Each round runs warm-up, then 30 timed
  dispatches per case.
  - Reported: per-arm p50 and p95, and the seeded bootstrap 95% CI of the
    median paired-round speedup.
  - "Slower" means a CI upper bound below 1.0, and "regression" below 0.95.
    A CI spanning 1.0 is inconclusive.
  - Gates use the candidate's p50 over all samples.
- **Python cases.** These reuse their harness's existing paired protocol.
  Both arms validate outputs against the oracle or tolerance. Cross-contract
  pairs use `--numeric-contract-change`.
- **Reports.** `benchmarks/rolling-kernel-v2/p2.<n>-<date>.{json,md}`.

### 10.2 Gates

The gate boundary is a release current-thread `block_on` of real
`dispatch_data` plus `dispatch_watermark_handler` for a `RollingOperator` with
dual mean(5) and mean(20). The recorder is absent unless stated, and there is
no projection (D4).

- **G1** (P2.6): `1x1-dual` candidate p50 ≤ 1 µs.
- **G2** (P2.7): `numeric_update` stage ns ÷ (6,400 × 2) ≤ 4 ns in
  `6400x64-dual` with the recorder present (plan target 2–4 ns).
- **G3** (P2.7): `6400x64-dual` candidate p50 ≤ 100 µs. The plan's reference
  is 329 µs including projection; the registered baseline is re-measured here.
- **Stop rule.** If a gate misses, the PR reports the stage breakdown and stops
  for a decision instead of expanding measurement (plan §8).

## 11. Dependencies, indicator placement and risks

- **P1.**
  - P2 builds on `feature/stream-session`. Determinism tests include a
    `StreamSession` push sequence (P1 spec §3).
  - P2.5's admission change and P2.6's Python session case need P1 merged.
  - Everything else is independent of P1. If P1's stop rule fires, P2.5 and
    P2.6 drop those two items.
- **P3.**
  - P2.3 defines the `state/table.rs` seam with a whole-table snapshot. P3
    replaces snapshot/restore internals and adds manifest v4, without kernel
    changes.
  - P2 does not wait for P3, and P3's protocol work may proceed in parallel
    (plan §7.9).
- **Indicators.** RSI/ATR (declaration 5) and MACD (declaration 6) all move
  in P2, in P2.5. Nothing about indicator state is deferred. Only P3's
  snapshot implementation applies to them later.
- **R1.** Deleting lanes (D3) can slow streaming batches of at least 64,000
  rows on many-core hosts until P6. P2.3 reports it at four CPUs.
- **R2.** The §4.4 constants are analytic and need the critic's check.
  Correlation null classification for constant windows is exact only after
  `R` constant positions.
- **R3.** P2.3 is the largest PR and is mostly deletion. Split it by family
  only if review requires, without leaving two state paths for the same family.
- **R4.** Between P2.2 and P2.3, restore is not bit-identical (as today).
- **R5.** Every visible change ships with P2: the last bits of rolling
  sum/mean/dispersion/pair outputs on batch, stream, array and SQL AVG-rewrite
  paths; overflowed windows reading ±inf; the `{−1e308, +1e308}` mean reading
  0; old rolling checkpoints failing; and `rolling_metrics` counter changes.

## 12. Acceptance criteria

- [ ] For every position of the §4.4 corpus, under both policies, Rows and
  Duration frames and integer, Float32 and Float64 inputs, classification
  equals the oracle and finite values lie within the declared bounds.
- [ ] Given random Data/watermark splits (≥ 50 seeded patterns), finite
  evaluation and a session push sequence, outputs are bit-identical. The same
  holds for checkpoint/restore at random barriers versus uninterrupted runs,
  forced tiers, and mean alone versus mean + variance.
- [ ] Kinds marked "bitwise equal to baseline" in §4.1 pass their existing,
  unregenerated expected values.
- [ ] Undo: an injected failure after state mutation (emission, budget,
  integer overflow, cancellation) leaves the snapshot bytes equal to those
  before the callback, with checkpointing on and off, for every group kind.
- [ ] A layout 1–6 snapshot, a document or IR carrying `numerical_profile`,
  and an array sequence state with mean marker 1 or 2 are each rejected with
  the documented error.
- [ ] `schemas/project-v3.schema.json` differs only as in §4.6.
  `cargo test -p calc-flow --test project_v3` passes.
- [ ] No rolling primitive is `"general"`. `rg -n 'ScalarValue|checkpointing\(\)'`
  over `operator/rolling.rs` and `operator/rolling/` has no matches. No
  `RowConverter` is constructed per dispatch (test counter).
- [ ] G1, G2 and G3 pass at §10.2 with registered reports committed. The
  P2.2–P2.5 controls show no regression (CI upper bound ≥ 0.95).
- [ ] Focused local checks per PR (full regression in CI):

  ```bash
  cargo fmt --all --check
  cargo test -p calc-flow --lib operator::rolling
  cargo test -p calc-flow --lib accumulate state::table
  cargo test -p calc-flow --test rolling_spec --test rolling_state --test project_v3
  cargo clippy -p calc-flow -p calc-flow-python --lib --tests -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
  uv run maturin develop
  JAX_PLATFORMS=cpu uv run pytest python/tests/test_rolling_numeric_contract.py python/tests/test_native_array.py -q
  uv run ruff check . && uv run ruff format --check .
  git diff --check
  ```

- [ ] Documentation is updated through `cf-doc-writer` (§13), and the Rust
  90% line-coverage floor holds in CI.

## 13. Documentation (`cf-doc-writer`, after each public-surface PR)

- `docs/rust-api.md`:
  - array `rolling_mean` (≈120–200);
  - native array plan profiles (≈526–533);
  - Rolling windows, the numeric contract and the three corner properties
    (≈940–1180);
  - RSI/ATR/MACD worker and fallback text;
  - checkpoint layouts;
  - `numerical_profile` (≈1080–1090).
- `docs/python-api.md`: `rolling_mean` (≈1341–1345) and the IR options
  (≈1483).
- `docs/symbolic-design.md#native-rolling-state`: profiles, layouts 3/4 and
  the decay transactional copy.
- `docs/runtime-envelope.md`: rolling metrics, sparse undo and private copy,
  entity-parallel lanes, and the state budget (≈836–940).
- `docs/sql-datafusion-performance.md`: the rolling rewrite boundary
  (≈262–267) and D1.
- `docs/design.md` (≈228–236).
- Rolling layout text in `docs/streaming-guide.md` and `docs/batch-guide.md`
  (≈275, 296). Grep for `stable_v`, `West`, `layout 3`/`4`/`5`/`6` and
  `lanes`, checking rolling-specific hits only.
- `CHANGELOG.md` (visible changes per §11 R5) and the plan's execution record.

## 14. Open questions

- `cf-api-designer`:
  - names of the removed and added `RollingMetrics` counters (§6.3);
  - the typed-transition name for `lag`/`delta` in the manifest;
  - D2's rustdoc rewording instead of renaming the declaration constants.
- `cf-critic`: the §4.4 bound derivations, and the pairing rule's effect on
  the bound (factor 2 on `γ_m²`).
