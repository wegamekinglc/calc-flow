# P2 Rolling kernel v2 — specification

Date: 2026-11-15. Source baseline: `main` `39cab2fb` (rolling, array and
benchmark sources are unchanged on `feature/stream-session`). Branch:
`feature/rolling-kernel-v2`. Plan:
[streaming core re-architecture](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/docs/plans/2026-11-15-streaming-core-rearchitecture.md),
phase P2 (§3.6, §4.1, §4.2, §4.4, §5, §7.3, §7 gates, §8). Status:
revision 4, with the `cf-critic` critique applied; `cf-api-designer` still
confirms public removals (§4.6), metrics (§6.3) and the trait in §6.4 before
the PR series in §9.

Revision 2 (user decision, 2026-11-15) moves P1's ≤ 20 µs session gate into
P2. P2 now also covers:

- the synchronous operator trait for Rolling and Expression on the session
  path (§6.4);
- gate G4, session latency (§10.2);
- gate G5, a zero-allocation push with bounded footprint (§6.5).

Revision 3 renames the session gates to G4 and G5. Only the session-path
steps depend on #618 (P1).

Revision 4 applies the numeric-oracle and G5 critique: family-local overflow
rebuilds, NaN-preserving dispersion readouts, exact rational bound enclosures,
and a one-output rolling graph whose recycling is limited to batch shells.

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
- User decision, 2026-11-15: P1 continues, and its ≤ 20 µs gate moves into
  P2. The gate is the warm-stream session case-1 p50 in normal-GC mode;
  forced-GC is reported but not gated. Rolling and the projection/expression
  node move to the synchronous operator trait of plan §4.1 on the session
  path. A per-call allocation and footprint budget joins P2's acceptance.

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
- **D4**: G1–G3 are evaluated by direct synchronous calls on the
  `RollingOperator` (§10.2). The plan's 6.1 µs and 329 µs figures included the
  projection, so the baseline is re-measured at this boundary. The projection,
  session and Python costs are gated end to end by G4.
- **D5**: the synchronous operator trait is crate-private in P2.
  - It covers `RollingOperator` and `ExpressionOperator`.
  - The runner drives both through an adapter in their existing async
    operator tasks. Their async `StreamOperator` implementations are deleted.
  - The public async `StreamOperator` trait (external operators), Union and
    every other operator are unchanged. Exposing the trait publicly and
    migrating the remaining operators is P7.
- **D6**: Expression shapes that compile to a row-local program take the sync
  path. These are column projections and the `CompiledExpression` steps
  (`datafusion_template/compiled.rs:25-87`), compiled once at preparation from
  the exact input schema. Shapes with UDFs or plans that do not compile keep
  an async implementation. A session runs them through its existing
  `block_on` adapter; they are outside the G4 graph.

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

P1's session measurement (`feature/stream-session` at `3676a502`,
`benchmarks/stream-session/p1-20261115-summary.md`) put the case-1 1-row
dual-SMA push p50 well above the ≤ 20 µs gate:

| Mode                        | Case-1 push p50 | Native call share   |
|-----------------------------|-----------------|---------------------|
| forced GC (pre-registered)  | 152.9 µs        | 126.9 µs            |
| normal GC                   | 90.5 µs         | 75.0 µs             |
| hot tight loop (diagnostic) | 18.6 µs         | —                   |

The bridge is cheap: a no-op session call costs 0.46 µs. The rest is native
operator dispatch on the async path, and it is sensitive to cache state:

- `block_on` over async handlers per push (`runtime/streaming/session.rs:430`
  at `3676a502`);
- `StreamOperatorContext::for_task` built per callback
  (`runtime/streaming/operator_task.rs:2041`, `:2101`);
- per-call `Vec`/`BTreeMap` construction (`session.rs:416`, `:436`) and the
  expression's per-call step vector (`datafusion_template/compiled.rs:72`);
- large async state machines;
- the rolling data/finalize work listed above.

Python argument handling (10.9 µs) and result assembly (5.1 µs) were the
remaining normal-GC phases.

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
6. Rolling and compilable Expression nodes on a synchronous operator trait.
   A session whose graph is all-sync runs a push with no async handler,
   future, `block_on` or per-call context construction (§6.4).
7. A steady-state 1-row session push makes no transient heap allocation. Its
   only allocations are the returned result. Its footprint is bounded and
   does not grow (§6.5).
8. Gates (§10):
   - G1: native 1-row rolling append ≤ 1 µs;
   - G2: ≤ 4 ns per window-row;
   - G3: 6,400 rows × 64 entities ≤ 100 µs;
   - G4: warm-stream session case-1 p50 ≤ 20 µs in normal-GC mode;
   - G5: zero transient allocations per steady-state 1-row push, with a
     bounded per-push footprint.

Non-goals:

- Migrating Union, SQL, window, join, ASOF, cross-section or any other
  operator to the sync trait, and making the trait public (P7).
- Changing runner topology, edges or the public async `StreamOperator`
  trait. Changing P1's session API, admission or edge budget rules.
- Snapshot pages, dirty bitmaps, background persistence and manifest v4
  (P3). P2 keeps the current barrier protocol and manifest v3.
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

Each window group is identified by `(input column or input pair, frame)`. It
carries only the accumulator families its readouts need, chosen at plan time. All
families in a group share validity; pair counters count pairwise-valid
positions. A readout's bits never depend on which other outputs share its
group.

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
- **Shift choice.** A family's shift is set by its first finite valid sample
  inserted into an empty group. At each rebuild of that family it is the
  newest finite valid sample in the window.
- **Per-row order.** For each finalized row of an entity and each group:
  evict, reset if `finite == 0`, insert, `since_rebuild += 1`, rebuild if due,
  then read out. Reset therefore precedes both the insert and the counter
  increment, allowing the insert to set `k`. This order is the same in every
  tier.
- **Rebuild.** A full group rebuild is due when `since_rebuild` reaches `R`.
  It resets every float sum to `+0.0`, re-chooses each family's shift, folds
  the window's finite valid samples oldest-to-newest with the single-operand
  rule, and sets `since_rebuild = 0`. For Rows frames, `R = max(W, 64)` with
  `W` the frame size. For Duration frames, `R = max(n_prev, 64)`, where
  `n_prev` is this group's own frame row count immediately after the previous
  full rebuild (64 before the first), never the shared ring length. A full
  rebuild updates `n_prev`. Counting every finalized entity row, valid or not,
  makes the cadence independent of batch splits.
- **Family-local overflow rebuild.** If any float sum in one family is
  non-finite, only that family is reset and refolded as above, with its own
  shift re-chosen. Other families, the group's `since_rebuild` and `n_prev`
  are unchanged. In particular, dispersion overflow cannot change mean bits.
- **Reset.** When `finite` reaches 0, all float sums reset to `+0.0`, family
  shifts become unset, `since_rebuild = 0` and `n_prev = 64`.
- **Readouts.** Every compensated readout uses
  `read(s, e) = s` if `s` is non-finite, else `fl(s + e)`.
  - Finite sum: `F = read(s, e)`.
  - Infinity term: `I` is +inf, −inf or NaN from `pos_inf` and `neg_inf`,
    and 0 when neither is present.
  - `sum = F` when `I = 0`, else `fl(F + I)`. Sticky NaN reads NaN.
  - `mean = fl(sum / valid)`.
  - `D = read(ds, de)`, `Q = read(qs, qe)`,
    `T = fl(Q − fl(fl(D·D)/n))`, and `M = 0` only when `T < 0` (which is
    false for NaN); otherwise `M = T`.
  - `variance = fl(M / (n − ddof))` and `stddev = fl(√variance)`. When finite
    inputs make `Q = +inf`, these read +inf if `T = +inf`, or NaN if the
    subtraction produces NaN (including `inf − inf`); NaN is never clamped to
    0.
  - Pair readouts use the same guarded `D`/`Q` rule for `Mx` and `My`, and
    `C = fl(Pxy − fl(fl(Dx·Dy)/n))`, where `Pxy = read(ps, pe)` for the
    compensated `Σdx·dy`. `covariance = fl(C / (n − ddof))`. `correlation`
    reads null when `Mx == 0` or `My == 0`, else
    `fl(C / fl(fl(√Mx)·fl(√My)))`, with no clamping, as today
    (`kernel.rs:4249-4257`).
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
Except for exact integer `sum` and the integer-`mean` numerator, its exact
quantities use the converted binary64 samples `x = fl(input as f64)`, not the
raw integers. It replays per-entity frame membership (Rows and Duration), the
§4.2 policies, the §4.3 reset/rebuild/shift schedule and each family's operand
sequence since its last rebuild or reset. From these it computes exact `S`,
`n`, `μ`, `Q_K` and co-moments, the bound terms below, and the in-domain
classification.

Every irrational in an exact value or bound is represented by an
outward-rounded rational enclosure, including `√(Q_K/n)`, `σ`,
`√(Q_Kx·Q_Ky)`, `r` and the correlation denominator. Scaled integer square
roots (for example `math.isqrt` on a value scaled by `2^2k`) provide root
enclosures; `σ` and `r` use two-sided enclosures. Every final absolute bound
is rounded upward, so acceptance uses only exact rational comparisons.

Outside the numeric domain, expected classification is instead obtained by a
binary64 replay of §4.3 in Python floats. This covers overflow to ±inf and
NaN, whose exact `Fraction` value need not have the same classification.

Correlation uses a three-valued in-domain rule:

- **Must-null** iff every x operand, or every y operand, folded, inserted or
  evicted since that family's last rebuild/reset equals its shift `k`.
- **Must-finite** iff `ΔMx < Mx` and `ΔMy < My`. Its value is checked only
  when the correlation bound's stronger premise below holds.
- **Either** otherwise: null or finite is accepted, and a finite value is not
  checked.

**Notation.** `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`, `γ_k = k·u/(1 − k·u)`. At position
`p`, `r ≤ p` is the family's last rebuild or reset. `O` is the operand
sequence of one compensated sum since `r`: the rebuild fold, then each insert
and each negated eviction, where a paired row counts as two operands. `m = |O|`
and `A = Σ|o|` over `O` (exact). `A_D`, `A_Q` and `A_P` are `A` for the
computed binary64 operands `fl(x−k)`, `fl(d·d)` and `fl(dx·dy)`, respectively:
form those operands in Python floats, then sum their exact `Fraction`
magnitudes. `K` is the shift, `Q_K = Σ(x − K)²` over the window's finite valid
samples, `M = Σ(x − μ)²`, and `n` is the valid count.

**Domain.** The bounds apply to finite windows with no infinity and no sticky
NaN, when `A`, `A_Q`, `A_P` and `n·Q_K` are at most 2¹⁰⁰⁰. Outside the domain,
outputs are deterministic and classified by the binary64 replay but unbounded.

Declared bounds on `abs(computed − exact)`:

- **sum**: `u·abs(S) + γ_m²·A`.
- **mean**: `3u·abs(μ) + 3γ_m²·A/n + η`, or `3u·abs(μ)` for integer input.
- **variance**:
  `Δv = [12u·Q_K + 3γ_m²·A_Q + 5γ_m²·A_D·√(Q_K/n) + 4m·η] / (n − ddof) + u·v`.
- **stddev**: `min(√Δv, Δv/σ) + u·(σ + √Δv)`, or `√Δv + u·√Δv` when `σ = 0`.
- **covariance**: `[12u·√(Q_Kx·Q_Ky) + 3γ_m²·A_P +
  5γ_m²·(A_Dx·√(Q_Ky/n) + A_Dy·√(Q_Kx/n)) + 4m·η] / (n − ddof) + u·abs(c)`.
- **correlation**: when `ΔMx ≤ Mx/2` and `ΔMy ≤ My/2`,
  `2(1+γ₄)·ΔC/√(Mx·My) + 2·abs(r)·(ΔMx/Mx + ΔMy/My) + 6u·abs(r)`;
  otherwise only the three-valued classification is asserted.
- **difference**: `Δa + Δb + u·(abs(d) + Δa + Δb)`, with `Δa` and `Δb` the
  leaf bounds.

`ΔC`, `ΔMx` and `ΔMy` are the bracketed numerators of the covariance and
variance rows, before division by `n − ddof`. The sum term follows
Ogita–Rump–Oishi's `Sum2` bound; pairing does not require an extra factor 2.
The first-order constants come from rounding `d`, `q`, `D·D/n` and the final
subtraction. A corpus violation of an applicable bound is an implementation
defect unless the critic shows the bound itself is wrong.

**Acceptance wording.** For every in-domain corpus position, the computed
output has the oracle's classification, using the three-valued correlation
rule. A finite output with an applicable bound satisfies the outward-rounded
rational bound relative to the exact value. Out-of-domain classification equals
the binary64 replay. Outputs are bit-identical (`f64::to_bits`) across the
determinism axes in §4.5.

**Corpus** (seeded, ≤ 100,000 rows in total, runs in seconds):

1. `1e9 + N(0, 1e−3)`, W = 20, 10,000 positions.
2. Alternating `±1e16` mixed with unit values.
3. Magnitudes `1e−300..1e300` with random signs for sum/mean; dispersion and
   pair cases use per-family caps so `A_Q`, `A_P` and `n·Q_K` stay in domain.
4. Integers near `2⁵³` and at the i64/u64 extremes for `sum` and `mean`.
5. Null, NaN and ±Inf injections under both policies.
6. Constant stretches of at least `W − 1 + R` positions, so the window itself
   is constant for `R` positions, including correlation's null classification.
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
  positions), EWMA/cumulative/indicator recurrence state, decay summaries,
  `since_rebuild` and per-group Duration `n_prev` are keyed columns. A group
  stores no history of its own.
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
- Thresholds are named constants that P2.8 and P2.9 tune within their
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
- `send_wait` is measured by the runner adapter (§6.4) after the synchronous
  call returns. Sessions record no rolling metrics (P1 spec §6.2).

### 6.4 Synchronous operator trait and session drive (`cf-api-designer` confirms names)

The trait lives in the crate-private module `operator/sync_operator.rs` (D5).
Its shape:

```rust
pub(crate) trait SyncStreamOperator: OperatorMetadata + Send {
    /// One-time preparation; may await (e.g. DataFusion planning). Never per call.
    async fn prepare(&mut self, cx: &PrepareContext<'_>) -> Result<()>;
    fn on_data(&mut self, ingress: usize, batch: Batch, cx: &mut SyncContext<'_>, out: &mut Emitter) -> Result<()>;
    fn on_watermark(&mut self, watermark: EventTime, cx: &mut SyncContext<'_>, out: &mut Emitter) -> Result<()>;
    fn on_end(&mut self, cx: &mut SyncContext<'_>, out: &mut Emitter) -> Result<()>;
    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot>;
    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()>;
    fn reset(&mut self) -> Result<()>;
}
```

- **`SyncContext`.** The driver builds it once per node and updates it in
  place per call. It carries:
  - the precomputed operator id and input watermark;
  - the ingress progress snapshot and output budget;
  - the late-metrics sink and the optional lock-free rolling recorder;
  - a reference to the job's cancellation token.

  It replaces the per-callback `StreamOperatorContext::for_task`
  (`operator_task.rs:2041`, `:2101`) for sync nodes. The `checkpointing()`
  flag is not exposed.
- **`Emitter`.** A per-node outbox with retained capacity, addressed by port
  index. Port, kind and exact-schema validation keep today's errors, with an
  `Arc::ptr_eq` schema fast path. No string lookup or `format!` on success.
  On the session path no context, collector or emitter is constructed per
  call.
- **Batch-shell recycling.** After the consumer finishes with a spent
  intermediate batch, the session driver returns its shell to the producer's
  `Emitter`. Only uniquely owned batch shells are recycled: the `RecordBatch`
  column `Vec`, `Arc<[RecordBatch]>` (via `Arc::get_mut` and a zero-column
  placeholder swap), and the cost cache (via `OnceLock::take`). Arrow value
  buffers and typed array handles are not recycled: rebuilding their control
  blocks would allocate. Computed columns consumed inside the graph therefore
  cannot satisfy G5a without `unsafe`; the gated graph emits its only computed
  column to the caller (§6.5). Passed-through columns remain shared `Arc`s.
- **Behavior parity.** Errors, panics and ordering are as today. Each
  callback's outputs precede the forwarded control (plan §4.1).
- **Cancellation.** Checked at entry and every 8,192 rows. Kernels contain no
  `.await` (§6.1).
- **Shared control rules.** Input-watermark computation, late-port
  watermark/idle suppression, end-of-input fan-out and output frontiers move
  out of `runtime/streaming/operator_task.rs` into synchronous functions. The
  operator task and the session driver both call them; the logic is not
  copied.
- **Runner adapter.** `CompiledStreamOperator` gains sync variants. The
  existing async operator task calls the sync methods. It then sends the
  outbox through the existing edges, awaiting capacity, and forwards control.
  Runner-visible behavior is unchanged, including P1's session/runner parity.
  The one difference: a callback's outputs are buffered until the call
  returns, bounded by that callback's output (§11 R7).
- **Session drive.**
  - Session nodes are either sync, called directly, or async, run through
    P1's existing `block_on` adapter.
  - For a graph whose nodes are all sync, `push`, `advance` and `finish` do
    not enter a Tokio context, poll a future or call `block_on`. The session
    keeps no Tokio runtime after preparation.
  - Internal edges are in-place queues with retained capacity. P1's
    validation, oversize rule (`EnvelopeCost` computed once per emitted
    batch, without allocating), failure semantics and the in-runtime
    rejection are unchanged.
- **Migrated operators.**
  - `RollingOperator`: all output kinds.
  - `ExpressionOperator`: the compilable shapes of D6.
    - It compiles its row-local program at `prepare`.
    - It keeps per-call step state such as filter coalescers across calls
      instead of rebuilding it (`datafusion_template/compiled.rs:72`).
    - It evaluates every record batch of a table payload in one call, checking
      cancellation per record batch instead of `yield_now`
      (`compiled.rs:76`).
    - Output bits equal today's async route.
  - Their async `StreamOperator` implementations are deleted for these shapes,
    along with the expression route's lazy template capture
    (`datafusion.rs:678-685`) when nothing else uses it.

### 6.5 Gate G5: zero-allocation push and bounded footprint

**The G4/G5 graph** matches P1's harness shape: a Rust `StreamSession` with one
rolling output, `difference(mean(5), mean(20))`, followed by a column-`select`
projection. The projection performs no arithmetic, and no computed column is
consumed inside the graph. The session is warmed with 4,096 history rows, then
at least the largest group `R` unmeasured 1-row pushes to exercise the scalar
path, then 1,000 measured steady-state 1-row pushes with explicit per-append
watermarks.

**Measurement** uses the existing thread-local `allocation_counter` seam
(`runtime/streaming/operator_task/performance_evidence.rs:230-234`). For each
per-push measurement, the harness keeps that push's input batch handle and
returned output alive inside the measured scope, so the push frees nothing
created before it.

- **G5a, zero transient heap allocations.** `count_total(push) ==
  count_current(push)`: every allocation made during the push is still live
  when it returns, and all of it is owned by the returned `SessionOutput`.
  - Nothing is allocated and then freed inside the push: state, scratch,
    routing, validation, metadata stamping, contexts, emitters and collectors,
    queues, metrics, futures, or intermediate batch shells.
  - Only intermediate batch shells are recycled (§6.4).
  - The same test, on the same thread, asserts zero `block_on` calls and
    future polls.
- **G5b, result-only allocations.** `count_total(push) == count_total(probe)`
  and `bytes_total(push) == bytes_total(probe)`.
  - The probe runs in the same process and thread and builds only the returned
    output, using the engine's cached/shared `BatchMetadata` and the same
    container construction. For each returned batch it builds a `RecordBatch`
    from the same schema. Passed-through `ArrayRef`s are reused. Each newly
    computed column is built from a `Vec` of the emitted length, plus a null
    buffer only when it has nulls. The batch is wrapped as a table `Batch`
    inside one `SessionOutput`.
  - Equality leaves no allowance for hidden engine allocations. If the engine
    allocates less than the probe, the probe is tightened.
  - **Why the push total is not literally zero.** P1's `push` returns
    `SessionOutput` by value, and the new result arrays are heap objects
    handed to the caller. G5a and G5b make every other allocation zero.
- **G5c, bounded per-push footprint.**
  - Bytes allocated per push, and its peak net bytes (`bytes_max`), are at
    most `bytes_total(probe)`. For G4 that is one f64 per output row per
    computed column plus a fixed container cost. It does not depend on history
    length (4,096 vs 65,536 rows), window (20 vs 4,096) or push count.
  - Over 10,000 pushes that drop each output, net live bytes stay at 0.
  - Before measuring, construct and drop one identical session on the same
    thread to prime process-global lazy statics. Then take the `bytes_current`
    baseline, construct the warmed session, and build the warm-up inputs
    inside that measured scope. Drop the harness's warm-up input handles
    before the final reading; buffers retained by session pending slices stay
    counted. The per-push keep-alive rule does not apply to this measurement.
  - The warmed session holds at most 256 KiB of net live heap beyond logical
    history and pending bytes. Pending bytes are logical rows × row width,
    not allocator-resident size.
  - A structural assertion checks that the session runtime field is `None`;
    the graph also retains no DataFusion `SessionContext` or worker pool.
  - The 256 KiB gate applies to the CI test build. The release evidence arm
    reports the same quantities, not necessarily the same absolute numbers.
- **Stability and scope.** Results are identical across 1,000 consecutive
  measured pushes. The same quantities are reported, not gated, for 99-row and
  6,400 × 64 pushes, whose scratch may grow only on first use.
- **Where it runs.** G5 is a set of deterministic CI tests in
  `runtime/streaming/session/tests.rs`. The release evidence session arm
  reports the same quantities.

## 7. Checkpoint

- **Layout.** One writer layout, `ROLLING_STATE_TABLE_LAYOUT_VERSION = 7`
  (crate-private), is reported by `OperatorCheckpointCapability`. It encodes
  the whole table as Arrow IPC `StateSegment`s:
  - the entity dictionary;
  - every keyed column, with raw f64 bits;
  - each ring in logical order with validity;
  - value logs and pending rows;
  - `since_rebuild`, per-group Duration `n_prev`, family shifts `k`, output
    sequence, last identity and last epoch;
  - fingerprints: configuration hash, input schema, and the plan with numeric
    contract id `compensated-sliding-1`.
- **Families.** Later P2 PRs may add group families to layout 7. They never
  change existing families' encodings.
- **Restore validation.** Restore checks the layout, fingerprints and column
  inventory, plus the invariants: ring length ≤ capacity, counters equal a
  recount from the rings, family shifts `k` are set iff `finite > 0`, and
  `since_rebuild < R`, with Duration `R` recomputed from the persisted
  per-group `n_prev`.
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
- **Async handlers.** The async `StreamOperator` implementations of
  `RollingOperator` (`operator/rolling.rs:1701-1933`), with
  `prepare_checkpoint_async` and `restore_async`, and of the compilable
  `ExpressionOperator` shapes (`operator/expression.rs:405-442`).
  - Also deleted: the per-callback context construction for sync nodes, and
    the stream route's lazy template capture once unused.
- **Fixtures.** Rolling fixtures that pin West, `stable_v1` or `stable_v2`
  bits are rebuilt from the oracle.

## 9. Implementation commit sequence

Each step is an independently revertible commit or small commit group in the
single integrated re-architecture PR. There are no per-step PRs. Each step is
red→green, deletes what it replaces, and runs only its focused checks locally
(§12); full regression runs in CI. Every measurement follows §10.

| PR   | Prefix      | Content                                                         | Gate           |
|------|-------------|-----------------------------------------------------------------|----------------|
| P2.1 | `test:`     | harness extensions, baseline registration                       | —              |
| P2.2 | `feat:`     | numeric contract, shared accumulator module, profile removal    | —              |
| P2.3 | `refactor:` | state table, entity slab, undo log, layout 7 for typed kinds    | —              |
| P2.4 | `feat:`     | typed plan for every output kind; delete general path, 1–4      | —              |
| P2.5 | `feat:`     | RSI/ATR/MACD on the slab; delete indicator subsystem, 5–6       | —              |
| P2.6 | `refactor:` | sync operator trait, session sync drive, Expression migrated    | —              |
| P2.7 | `refactor:` | Rolling on the sync trait; delete its async handler             | —              |
| P2.8 | `perf:`     | scalar tier, fixed costs, zero-allocation push, Python bridge   | G1, G4, G5     |
| P2.9 | `perf:`     | run tier with pre-scan, full-tick tier                          | G2, G3         |

Ordering and the P1 dependency:

- P1 is already integrated in the single re-architecture PR.
- P2.1–P2.4 do not depend on additional P1 work and can start now.
- P2.5 and the session-path steps, P2.6–P2.8, build on the integrated session.
  P2.9 follows P2.8.
- P2.6 needs only P2.1 and the integrated session, so it can proceed in
  parallel with P2.2–P2.5.
- P2.7 needs P2.5 and P2.6. Lanes, the indicator worker and the async
  checkpoint hooks are already gone by then.
- The sync-trait migration is P2.6 (trait, session drive, Expression) plus
  P2.7 (Rolling). G4 and G5 are gated in P2.8.

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
  - Add a shared `--numeric-contract-change` option to native and session
    comparisons that skips `_validate_pair_digest`
    (`scripts/profile_warm_stream.py:630-634`) while keeping oracle tolerance
    validation. Native P2.2 controls and every session comparison whose arms
    straddle P2.2 must enable it.
- Red: the script test for the missing subcommand, and an unknown case
  selector rejected.
- Measurement: register the G1–G3 baselines at the dispatch boundary, at this
  PR's merge commit, in `benchmarks/rolling-kernel-v2/baseline-<sha>.{json,md}`.
  The G4 baseline and first informational G5 values are registered by P2.6.

**P2.2: numeric contract.** Needs API-designer confirmation; the numeric
critique is applied.

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
  - Rust batch-split identity for the new accumulators;
  - mean-only and mean + variance stay bit-identical across a dispersion
    overflow rebuild, and overflowed dispersion readouts follow §4.3's
    +inf/NaN rule without clamping NaN to 0.
- Measurement (no-regression controls; a CI upper bound below 0.95 stops the
  PR): `1x1-dual` and `6400x64-dual` (evidence, with
  `--numeric-contract-change`), and `benchmarks/test_rolling_kernel.py`
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
    (`runtime/streaming/session.rs:696-697`). This builds on the integrated
  StreamSession.
- Red tests:
  - RSI/ATR/MACD restore identity;
  - a session admits an RSI + MACD node and matches the runner;
  - an indicator + mean node selects the typed plan;
  - indicator outputs stay bitwise equal to baseline.
- Measurement: `benchmarks/rolling_indicator_comparison.py` RSI(14) and MACD
  at 1 row and 6,400 × 64, against baseline, with the TA-Lib arm as reference
  only. Four cases.

**P2.6: sync operator trait, session drive, Expression.** Builds on the
integrated StreamSession.

- Changes:
  - Add the §6.4 trait, `SyncContext`, `Emitter` with batch-shell recycling,
    and the runner adapter.
  - Extract the shared control functions from `operator_task.rs`.
  - Session nodes become sync or async.
  - Expression's compilable shapes (D6) move to the trait. Their async route
    and lazy template capture are deleted.
  - Add the evidence session arm: the Rust G4/G5 graph (§6.5). It reports
    push latency and the G5 quantities, including the probe.
  - Register the G4 baseline at this PR's parent: warm-stream case 1 in
    normal-GC and forced-GC modes. P2.6 also records the first absolute G5
    values as informational; no parent comparison is required because this PR
    introduces the session arm, and G5's gates are absolute.
- Red tests:
  - a `#[cfg(test)]` counter records `block_on` or future polls during
    pushes of an all-sync session graph (input → exact projection), and must
    record zero;
  - sync Expression output is bitwise equal to the DataFusion SQL route for
    column projections, compiled projections and filters;
  - a UDF expression still runs in a session (through the async adapter) and
    in the runner;
  - P1's session/runner parity tests and the existing runner expression
    tests stay green, unmodified.
- Measurement:
  - warm-stream session case 1, normal GC, `drivers=session:session` against
    the registered G4 baseline, with `--numeric-contract-change` when the arms
    straddle P2.2;
  - a runner 1-row control (`drivers=runner:runner`, regression below 0.95);
  - a runner 6,400 × 64 control.

**P2.7: Rolling on the sync trait.**

- Changes:
  - `RollingOperator` implements only the sync trait, including late side
    output through the `Emitter`.
  - The runner drives it through the adapter, with `send_wait` measured by the
    adapter.
  - Delete its async implementation.
- Red tests:
  - the all-sync counter records zero for rolling → exact projection;
  - P1's late side-output test (session vs runner, `_cf_late_*` columns)
    passes on the sync path;
  - existing rolling runner and checkpoint tests stay green.
- Measurement:
  - warm-stream session case 1, normal GC, against the registered G4 baseline,
    with `--numeric-contract-change` when the arms straddle P2.2;
  - the sync `1x1-dual` (G1 boundary, informational until P2.8);
  - a runner 6,400 × 64 control (regression below 0.95 stops the PR).

**P2.8: scalar tier, fixed costs, allocations, bridge (G1, G4, G5).**

- Changes:
  - the scalar tier, F1–F6, the §6.3 metrics changes, and the §6.5
    allocation and footprint work;
  - an allocation-free session drive: metadata stamping through shared
    `Arc`s, queues and outboxes with retained capacity, and no per-call
    `Vec`/`BTreeMap` except the returned `SessionOutput`;
  - the Python bridge (`python/calc_flow/session.py` and
    `crates/calc-flow-python/src/session.rs`):
    - input validation is an identity or fingerprint check against the
      cached schema, not a full schema comparison per push;
    - metadata normalization is skipped when the batch already carries the
      session's normalized metadata;
    - results are assembled with the cached output schema.

  The public API and errors are unchanged. Phase targets, reported but not
  gated: argument handling ≤ 4 µs, native call ≤ 10 µs and result assembly
  ≤ 3 µs, in normal GC.
- Red tests:
  - G5a, G5b and G5c (§6.5) fail;
  - a `#[cfg(test)]` converter-construction counter is non-zero per dispatch;
  - a test clock observes more than one publication per callback;
  - a forced scalar tier must be bitwise equal to the run path;
  - Python: a push with an equal-but-not-identical schema object still passes
    the full validation path, and a mismatched schema still raises
    `ValueError`.
- Measurement:
  - warm-stream session case 1, normal GC (G4, with the TA-Lib reference arm
    and `--numeric-contract-change` when the arms straddle P2.2);
  - the same case with forced GC (reported);
  - sync `1x1-dual` recorder absent (G1);
  - warm-stream session case 2 (99 rows) and case 3 (6,400 × 64), normal GC
    (reported).

**P2.9: run and full-tick tiers (G2, G3).**

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
- **Baselines.**
  - G1–G3 use the P2.1 registration.
  - G4 uses the P2.6 registration, made at P2.6's parent: P1's session over
    the pre-sync operators. G5 is absolute; P2.6's first measurements are
    informational rather than a parent baseline.
  - Each PR's no-regression controls use its parent commit.
- **Evidence cases.** Release test binaries for both arms run alternately in
  ABBA order for 10 rounds. Each round runs warm-up, then 30 timed
  dispatches per case.
  - Reported: per-arm p50 and p95, and the seeded bootstrap 95% CI of the
    median paired-round speedup.
  - "Slower" means a CI upper bound below 1.0, and "regression" below 0.95.
    A CI spanning 1.0 is inconclusive.
  - Gates use the candidate's p50 over all samples.
- **Python cases.** These reuse their harness's existing paired protocol.
  Both arms validate outputs against the oracle or tolerance.
- **Cross-contract comparisons.** Every native or warm-stream session
  comparison whose arms straddle P2.2 uses `--numeric-contract-change`. This
  skips only digest equality; oracle/tolerance validation still runs.
- **Warm-stream session cases.** These use P1's `scripts/profile_warm_stream.py
  compare` protocol and configuration (P1 spec §8;
  `stream-session-measurement.md` on `feature/stream-session`):
  - one rolling `difference(mean(5), mean(20))` output followed by a column
    `select`, with no arithmetic in the projection;
  - 4,096 history rows in one segment, checkpointing off;
  - `SourceProvidedWatermarks`, with a per-append watermark at the last
    appended timestamp;
  - 30 AB/BA pairs in persistent workers, `TOKIO_WORKER_THREADS=4`;
  - seed `20261115` and the harness bootstrap (seed 771031, 20,000
    resamples).

  Baseline and candidate wheels are sealed with `profile_warm_stream.py
  build`. The baseline is the P2.6 registration wheel. Normal-GC and
  forced-GC runs are separate invocations of `--gc-modes`, inside one 600 s
  budget.
- **Reports.** `benchmarks/rolling-kernel-v2/p2.<n>-<date>.{json,md}`.

### 10.2 Gates

- **G4** (P2.8): warm-stream session case 1 under the P1 protocol
  (`stream-session-measurement.md` on `feature/stream-session`): one rolling
  `difference(mean(5), mean(20))` output, history 4,096, 1-row append, a
  column-`select` projection and explicit per-append watermark. It is run as
  `--case name=1,append_rows=1,entities=1,drivers=session:session,reference=ta_lib,p50_limit_us=20`
  with `--gc-modes normal`, plus `--numeric-contract-change` whenever the arms
  straddle P2.2, and the candidate session p50 must be ≤ 20 µs.
  - The same case with `--gc-modes forced` is reported with its phase timers
    (argument handling, native call, result assembly) and is not gated.
  - P1 baselines (`benchmarks/stream-session/p1-20261115-summary.md` on
    `feature/stream-session`): 90.5 µs normal GC, 152.9 µs forced GC and
    18.6 µs hot loop.
- **G5** (P2.8): G5a–G5c (§6.5) pass as CI tests. G5a is zero transient
  allocations. G5b limits allocations to the returned result (probe). G5c
  bounds the per-push footprint and forbids growth. The release evidence
  session arm reports the same quantities.
- **Boundary for G1–G3.**
  - The candidate is measured by direct synchronous `on_data` plus
    `on_watermark` calls on a `RollingOperator` with dual mean(5) and
    mean(20). It has a reused `SyncContext` and `Emitter`, the outbox is
    cleared inside the timer, and there is no projection (D4).
  - The P2.1 baseline predates the trait, so it uses real `dispatch_data`
    plus `dispatch_watermark_handler` under a current-thread `block_on`.
  - G1–G3 are absolute. The cross-boundary ratio is informational.
  - The recorder is absent unless stated.
  - Ordering keeps every gate measurable: G1's sync boundary exists from
    P2.7, G2's stage attribution from P2.8, and the G5 CI tests may start with
    P2.6 once #618 is available.
- **G1** (P2.8): `1x1-dual` candidate p50 ≤ 1 µs.
- **G2** (P2.9): `numeric_update` stage ns ÷ (6,400 × 2) ≤ 4 ns in
  `6400x64-dual` with the recorder present (plan target 2–4 ns).
- **G3** (P2.9): `6400x64-dual` candidate p50 ≤ 100 µs. The plan's reference
  is 329 µs including projection; the registered baseline is re-measured here.
- **Stop rule.** If a gate misses, the PR reports the stage breakdown and stops
  for a decision instead of expanding measurement (plan §8). For G4 the report
  also carries the phase timers and the G5 quantities.

## 11. Dependencies, indicator placement and risks

- **P1 (#618).**
  - P2.5 lifts P1's indicator rejection.
  - P2.6–P2.8 modify P1's session.
  - G4 and G5 measure P1's session with P1's harness (session driver,
    `--gc-modes`, per-arm drivers).
  - Those steps start only after #618 merges. P2.1–P2.4 do not depend on it.
  - The spec itself needs no rebase. Implementation PRs branch from `main`.
  - Determinism tests add a `StreamSession` push sequence once #618 is on
    `main` (P1 spec §3).
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
- **R2.** The §4.4 constants and oracle rules incorporate the critic's review.
  Constant-window correlation is must-null only after the window itself has
  been constant for `R` positions; ambiguous intermediate classifications use
  the three-valued rule.
- **R3.** P2.3 is the largest PR and is mostly deletion. Split it by family
  only if review requires, without leaving two state paths for the same family.
- **R4.** Between P2.2 and P2.3, restore is not bit-identical (as today).
- **R5.** Every visible change ships with P2: the last bits of rolling
  sum/mean/dispersion/pair outputs on batch, stream, array and SQL AVG-rewrite
  paths; overflowed windows following §4.3's ±inf/NaN classification; the
  `{−1e308, +1e308}` mean reading 0; old rolling checkpoints failing; and
  `rolling_metrics` counter changes.
- **R6.** P1's normal-GC Python phases already total 16 µs: 10.9 µs argument
  handling and 5.1 µs result assembly. G4 can miss even with a near-zero
  native call, so P2.8 owns the bridge (§9). If G4 misses, the stop rule
  applies, with phase timers and G5 quantities.
- **R7.** The runner adapter buffers one callback's outputs before sending,
  where today it sends chunk by chunk under capacity waits. Peak memory per
  callback grows by at most that callback's output columns. Input columns stay
  shared `Arc`s.
- **R8.** G4's normal-GC mode includes the harness's IPC and validation cache
  pressure. The levers are G5's footprint bound and the removal of transient
  allocations. P1's hot-loop result (18.6 µs on the async path) suggests the
  budget is achievable only with that footprint work.

## 12. Acceptance criteria

- [ ] For every position of the §4.4 corpus, under both policies, Rows and
  Duration frames and integer, Float32 and Float64 inputs, in-domain
  classification follows the oracle (including correlation's three-valued
  rule), finite values satisfy applicable outward-rational bounds, and
  out-of-domain classification matches binary64 replay.
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
- [ ] For an all-sync session graph (rolling → exact projection),
  `push`/`advance`/`finish` record zero `block_on` calls and future polls
  (test counter). `RollingOperator` and the compilable `ExpressionOperator`
  shapes have no async `StreamOperator` implementation. P1's session/runner
  parity tests pass unmodified.
- [ ] G5a, G5b and G5c (§6.5) pass as CI tests.
- [ ] G4 passes in normal-GC mode with the forced-GC run reported. G1, G2 and
  G3 pass at §10.2. Registered reports are committed. The P2.2–P2.7 controls
  show no regression (CI upper bound ≥ 0.95).
- [ ] Focused local checks per PR (full regression in CI):

  ```bash
  cargo fmt --all --check
  cargo test -p calc-flow --lib operator::rolling
  cargo test -p calc-flow --lib accumulate state::table
  cargo test -p calc-flow --lib runtime::streaming::session operator::expression
  cargo test -p calc-flow --test rolling_spec --test rolling_state --test project_v3
  cargo clippy -p calc-flow -p calc-flow-python --lib --tests -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
  uv run maturin develop
  JAX_PLATFORMS=cpu uv run pytest python/tests/test_rolling_numeric_contract.py python/tests/test_native_array.py python/tests/test_stream_session.py -q
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
- `docs/runtime-envelope.md`:
  - rolling metrics, including `send_wait` measured by the adapter;
  - sparse undo and private copy;
  - entity-parallel lanes;
  - the state budget (≈836–940);
  - the session path: sync nodes, no `block_on` for all-sync graphs, and the
    async adapter for the remaining nodes;
  - the runner adapter's per-callback output buffering.
- `docs/streaming-guide.md` and `docs/python-api.md`: session latency
  expectations and the expression shapes that stay async (D6).
- `AGENTS.md` Rust architecture bullets: built-in Rolling and Expression run
  on the crate-private synchronous trait, and the public `StreamOperator`
  stays async. Mirror the change in `CLAUDE.md` guidance only if it restates
  the bullet.
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
  - D2's rustdoc rewording instead of renaming the declaration constants;
  - the §6.4 trait, context and emitter names, and whether `prepare` stays
    async or moves to plan compilation.
