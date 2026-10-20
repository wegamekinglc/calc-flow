# Join Single-Thread Performance - Specification

## Source and baseline

- User request on 2026-10-20: implement the ten performance recommendations in
  `.codex/artifacts/reviews/pr482-single-thread-performance.md`.
- Implementation baseline: main
  `78b75724f408f9d46a479488b4ecc76ea17aec68`, after PR #482 merged.
- Vocabulary and contracts: [introduction](../../../docs/introduction.md),
  [ASOF Join](../../../docs/asof-join-guide.md),
  [streaming](../../../docs/streaming-guide.md), and
  [repository verification](../../../AGENTS.md#verification).
- The review profiles `aff245e4`, an older head. Its percentages locate possible
  work; they are neither current-main measurements nor promised speedups.

## Problem statement

Streaming joins spend substantial CPU on repeatedly resolving retained payloads,
inspecting source-level metadata, hashing short repeated keys, and constructing
or retiring row-level descriptors. Large default source batches make this work
visible on one CPU. The change must remove redundant work while retaining the
streaming job's admission, cancellation, event-time, output, and checkpoint
contracts and the multicore improvements already merged.

## Goals and scope

- Assess all ten review recommendations against current main, implement the
  applicable residual hotspots in this patch, and demonstrate already-satisfied
  portions with existing source and focused behavioral/work-count checks.
  Record larger unimplemented redesigns explicitly as limitations.
- Reduce redundant operations with falsifiable operation-count criteria below;
  measure the combined result against main with unchanged workloads.
- Keep changes internal to the Rust join runtime. Python application APIs,
  historical benchmark worker settings, project declarations, and state wire
  contracts remain compatible.
- Preserve new constructor defaults: 1,000,000-row source/transport batches and
  unlimited memory unless the caller supplies limits. Preserve existing
  omitted-field project-v3 compatibility defaults and fingerprints.

This work does not reopen PR #482, change historical benchmark registries or
workload scopes, add another table engine, change JavaScript integer serialization,
or promise parity with batch-only Polars/DataFusion timings. It does not introduce
a public tuning API or a mandatory memory cap.

## Current-main applicability

| Review item         | Current behavior                                                                                                                                                     | Remaining obligation                                                                                                                                 |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Source width     | `JoinOutput::constant_row_bytes` visits distinct rows and invokes column-level `side_row_bytes` repeatedly.                                                          | Cache the previous source/projection on each side without allocation; inspections scale with source transitions.                                     |
| 2. Retained lookup  | `StateRows::payload` resolves a chunk through `BTreeMap` for each coordinate.                                                                                        | Reuse resolution for consecutive coordinates in the same chunk without allocating storage proportional to coordinate gaps.                           |
| 3. Internal morsels | Deferred count/gather morsels and 64-row stack hash/position arrays already exist; the local deduplication table still scales with the complete group.               | Improve bounded block string/hash/pack work, preserve the input transaction, and record unchanged full-group capacity.                               |
| 4. Short keys       | Typed, column-bound local interning and one persistent resolution per distinct key already exist.                                                                    | Remove redundant short-string hashing/equality work without changing canonical identity.                                                             |
| 5. Eviction         | Prefix partitioning, bulk dead-chunk removal, and aggregated dictionary retirement already exist; repeated plan passes and owner/key handling remain.                | Reuse existing plan-order summaries and preserve canonical-binding fallback, deterministic retirement, and tombstone semantics.                      |
| 6. Window scan      | Both serial and owned probes already use per-key `WindowCursor`, forward galloping, disorder fallback, and reusable count windows.                                   | Preserve window reuse; append match runs in worker fill and owner assembly without a second probe engine.                                            |
| 7. Index output     | Join already uses `UInt64Array` plus Arrow `take` for one source and `interleave` for multiple sources; admitted payload cursors exist.                              | Reuse retained resolution and compact supported ASOF single-source positions; retain ownership safeguards and unsupported-schema rejection.          |
| 8. ASOF groups      | Columnar admission already counts rows per key, scatters stable coordinates into `key_rows`, and calls `merge_admitted` per reserved key.                            | Preserve this path and remove remaining unnecessary repeated descriptor/lookup construction where shown by current source; avoid redundant grouping. |
| 9. Cooperation      | Local/worker flows already prepay 64 visits and 4,096 bytes, but short strings debit per value/chunk.                                                                | Debit bounded blocks once while enforcing both quanta, including long values and collision chains.                                                   |
| 10. CPU capacity    | Native gather capacity uses `available_parallelism`; Python's native runtime uses Tokio's automatic default. Benchmark `THREADS = 32` still forces oversubscription. | Preserve native CPU-derived capacity and historical benchmark settings; use explicit identical per-experiment workers and provenance.                |

Paths in this table are under `crates/calc-flow/src/operator/`, except benchmark
defaults under `scripts/benchmark_suite/`. Existing optimizations are part of the
baseline and must not be reported as newly implemented gains.

## Functional requirements

### FR1 - Source-level width inspection

For a fixed-width, null-free output selection, use a bounded last-source cache
on each side: column-level eligibility and byte width are evaluated at most once
per consecutive source/projection run during one output preflight. Repeated
matching rows in that run reuse the result; returning after an intervening
source may inspect again. The cache introduces no allocation. Cached
ineligibility is also reused; a nullable, variable-width, unsupported, or
overflowing source follows the existing measured fallback. Borrowed immutable
owner identity distinguishes relevant source/slice/projection changes and cannot
outlive its owners. No uncharged distinct-source map is required or permitted.

Acceptance fixtures cover one source, alternating sources, sliced payloads,
empty/duplicate/reordered projections, nulls, variable-width columns, and a tight
byte budget. Range boundaries, first oversized-row error, and materialized Arrow
values must equal the existing uncached reference. For one stable source per
side the column-inspection counter is at most two, independent of matched row
count; alternating-source inspections are bounded by source transitions.

### FR2 - Retained-coordinate resolution

For `N` consecutive payload lookups touching `C` chunk runs, retained map
resolutions are bounded by `C`, rather than `N`. A run means consecutive lookups
within the same retained chunk; arbitrary backwards/cross-chunk coordinates
remain supported. No allocation may scale with the distance between the oldest
and newest chunk IDs: sparse survivors must retain storage proportional to live
chunks, not historical coordinate span.

Stable chunk/slot coordinates, row IDs, key leases, physical payload offsets,
admission order, and restore behavior are unchanged. Cache lifetimes prevent
using a chunk after mutation/removal. Tests include a removed middle chunk,
partial chunk survival, large logical ID gaps, restored chunks, and revisiting an
earlier chunk. Payload lookup must not reconstruct full `StoredRow` identities.

### FR3 - Existing bounded scratch and internal work units

An internal work unit is an implementation detail, not a new source `Batch`, edge
envelope, watermark, checkpoint cut, or independently committed admission.
`LocalIntern::intern_blocks` already reuses 64-row stack arrays for positions and
hashes; preserve this existing bounded primitive and use bounded short-string
hash/pack processing with FR9's row/byte accounting. A single large
variable-width value is itself processed with bounded cooperation. Existing
count/gather morsels and the 1M-row transport default are unchanged.

This patch does not replace the full-group local deduplication table, its
funding/clearing, or final per-input identity arrays. Their capacity remains
proportional to the complete input and must be reported as a limitation. Do not
claim reduced whole-group preparation capacity from stack-scratch reuse or
rewrite the input transaction solely to fulfill the historical morsel proposal.

Before the first output, complete the same whole-input validation, global match
limit check, mandatory state/result budget preflight, and cancellation/deadline
checks as the baseline. `max_matches_per_input_batch` counts matches across every
work unit and worker of the original input. A limit/refusal/cancellation in a
later unit leaves no partial input admission or early output. Previously
committed inputs remain intact. Optional acceleration refusal uses the existing
funded fallback and cannot turn an accepted baseline input into a mandatory
workspace failure merely to pay for the optimization.

Prepared dictionary births, key frequencies, retained rows, status, and dirty
checkpoint bookkeeping publish only at the original commit boundary. Checkpoint
and barrier processing cannot capture a partially prepared input. Within each
side, output row order and edge-budget envelope boundaries remain the reference
order and boundaries; internal scheduling does not reorder results.

Tests for changed block processing cross the 64-row boundary and compare
canonical identities, frequencies, birth order, and cancellation accounting with
the generic path. Retain existing whole-input morsel limit, refusal, output
ordering, and checkpoint regressions. Allocation/work evidence distinguishes the
existing fixed position/hash scratch, any new fixed short-key scratch, and the
unchanged complete-group local table/final identity arrays.

### FR4 - Compact local short-key work

Short-string specialization is batch-local acceleration only. Canonical framed
key bytes and hashes, type/length/null distinctions, dictionary slot resolution,
first-appearance birth order, project fingerprints, and checkpoint identities
must be byte-for-byte equivalent to the generic path. Each selected key column
is hashed once per row per local pass; equality/probing must not cause another
canonical encoding for every repeated row.

The optimized domain and length limit must be explicit and tested at its
boundary. Mixed lengths, embedded NUL, non-ASCII UTF-8, sliced arrays, multiple
columns, nulls, empty strings, unsupported layouts, forced hash collisions, and
long-value fallback retain the reference behavior. A key appearing in a short
batch, then a mixed/long batch, then after restore resolves to the same identity.
Retain 64-bit/wider counters where currently required; no unchecked narrowing.

### FR5 - Eviction summaries and safe retirement

Reuse summaries prepared in the existing expiration plan's order: row counts,
charges, tombstone eligibility, first-seen canonical retirement groups, and
chunk-coordinate groups. Prefix partitioning per sorted expiration run and
whole dead-chunk release already exist; preserve them rather than introducing a
different native-key expiration order. Expiration runs, key buckets, and physical
chunks cannot be assumed to have the same order.

Canonical binding resolution can be reused only with a stated grouping proof
that remains valid after restore/rebuild. A raw `Arc<FramedKey>` is not by itself
such proof: restored rows can resolve through `SideBindings::canonical_key` and
their leases. Retain per-row canonical resolution where necessary while removing
redundant passes; do not claim an unconditional per-key lookup bound there.

Before deleting any live row, capture all identity needed by its checkpoint
delta and acquire mandatory tombstone/control funding. Captured rows below the
delta watermark produce exactly one ordered tombstone; never-captured rows at
or above it coalesce with their existing counters. Preserve the current plan's
tombstone row order and each key's **first-appearance plan order** for dictionary
retirement/slot recycling. A key-sorted summary alone does not satisfy this
requirement. Decrement each lease by exactly the removed row count and preserve
native-index, identity-binding, retained-owner, and metric consistency.

All newly introduced fallible preparation and overflow checks happen before
removal. Cancellation before commit leaves the old state; commit cannot await
or observe an incompletely applied retirement. Test interleaved key expiration,
partial/full chunks, captured/uncaptured mixtures, ID gaps, exhausted budget,
cancelled preparation, checkpoint/restore continuation, and deterministic future
slot reuse. Work-count tests establish the removed repeated passes and the
actual fast-path/fallback canonical-resolution counts; unavoidable tombstone
identity emission remains proportional to captured expired rows.

### FR6 - Preserve forward window reuse and reduce match bookkeeping

Use the existing per-key forward-window reuse for nondecreasing probe windows;
retain the reference search when either bound regresses. Cache/probe state is
local to its immutable native-index snapshot and worker, not shared unsafely.
Run-oriented output construction must be equivalent to enumerating all existing
matches in incoming-position then native `(time, row_id)` order, including equal
timestamps, missing keys, both incoming sides, inclusive endpoints, and saturated
time bounds.

Given a run of `M` matches for one probe, selection metadata construction must
append the run without `M` calls to the per-pair selection method. It may still
write the required output indices. Cover both worker fill and owner-side
fragment assembly (`pairs::ordered_output`); rebuilding tuples at assembly would
reintroduce the removed work. Preserve identity-to-run transitions, sliced
views, and funded overlap of fragment/destination buffers. Keep the existing
4,096 probe/match-item stopping cadence during large range copies; dictionary's
64-visit quantum is not a new global probe contract. Global match counting must
complete before result fill/output and retain checked arithmetic and limit
sentinels. Existing cursor edge/seeded-sweep and owned-window cache/refusal tests
remain valid.

### FR7 - Output index representation and ownership

Reuse source-level gathers already present in main. Eligible ASOF output stores
one 64-bit row index plus validity per output row instead of a source-and-row
tuple. Preserve unmatched left rows with explicit right null validity; row zero
is a valid matched row. Native workers may promote to the original tuple layout
when a second source appears, with checks before and after each 64-position
conversion block and before publishing the completed replacement. Multi-source
output retains the original single `interleave`, without an extra full output
copy.

The Tokio owner must never perform a growing-prefix conversion. Enable its
compact layout only after a bounded immutable metadata proof: the retained pool
has at most two payload owners (the two input sides), and inspecting those
owners establishes at most one right source. Check the ID table's backing is
also bounded to those owners' minimal four-bucket capacity before iteration;
sparse historical capacity cannot turn the proof into a long scan. Otherwise
initialize the funded legacy tuple layout directly. This source-cardinality
proof is not a CPU or parallelism limit. It conservatively forgoes owner-side
compaction when a larger
pool contains only one right source; it does not scan retained rows or change
the already-compact native worker path. Tests prove owner conversion work is
zero for the fallback and preserve exact early/late-source and null ordering.

The compact representation remains consumable through ordinary and segmented
gather, range slicing, and shared-right detection. Do not unconditionally expand
it back into tuples before output. Promotion after leading unmatched rows and a
later second source must preserve all prior rows and fund live old/new overlap.

Use a narrow index only after proving all relevant source/state offsets fit;
otherwise use a checked wide representation. Synthetic index-width/overflow
tests must not allocate billions of rows. Positive ASOF coverage uses supported
flat nullable payloads, slices, timezone/schema metadata, projections,
multi-record inputs, unmatched rows, matching physical row zero, sparse indices,
and source promotion. Nested, dictionary, run-end, and view ASOF payloads remain
rejected according to the existing schema contract. Dictionary/nested/view and
zero-column positive coverage applies only to ordinary join where supported.
Preserve required compaction: output must not retain unselected giant parents
or encoded children contrary to existing owned-output contracts. Fund live
buffers, promotion peaks, and worker-owned backing until the corresponding
owners actually drop; keep existing budget refusal and cancellation-before-send
behavior.

### FR8 - Stable ASOF admission groups

Retain existing compact typed-sequence grouping and grouped right-bucket merge.
Any further removal of per-row planning must preserve accepted physical-row
coordinates, late filtering, sequence total order, stable within-key ordering,
canonical key rank ordering where observed, and current duplicate-error
precedence across records and resident state. General/canonical sequences keep
their reference fallback and ownership ledger.

Reserve/prepare all affected bucket changes before commit. State mutation cannot
partially install a key group if a later group refuses budget or is cancelled.
At least one work-count check establishes the applicable existing grouped path
or new residual improvement without claiming that grouping itself is new.
Focused equivalence fixtures cover repeated timestamps with distinct sequences,
cross-record duplicates, restored resident identities, disorder, mixed lateness,
forced optional parallel refusal, and cancellation before installation.

### FR9 - Bounded cooperative accounting

Amortize short-value bookkeeping over blocks. Account for work **before** doing
it, enforcing both existing maxima of 64 visits and 4,096 bytes between required
cooperative steps in dictionary encoding/hash/probe/equality/copy paths. Long
single values, long collision chains, and empty/null keys must not bypass either
bound. These are work-unit bounds, not a wall-clock cancellation SLA or a cap on
an indivisible Arrow kernel.

Observe job cancellation/deadline and worker abandonment at the established
ownership boundaries and again before input-state commit and output send.
Cancellation settles owned workers and releases their owners before associated
credit. Tests cancel just before/after each quantum and during a long value or
collision tail, asserting bounded observed visits/bytes, no premature output,
unchanged committed state, and complete credit refunds. Counter tests show short
string accounting calls scale by blocks rather than one per value.

### FR10 - Preserve CPU-derived production capacity and benchmark identity

All introduced production parallel capacities derive from effective CPU
availability; existing native capacities already satisfy this. Availability
failure falls back to one. Useful-work thresholds may reduce parallel work, but
must not replace available capacity with an arbitrary fixed ceiling.

Native gather and Python runtime defaults already meet the production
requirement, so this item needs evidence rather than another worker selector.
Do not silently change historical benchmark catalog `THREADS = 32`, workload
fingerprints, or stored results in this patch. For bounded experiments use
identical explicit CPU/worker settings in baseline and candidate, recording
affinity, resolved worker count/override, and engine partition settings. A
separate experiment helper may derive effective CPU availability with a portable
one-worker fallback if justified and tested; it does not become a prerequisite
or silently rewrite the maintained historical benchmark contract.

## Delivery and focused verification

Implementation proceeds in reviewable stages within this work item:

1. Source-width caching and retained payload lookup reuse (FR1/FR2), with focused
   range/gather, sparse-coordinate, and work-count regressions first.
2. Local short-key work and block cooperation (FR3/FR4/FR9), retaining canonical
   dictionary tests and recording unchanged full-group table capacity.
3. Selection run append, ASOF output compaction, and applicable residual grouped
   admission work (FR6/FR7/FR8), with
   whole-input refusal, atomicity, ordering, and ownership tests before enabling
   each path. Already-satisfied portions require evidence, not duplicate code.
4. Retirement summaries and chunk reclamation (FR5), after deterministic
   tombstone/lease/slot-order regressions cover its commit boundary.
5. Preserve already CPU-derived production capacity and historical benchmark
   identity (FR10), then complete specialist review and bounded performance
   comparison. Record each item as a residual change, already-satisfied behavior,
   or explicit out-of-patch redesign limitation. Do not present existing
   optimizations or a planned redesign as newly delivered performance work.

Each behavior change begins with its focused failing regression and records the
expected failure before implementation. Reuse adjacent test modules, including
`join/materialization.rs`, `join/retained/tests.rs`,
`join/key_dictionary/local_tests.rs`, `join/tests/morsel_tests/boundaries.rs`,
`join/tests/native_window_probe_tests.rs`,
`join/tests/native_gather_window_tests.rs`,
`join/tests/persistent_dictionary_cooperation_tests.rs`,
`join/tests/lazy_delta_tests.rs`, `asof/output_plan/tests.rs`, and
`asof/admission/identity_tests/`. New tests belong beside the affected source.

The root coordinator selects the smallest relevant Cargo test filters and
compiles affected modules once per necessary source state. Run Rust formatting
and scoped Clippy for changed Rust surfaces; Ruff and relevant script unit tests
only for changed Python benchmark logic. Full workspace, connector coverage,
cross-platform regression, and package artifact gates remain assigned to CI.
Schema and historical workload/dependency registries must have no unintended
drift. Finish with `git diff --check` and final specialist review.

## Performance acceptance

- Baseline is the sealed default-feature release from main `78b75724`; candidate
  is the final source-frozen default-feature release. Record source/tree/native
  hashes, dependencies, build settings, harness identity, workload dimensions,
  worker settings, CPU affinity, correctness oracle, and output counts.
- Before acquisition, record at most five representative cases and their link
  to changed code, exact scale, fixed samples/order, and one total 600-second
  acquisition budget including fixtures, startup, preflight, warmup, oracles,
  restore if selected, and cleanup. Estimate builds separately. The case set
  must include 1M-row/default-transport single-CPU join, ASOF, and interval, plus
  a multicore control directly exercising the changed path. The fifth case, if
  used, is a directly affected small-batch or sparse-retained control. Do not
  expand this into a type/mode/scale matrix.
- Use maintained paired statistics: two balanced rounds, ten adjacent AB/BA
  pairs per case per round, unless a maintained harness mandates an already
  documented different fixed sample contract. Ready-source enqueue to Arrow
  output is the end-to-end window. Compare identical source segmentation,
  explicit budget settings, effective CPU/worker settings, and checkpoint mode
  in both arms. Do not credit worker-count changes as algorithmic gains; test
  CPU-default selection separately if it would change paired conditions.
- Quantitative non-regression gate is the maintained `report.comparison` rule:
  both round paired-median CI lower bounds above +5% confirm regression; any
  upper bound above +5% is inconclusive; both upper bounds below -5% establish
  improvement; remaining outcomes are no confirmed regression. A confirmed
  regression blocks acceptance. Inconclusive cases are reported as uncertain,
  never equivalent or proved regression-free. No favorable resampling or budget
  extension is permitted.
- Improvement is established for an affected workload only by its maintained
  improved verdict, supported by the stated operation-count reductions. Report
  per-case P50, maintained P95, round intervals/verdicts, peak/resource evidence
  available from the harness, and unmeasured paths. Historical profile fractions
  do not establish a numerical speedup target. Do not declare the whole effort a
  measured performance improvement if every end-to-end case is inconclusive.

## Acceptance checklist

- [ ] Each of FR1-FR10 has residual implementation evidence, an existing-main
  satisfaction record, or the explicit redesign limitation defined above, with
  focused tests for changed behavior.
- [ ] Fixed-schema source inspections scale with source transitions and use no
  new allocation; consecutive retained lookups scale with chunk transitions;
  fixed 64-row scratch is distinguished from unchanged full-group capacity.
- [ ] Canonical identities, project fingerprints, checkpoint wire/continuation,
  first-output refusal, input atomicity, global match limits, deterministic
  row/tombstone/slot order, null/unmatched semantics, and credit ownership pass
  their focused equivalence and failure regressions.
- [ ] Cancellation accounting respects both row/visit and byte bounds, including
  a single long value, and catches cancellation before commit/send.
- [ ] Production capacity remains CPU-derived; historical benchmark settings,
  caller overrides, and 1M/unlimited-memory defaults remain intact.
- [ ] Named focused checks, whitespace, scoped format/lint, and final specialist
  review pass; CI and coverage status are reported accurately at handoff.
- [ ] The bounded paired report records completed cases and uncertainty, with no
  confirmed regression; any measured improvement claim meets the maintained
  statistical rule. Pending CI is recorded and does not imply merge readiness.

## Open questions

No public API or user-preference question is required to begin. The implementer
and reviewer must record selected bounded work, optimization eligibility,
unchanged whole-group capacity, and the final five-or-fewer measurement cases
before those paths are enabled/measured. These engineering choices are
constrained by FR1-FR10 and do not permit weakened contracts or all-or-nothing
replacement of already-satisfied paths.
