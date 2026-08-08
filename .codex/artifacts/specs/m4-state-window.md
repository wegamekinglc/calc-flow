# Continuous Streaming 3.0 — M4 State and Window Specification

| Field             | Value                                                                    |
| ----------------- | ------------------------------------------------------------------------ |
| Status            | **Proposed — blocks M4 implementation until review is complete**         |
| Priority          | **P0 — state layout and window semantics become durable in M5**          |
| Baseline          | `main@6275599ca3bb872ab480b677f13bcd9698b144f0`                          |
| Milestone         | M4 — incremental local state, final-only windows, and late-data policy   |
| Artifact slug     | `m4-state-window`                                                        |
| Intended audience | state, operator, runtime, compiler, benchmark, test, and review owners   |

## 1. Authority and precedence

This document is the controlling M4 delta wherever the following inputs are
silent, stale, or inconsistent with the current `main` implementation:

1. `docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`;
2. `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`;
3. `.codex/artifacts/specs/continuous-streaming-runtime.md`;
4. `.codex/artifacts/api-notes/continuous-streaming-runtime.md`;
5. the M3 delta package and the implementation merged by PR #86;
6. baseline commit `6275599ca3bb872ab480b677f13bcd9698b144f0`.

Precedence for an M4 conflict is:

```text
this M4 delta
> compatible continuous-streaming runtime specification and API note
> compatible main-plan requirements
> research recommendations and historical milestone notes
```

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD
NOT**, and **MAY** are normative.

## 2. Current-main reconciliation

The implementation baseline changes how the historical plan must be read.

| Topic                         | Current evidence                                                        | M4 interpretation                                                                                           |
| ----------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Completed predecessor         | PR #86 is latest `main` and implements M3 progress coordination         | M4 starts from typed monotonic progress, idle/reactivate, transient replay, and the existing M2 runtime     |
| Research milestone numbering  | Research §9 calls state plus windows “M3”                               | That scope is detailed-plan M4; the research and plan milestone numbers are not interchangeable             |
| Segment format                | Runtime specification D6 selected Arrow IPC                             | M4 MUST NOT reopen the Arrow IPC versus Parquet decision                                                    |
| Existing checkpoint module    | Core uses the flat file `src/checkpoint.rs` for public v2 persistence   | M4 MUST NOT create `src/checkpoint/model.rs` beside that file; v3 models live under `src/state/manifest.rs` |
| Durable coordination          | `operator_task` rejects barriers before M5                              | M4 supplies state/operator seams and a private commit harness; M5 owns runtime barrier and job recovery     |
| Operator snapshot seam        | `OperatorStateSnapshot` already carries inline JSON plus byte segments  | M4 implements bounded metadata and Arrow IPC window segments without changing that information boundary     |
| Late-data recorder            | each `StreamOperatorContext::new` currently creates a fresh recorder    | M4 moves ownership to the operator task so metrics accumulate across handler calls                          |
| Public runner                 | source-driven runtime remains crate-private                             | public A6 remains post-M5; M4 MUST NOT expose runner checkpoint, recovery, or lifecycle controls            |

## 3. Scope

M4 includes:

1. the final v3 checkpoint-manifest data model, implemented without replacing
   the v2 checkpoint store;
2. immutable state handles and an asynchronous state-backend contract;
3. a local managed-root backend with staged and committed state segments;
4. length and SHA-256 validation before decode;
5. retained-manifest reachability, orphan collection, and immutable
   compaction;
6. Arrow IPC encoding for built-in window state;
7. final-only tumbling and hopping aggregate windows;
8. assignment-level late-data dropping and persistent bounded metrics;
9. deterministic operator snapshot/restore and state-inventory round trips;
10. compile-time validation, property tests, paired benchmarks, and the
    standard 20-minute soak.

## 4. Non-goals and milestone firewall

M4 does not include:

- source barrier injection or multi-input barrier alignment;
- checkpoint scheduling, timeout, or epoch coordination in a running job;
- replacing the public v2 `CheckpointStore` or `FileCheckpointStore`;
- selecting a manifest and reopening a complete job after process failure;
- sink pre-commit, transactional commit, abort, or recovery;
- an exactly-once claim at either operator or sink boundaries;
- public `StreamingRunner` or `StreamingJob` A6 controls;
- session windows, early triggers, processing-time triggers, allowed
  lateness, side outputs, updates, retractions, or changelog streams;
- arbitrary SQL watermark expressions, incremental joins, or DataFusion
  private planner nodes;
- state compression without a separately reviewed paired benchmark;
- Python, Studio, REST, OpenAPI, or project-v3 surface changes.

M4 MAY exercise the manifest model and commit order through an internal test
harness. Such a harness is not a running-job recovery implementation and MUST
NOT be documented as one. Durable job recovery remains M5.

## 5. Controlling decisions

### D1 — The v3 manifest model is finalized in M4, but store replacement is M5

M4 MUST define the complete `CheckpointManifest` v3 data model already frozen
by the runtime API note G3. The model MUST include format version, pipeline
identity and hashes, epoch, creation time, final recovery status, source
progress, operator progress plus state handles, sink delivery/pre-commit
metadata, and deterministic state checksum.

M4 MUST place this model under `state/manifest.rs`. It MUST NOT convert the
flat `checkpoint.rs` module or replace its v2 public exports. M5.1 MUST move or
re-export the exact M4 model into the checkpoint module and wire store
operations without redefining fields or serialization.

Manifest validation MUST reject unknown fields, duplicate JSON keys, missing
required nullable fields, invalid version, over-depth, over-size, invalid
identifiers, inconsistent pipeline/operator/epoch handles, and an invalid
state checksum. Canonical serialization MUST be byte-identical across runs.

### D2 — State handles are immutable, portable, and path-safe

A `StateHandle` contains exactly:

```text
operator_id, epoch, segment_id, relative_path, byte_len, sha256
```

The handle MUST reject:

- empty or non-portable operator and segment identifiers;
- epoch zero;
- absolute paths, empty components, `.` or `..`, platform prefixes, and NUL;
- a path outside the committed subtree;
- uppercase, non-hex, or non-64-character SHA-256 values;
- a handle whose operator or epoch differs from its manifest entry;
- duplicate `(operator_id, epoch, segment_id)` identity;
- two handles that claim the same committed path with different metadata.

Raw pipeline names, operator IDs, and segment IDs MUST NOT become filesystem
path components. Managed paths use deterministic SHA-256 encodings while the
original logical IDs remain only in validated metadata.

`StateBackend` MUST open an exclusive lineage-scoped session before exposing
segment operations. The lineage key is the validated pipeline name plus
semantic fingerprint; backend identity supplies the state-root component.
Dropping the session releases its cross-process lease. This session boundary
is mandatory so the later `Arc<dyn StateBackend>` runner cannot bypass lineage
exclusivity through backend-specific downcasting.

### D3 — Segment publication precedes manifest publication

The only valid epoch publication order is:

1. acquire the lineage-exclusive lease;
2. stage every segment under the managed staging subtree;
3. finish the durable write and required file synchronization;
4. re-read as required and validate byte length plus SHA-256;
5. atomically rename every validated segment into the committed subtree;
6. synchronize the committed parent directory;
7. construct and validate a manifest that references committed paths only;
8. write, synchronize, atomically rename, and parent-synchronize the manifest
   last.

A manifest MUST NOT reference a staged, missing, unvalidated, or unpublished
segment. A failure before step 8 leaves the previous manifest authoritative.
A segment published before a failed manifest publication is an orphan and
MUST NOT influence recovery.

### D4 — The manifest is sole truth; cleanup is reachability based

Only a valid retained manifest makes state live. Directory order, modification
time, the highest segment epoch, and the presence of a staging directory MUST
NOT select recovery state.

Collection MUST be serialized against publication and MUST retain:

1. every handle reachable from every retained manifest;
2. every segment belonging to the in-flight epoch;
3. every segment newer than the latest retained manifest while publication is
   in flight.

Unknown files, symlinks, unexpected file types, lock failures, and delete
failures MUST stop cleanup without broadening the target. Cleanup MUST never
follow a symbolic link or retry with a broader path.

### D5 — Window state uses immutable Arrow IPC deltas plus compaction

The local backend treats segment bytes as an opaque validated envelope. The
built-in window operator MUST encode its state as Arrow IPC file format. This
preserves the already public custom-operator snapshot boundary, whose segment
bytes remain operator-owned and length/checksum protected.

Window checkpoint state consists of deterministic immutable delta segments:

- upsert rows replace the accumulator for one exact window/group key;
- tombstone rows delete one exact window/group key;
- restore applies referenced segments in ascending `(epoch, segment_id)`
  order;
- later operations for the same key win;
- duplicate operations for one key inside one segment are invalid;
- every segment carries a state-layout version and the expected logical
  schema fingerprint in operator metadata.

Only dirty keys are emitted into a normal checkpoint delta. Compaction reads a
complete referenced inventory and writes one new immutable base segment. A
manifest after compaction references the new base plus only later deltas. Old
segments remain live while any retained manifest references them.

Compaction MUST trigger before the predicted manifest exceeds its byte limit.
It MAY also trigger by a configurable delta-count threshold. Compaction MUST
not change logical rows, scalar bit patterns, output schema, or deterministic
row order.

### D6 — State capture is cheap on the async executor

`StreamOperator::checkpoint()` MUST perform O(number of dirty keys/segments)
metadata work and MUST NOT bulk-encode state on a Tokio executor thread. The
window operator MUST maintain an incrementally prepared state representation
or hand owned work to a blocking boundary before the synchronous capture.

Filesystem access, checksum calculation over large buffers, Arrow IPC bulk
encoding/decoding, restore reads, and compaction MUST run through true async
I/O, `spawn_blocking`, or a registered dedicated worker. M4 MUST add a paused-
time control-responsiveness test for a large dirty set.

### D7 — Window geometry is fixed and checked

Windows use UTC microseconds since the Unix epoch and half-open intervals
`[start, end)`. The origin is exactly Unix epoch zero. Tumbling windows have
`slide == size`. Hopping windows require `size = m * slide` for integer
`m >= 1`; each non-null row receives exactly `m` concrete assignments. M4
sets `MAX_WINDOW_OVERLAP = 1_024`; a larger `m` fails compilation so one input
row cannot create unbounded assignment work.

Assignment uses Euclidean floor division. Checked arithmetic MUST reject a
window start or end outside `EventTime`'s `i64` range rather than wrap. A row
exactly at an end belongs to the next aligned window.

Input timestamps may be Arrow second, millisecond, microsecond, or nanosecond
timestamps and MUST use the existing checked/floored `EventTime` conversion.
Timezone-naive means UTC; explicit `"UTC"` is accepted; every other timezone
is rejected during compilation.

A null event-time row has no concrete window assignment. It is dropped while
valid rows in the same batch continue, increments `null_event_time_rows`, and
increments `null_event_time_batches` once for that input batch. It is not late
and MUST NOT affect late counters or maximum lateness.

### D8 — Group-key encoding and equality are identical

Group columns are interpreted in declared order. Supported key types are:

- Boolean;
- signed and unsigned integers through 64 bits;
- Float32 and Float64;
- Utf8 and LargeUtf8;
- Date32 and Date64;
- timestamp microseconds, timezone-naive or `"UTC"`.

The order-preserving encoding is the G1 encoding from the runtime API note.
Null sorts before non-null. The complete encoded key MUST NOT exceed 64 KiB.
Oversize keys fail the operator transaction with the group-column path and no
partial accumulator mutation.

Grouping equality is byte equality of the stable encoding. Consequently
`-0.0` and `+0.0` are distinct groups, and NaN payload/sign bit patterns are
distinct groups. The same bytes determine in-memory key identity, Arrow IPC
state order, compaction order, and final output order.

### D9 — Aggregate functions use an explicit first-version matrix

`count(column)` counts non-null values and returns `UInt64`. There is no
`count(*)` spelling in the first M4 API.

| Function | Accepted input                                                                | Output                                  |
| -------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| `count`  | every Arrow type accepted by the input schema                                 | `UInt64`                                |
| `sum`    | `Int8..Int64`, `UInt8..UInt64`, `Float32`, `Float64`                          | `Int64`, `UInt64`, or `Float64`         |
| `min`    | numeric types above, Boolean, Utf8/LargeUtf8, Date32/Date64, Timestamp(us)    | same logical type as the input          |
| `max`    | numeric types above, Boolean, Utf8/LargeUtf8, Date32/Date64, Timestamp(us)    | same logical type as the input          |
| `avg`    | `Int8..Int64`, `UInt8..UInt64`, `Float32`, `Float64`                          | `Float64`                               |

Decimal128, Decimal256, binary, nested, dictionary, interval, duration, and
other unlisted combinations are rejected at compilation. This is the first-
version decimal decision; later decimal support requires a specification
amendment defining precision/scale growth and overflow.

All aggregates ignore null inputs. For a materialized group/window containing
rows but no non-null aggregate input, `count` is zero and `sum`, `min`, `max`,
and `avg` are null.

Integer count and sum operations use checked arithmetic. Signed sums widen to
an internal `i128` and unsigned sums to `u128`, but MUST fail before committing
an update that cannot be represented by the declared output type. Integer
averages use checked widened sum plus `u64` count and convert deterministically
to `Float64` at finalization.

Float32 values widen to Float64 for sum and average. Processing order is the
runtime's accepted per-ingress FIFO row order and is independent of input
RecordBatch partitioning. Every NaN entering float sum/average, and every
indeterminate infinity operation, produces one canonical quiet Float64 NaN.
Float min/max use the same IEEE total order as the stable key encoding and
preserve the selected scalar's bits. Infinity is otherwise accepted.

An overflow or invalid aggregate update fails the whole input-batch operator
transaction. No accumulator, dirty set, metric, or emitted output from that
batch may become visible.

### D10 — Output schema and order are deterministic

Output columns are, in order:

1. `window_start: timestamp(us, "UTC")`;
2. `window_end: timestamp(us, "UTC")`;
3. group columns in declared `group_by` order;
4. aggregate columns in declared aggregate order.

Group and aggregate declaration order is semantic and therefore fingerprinted.
Graph/node insertion order is not semantic and MUST NOT change the compiled
schema or fingerprint. Duplicate group columns, duplicate aggregate output
names, and collisions with `window_start`, `window_end`, or group names fail
compilation.

One close event emits rows ordered by `(window_start, window_end,
stable_group_key)`. Watermark close events remain ordered by strict watermark
advance. Empty windows emit nothing. The operator task supplies its minimum
effective output-edge budget through a crate-private context seam; output is
chunked into deterministic contiguous key ranges. A single output row larger
than that budget fails before enqueue rather than creating an unbounded staging
buffer.

Every emitted chunk uses the stable operator ID as `BatchMetadata.source` and
a checked operator-owned sequence beginning at zero. The next output sequence
is part of operator snapshot/restore state. Chunk boundaries, metadata
sequences, and empty attributes MUST be deterministic across replay of the same
committed prefix.

### D11 — Late data is classified per assignment and recorded once

For the current input watermark `WM_in`, a concrete assignment is late if and
only if:

```text
assignment.window_end <= WM_in
```

No row-level `event_time <= watermark` comparison is allowed. One hopping row
may have both closed and open assignments; only closed assignments are
dropped, and open assignments update state normally.

Per operator, task-owned cumulative metrics are:

- `late_rows`: dropped row-window assignments, not distinct physical rows;
- `affected_batches`: input batches with at least one dropped assignment;
- `max_lateness_micros`: maximum `WM_in - window_end` over dropped
  assignments;
- `null_event_time_rows` and `null_event_time_batches` from D7.

The recorder MUST be shared across every context created for the same operator
task. Counter updates MUST be checked and transactional with the handler
result. Payloads, keys, event times, and source cursor values MUST NOT appear
in metric labels or logs.

### D12 — Final-only emission and state lifecycle preserve replay

On a strict input-watermark advance, the window operator emits every newly
closed window before the runtime forwards that watermark. Within one
fault-free run it emits each window at most once. When every ingress ends, it
flushes every still-open non-empty window exactly once before EndOfInput is
forwarded.

Closed accumulators remain in an emitted-but-not-checkpointed set. A subsequent
operator snapshot records the required state transition and may emit
tombstones for state already represented by earlier handles. If staging or
manifest publication later fails, M5 terminates the job and recovery uses the
previous manifest, whose state can replay and re-emit the result. M4 MUST NOT
require a checkpoint-complete callback absent from the frozen operator trait.

Operator-layer replay MAY re-emit a previously visible window. M4 guarantees
identical final values, schema, and deterministic order after snapshot/restore
or compaction. Exactly-once external visibility exists only after M5 adds a
transactional or unbounded epoch-idempotent sink boundary.

## 6. Implementation work packages

### WP0 — Reconcile documents and freeze the delta

Files:

- `.codex/artifacts/specs/m4-state-window.md`;
- `.codex/artifacts/api-notes/m4-state-window.md`;
- `.codex/artifacts/critiques/m4-state-window.md`;
- `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`.

Exit gate: all critique blockers are resolved, every decision maps to a named
acceptance criterion, and no implementation file changes.

### WP1 — Manifest, handle, and state inventory

Files:

- new `crates/calc-flow/src/state/{mod,backend,manifest,segment}.rs`;
- modify `crates/calc-flow/src/lib.rs` and `error.rs` only for approved M4
  exports/errors;
- new `crates/calc-flow/tests/state_backend.rs`.

Implement the exact model and pure validation first. Do not touch the v2
checkpoint module. Add unit tests beside pure state code and public integration
tests under `crates/calc-flow/tests/`.

### WP2 — Local backend, retention, and compaction

Files:

- new `crates/calc-flow/src/state/local.rs`;
- new `crates/calc-flow/tests/local_state.rs`;
- modify workspace/core `Cargo.toml` files only for the reviewed
  MSRV-compatible advisory-lock dependency;
- modify `crates/calc-flow/benches/core.rs`.

Implement managed-root leasing, staging/publish/load, reachability collection,
immutable compaction, crash-point harnesses, and incremental/restore/compaction
benchmarks.

### WP3 — Window specification and compilation

Files:

- new `crates/calc-flow/src/operator/window.rs`;
- modify `operator/mod.rs`, `pipeline/compile.rs`, and `pipeline/stream.rs`;
- new `crates/calc-flow/tests/window_compile.rs`.

Implement constructor validation, output schema, configuration serialization,
fingerprint inputs, `NodeOperator`/`CompiledStreamOperator` variants, and
batch-mode rejection before execution logic. Add `checkpoint` and `restore`
dispatch to `CompiledStreamOperator` so M5 can reach the already frozen trait
methods, but do not call that dispatch from the M4 barrier-rejecting task.

### WP4 — Window execution and operator state

Files:

- extend `operator/window.rs`;
- modify `runtime/streaming/{operator_task,metrics,runner}.rs`;
- new `crates/calc-flow/tests/window_{tumbling,hopping,properties}.rs`.

Implement transactional batch updates, stable key encoding, incremental
accumulators, assignment-level late dropping, handler-before-forward emission,
EOF flush, dirty deltas, snapshot/restore, and task-owned metrics.

### WP5 — Evidence and merge readiness

Files:

- new `.codex/artifacts/analysis/m4-state-window-implementation-evidence.md`;
- update runtime documentation only where behavior is implemented.

Run all focused and repository gates, paired benchmarks, stress/property
tests, and one exact-head 20-minute soak. Record commands, exact commit, raw
artifact hashes, sample counts, queue/task terminal state, RSS slope, and any
approved variance.

## 7. Acceptance matrix

### State and manifest

- **AC-01** rejects every invalid handle path/ID/checksum case in D2.
- **AC-02** canonical manifest serialization is byte-identical for different
  map insertion orders.
- **AC-03** unknown, duplicate, missing, over-depth, over-size, and wrong-
  version manifest fields fail closed.
- **AC-04** source/operator/sink IDs and every handle must match the manifest
  pipeline and epoch before any segment load.
- **AC-05** state checksum mismatch fails before restore side effects.
- **AC-06** staged segments are invisible to manifest-based selection.
- **AC-07** checksum and length mismatch fail before Arrow or custom decode.
- **AC-08** a state larger than 10 MiB round-trips while the manifest contains
  handles only and stays below its bound.
- **AC-09** crash injection at every D3 step selects exactly the previous or
  newly completed manifest, never a partial epoch.
- **AC-10** a committed but unreferenced segment is ignored and collected.
- **AC-11** collection preserves retained, in-flight, and post-latest segments.
- **AC-12** symlink, unexpected-file, lock, and delete failures stop cleanup
  without expanding its target.
- **AC-13** compaction changes neither logical state nor deterministic order.
- **AC-14** retained historical manifests remain loadable after later
  compaction.
- **AC-15** delta-count/manifest-size thresholds compact before the manifest
  bound can be exceeded.

### Window compile contract

- **AC-16** zero, sub-microsecond, overflowed, and non-integral size/slide
  configurations fail with the exact field path.
- **AC-17** hopping `size % slide != 0` or overlap above 1,024 fails
  compilation.
- **AC-18** missing/non-timestamp event-time columns and non-UTC timezones fail
  compilation.
- **AC-19** duplicate group/aggregate names and reserved-name collisions fail.
- **AC-20** every D9 accepted aggregate/type pair produces the exact output
  type.
- **AC-21** every unlisted aggregate/type pair, including Decimal128/256,
  fails before source open.
- **AC-22** session, early, allowed-lateness, side-output, update, and retract
  options have no accepted representation.
- **AC-23** declared group/aggregate order affects schema and fingerprint;
  graph insertion order does not.
- **AC-24** the complete window/state layout configuration changes the semantic
  fingerprint.
- **AC-25** batch compilation rejects the stream-only window operator.

### Assignment, aggregation, and emission

- **AC-26** pre- and post-epoch tumbling assignment follows Euclidean floor
  division, including exact-boundary rows.
- **AC-27** a hopping row receives exactly `size / slide` assignments in
  deterministic start order.
- **AC-28** assignment start/end overflow fails without state mutation.
- **AC-29** null event-time rows are dropped, separately counted, and do not
  affect late metrics.
- **AC-30** null aggregate inputs follow D9, including all-null groups.
- **AC-31** integer count/sum overflow fails the entire input batch without
  partial mutation.
- **AC-32** float NaN, infinity, signed zero, min/max, sum, and average follow
  D8/D9 bit-level rules.
- **AC-33** a composite group key over 64 KiB fails without partial mutation.
- **AC-34** incremental results equal a finite batch group-by oracle.
- **AC-35** randomized RecordBatch partitioning preserves results and scalar
  bit patterns for the same ordered rows.
- **AC-36** one close event emits deterministic `(start, end, key)` order,
  chunk boundaries, and checked operator-owned BatchMetadata sequences.
- **AC-37** a window emits only when `window_end <= WM_in` and before the
  forwarded watermark.
- **AC-38** all-ended flush emits every remaining non-empty window once and
  forwards one EndOfInput.
- **AC-39** empty windows emit no rows.

### Late data and state lifecycle

- **AC-40** a row at 10:15 remains accepted for `[10:00, 11:00)` when the
  watermark is 10:30.
- **AC-41** a row assigned to `[10:00, 11:00)` is dropped when the watermark is
  11:05 and records five minutes of lateness.
- **AC-42** a hopping row with closed and open assignments drops only the
  closed assignments.
- **AC-43** affected-batch count increments once even when many assignments in
  the batch are late.
- **AC-44** metrics accumulate across multiple operator contexts and use the
  stable operator ID only.
- **AC-45** metric overflow fails deterministically rather than wrapping.
- **AC-46** a late or null row payload never appears in status, metrics, errors,
  or logs.
- **AC-47** normal snapshots encode only dirty keys; restore from the complete
  handle inventory reproduces exact state.
- **AC-48** closed state remains replayable until represented by a snapshot
  transition.
- **AC-49** snapshot/restore at arbitrary batch boundaries produces identical
  final output.
- **AC-50** replay may emit again but converges to identical final values and
  order.
- **AC-51** compaction between snapshot and restore does not affect output.
- **AC-52** large-state checkpoint capture does not bulk-encode on a Tokio
  executor and does not starve the next control handler.

### Regression and delivery gates

- **AC-53** M2 lifecycle, cancellation, task ownership, backpressure, fan-out,
  and terminal cleanup tests remain green.
- **AC-54** M3 watermark, idle/reactivate, progress trace, receipt, and exact-
  coordinate transient replay tests remain green.
- **AC-55** public runner, Python, REST, OpenAPI, project schema, and generated
  TypeScript surfaces have no M4 diff.
- **AC-56** paired benchmarks report incremental write, full restore,
  compaction, tumbling, and hopping costs; unrelated paths stay within the 5%
  evidence gate.
- **AC-57** the exact-head 1,200-second two-source slow-sink soak records 120
  ten-second samples after the defined warm-up, bounded queues/state, no task
  leak, and no sustained upward RSS trend.
- **AC-58** full repository checks, coverage, docs, generated-contract,
  supply-chain, Copilot, Codacy, and review-thread gates pass on the final SHA.

## 8. Required implementation order

Every behavior change follows:

1. add one focused failing test and record the expected RED reason;
2. implement only the smallest behavior needed for that test;
3. run the focused test to GREEN;
4. refactor without changing proven behavior;
5. run the affected regression group;
6. update the AC-to-test evidence mapping;
7. run the full merge gates after the final diff is stable.

WP1 precedes WP2. WP1 and WP2 precede window state persistence in WP4. WP3
may begin after D7–D10 are approved, but WP4 may not merge without WP1/WP2.
WP5 runs only on the exact final implementation head.

## 9. Pull-request and merge strategy

M4 uses two independently reviewable PRs:

1. a docs-only delta package containing this specification, the matching API
   note, the critique, and the reconciled main plan;
2. one M4 implementation PR with WP1–WP5 as narrow commits.

The implementation PR remains draft until AC-01–AC-58 are mapped to named
tests or explicit equivalent gates. Any push invalidates earlier remote CI,
review, Codacy, and soak evidence. Merge requires the exact final SHA to be
green, review threads resolved, and GitHub merge state `MERGEABLE/CLEAN`.

## 10. Schedule and completion definition

The detailed plan's single-engineer estimate remains six to nine engineer-
weeks:

| Work package | Expected share | Completion signal                                         |
| ------------ | -------------- | --------------------------------------------------------- |
| WP0          | 0.5–1 week     | approved documents with zero blockers                     |
| WP1–WP2      | 2–3 weeks      | state scale, crash, retention, compaction gates pass      |
| WP3          | 1–1.5 weeks    | complete compile contract and fingerprint tests           |
| WP4          | 2–3 weeks      | oracle, property, late, snapshot, and runtime tests pass  |
| WP5          | 0.5–1 week     | exact-head full gates, benchmarks, and soak pass          |

M4 is complete only when the implementation PR is merged. Merging this
documentation package approves an implementation contract; it is not M4
implementation evidence.
