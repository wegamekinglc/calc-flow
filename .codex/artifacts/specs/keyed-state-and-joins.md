# P4/P5 Keyed State, Grouped Aggregates, Join and ASOF — Specification

Date: 2026-11-16. Source inspection base: `f09d4e45bface589d27a8e0c7ff0d28190ed3722`.
Delivery target: the single integrated PR #619; P4 and P5 are ordered commit groups, not separate PRs.
Status: proposed for API design and adversarial critique before implementation.

## 1. Source and evidence

Normative inputs:

- [Approved streaming-core re-architecture](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/docs/plans/2026-11-15-streaming-core-rearchitecture.md), especially decisions 3, 4 and 6; §§3.5, 4.2–4.3, 4.6, 6, 7.5–7.6 and 7.9.
- [Domain vocabulary](../../../docs/introduction.md), [streaming SQL contract](../../../docs/streaming-guide.md#stream-ownership-and-sql-boundaries), and [ASOF contract](../../../docs/asof-join-guide.md).
- Join evidence: [single-thread results](../analysis/join-single-thread-performance.md), [remaining gap analysis](../analysis/join-remaining-gaps-2026-10-17.md), and [Issue 406 diagnosis](../analysis/issue406-eight-workload-diagnosis.md).
- Grouped-state evidence: [SUM-state index](../../../benchmarks/sum-state/README.md), [withdrawn key sharding](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/benchmarks/sum-state/sum-key-sharding-20261107.md), [withdrawn delta delegation](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/benchmarks/sum-state/sum-delta-delegation-20261107.md), [withdrawn composite-key prototype](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/benchmarks/sum-state/sum-columnar-keys-20261105.md), and the [core-count sweep](../../../benchmarks/stock-positions/engine-core-sweep-20261107.md).

Verified current-source anchors at the inspection base:

- SQL compact state is `Vec<Group>` plus `HashMap<Arc<[u8]>, usize>`; each group owns scalar state and a reservation (`crates/calc-flow/src/operator/sql/incremental.rs:95-144`). Updates create per-group candidates (`incremental.rs:984-1042`), yield in inner loops (`incremental.rs:447-500`, `1792-1822`, `1911-1933`), and emit all groups (`incremental.rs:1954-2167`). Dirty bits currently serve checkpoints (`operator/sql/dirty.rs:3-63`), while only key columns are cached across snapshots (`operator/sql/snapshot_keys.rs:1-111`).
- Join has row payloads and persistent key identities (`crates/calc-flow/src/operator/join.rs:2803-2953`), an SQL equality-probe fallback (`join.rs:4342-4453`), per-row/4 KiB async quanta (`join/columnar/owned_copy.rs:64-114`), and row-oriented retained chunks (`join/retained.rs:10`, `333-405`). Existing pair order is incoming position then opposite `(event_time, row_id)` (`join.rs:466-484`). Eviction is strict `time + extension < opposite_watermark` (`join.rs:7423-7467`).
- ASOF already has ordered right-side columns (`crates/calc-flow/src/operator/asof/state/right.rs:26-49`) but retains separate canonical encodings, key dictionaries and eight fixed storage shards (`asof/state.rs:83-96`, `asof/state/key_dictionary.rs:31-82`, `asof/state/key_shards.rs:3-10`). Its strict frontier and emission lifecycle are at `asof/mod.rs:362-397` and `495-603`.
- Window still uses `BTreeMap<WindowKey, AccumulatorRow>` (`crates/calc-flow/src/operator/window.rs:334-400`); its pane migration is P7, not this specification.

The historical figures are problem evidence, not fresh measurements of this branch: Join 1M × 64 at 78.5 ms, interval Join at 672 ms per 1M-row side, ASOF at 84.0 ms, and grouped SUM at 1.67 s for 10M rows. The SQL batch comparison has different timing boundaries. Withdrawn sharding, lanes and DataFusion delta delegation establish that serial routing/copies, state round-trips and snapshot/progress work must be removed together; they are not options to revive here.

## 2. Problem statement

Grouped SQL, Join and ASOF each maintain a different key identity, transaction, accounting and checkpoint stack. Their hot paths dispatch per row or per small chunk, copy state for atomicity, and rebuild excessive output. The result is poor single-core latency and no sound basis for P6 key partitioning. P4 must establish one keyed-state and columnar group-state core and move compact aggregates onto it; P5 must rebuild Join, interval Join and ASOF on that same core.

## 3. Fixed decisions and goals

The approved plan gates and decisions are unchanged:

- One new architecture; no dual production paths after each operator's phase boundary.
- Old checkpoints are not restorable. The new P3 manifest v4/state pages must nevertheless recover new state with bitwise-identical continuation outputs.
- `unsafe_code = "forbid"` remains in force. Vectorization comes from safe, typed, contiguous loops; P6 owns wheel target changes.
- Full cumulative aggregate snapshots remain the default. Changelog output is an explicit semantic opt-in.
- P4 final single-core gate: grouped SUM, 10M rows, p50 at most **0.5 s**.
- P5 final single-core gates: Join 1M p50 at most **30 ms**; interval Join with 1M rows per side p50 at most **350 ms**.
- Delivery accumulates in PR #619. Each step below is a commit with focused red/green tests; no step creates an independent integration branch or PR.

Goals are testable outcomes:

1. One crate-private keyed core supplies vectorized hashing, exact collision equality, persistent group IDs, columnar key storage, leases and state-page integration to compact SQL, Join and ASOF.
2. Compact aggregate state is typed columnar state indexed by group ID; batch atomicity uses undo rather than per-group candidates.
3. Default snapshots reuse clean output chunks. Changelog consumers can reconstruct exactly the same cumulative relation as default snapshots.
4. Join and ASOF retain columnar batch runs, probe through native indexes and materialize with Arrow `take`/`interleave`; no Join SQL probe tier remains.
5. Output order, resource refusal, cancellation, checkpoint recovery and state identity are deterministic and covered by independent oracles.

## 4. Scope and non-goals

In scope:

- Rust core keyed-state primitives, state-table declarations, undo integration, typed aggregate state and output caches.
- Compact streaming SQL, finite/stream Join and interval Join, and finite/stream ASOF implementations using those primitives.
- The changelog declaration through Rust, project-v3, symbolic Python and schema inference; associated generated schema/stub updates where the binding surface changes.
- Focused Rust/Python tests, maintained measurement harness selections, evidence, and user-visible documentation through `cf-doc-writer`.

Out of scope:

- P2 rolling arithmetic/kernel work and P3 protocol implementation themselves. Their merged interfaces are dependencies.
- P6 sharded executors, a general morsel pool, CPU-feature wheels and multicore scaling gates.
- P7 window panes, distinct, cross-section, order and snapshot-SQL migration. The P4 core must not pre-create unused window/distinct operators or placeholder modules.
- General outer joins, forward/nearest ASOF, unbounded history, aggregate retractions caused by input deletes, user-defined aggregate support, SQL `DISTINCT` aggregates, or a SQL probe fallback.
- Old manifest/state readers, read-only compatibility decoders, `tests/fixtures/v1/` changes, and checkpoint migration.
- Expanding `StreamSession` admission to these multi-input operators. That requires a separate session contract.

## 5. Dependencies and shared invariants

### 5.1 Required predecessors

- **P2:** synchronous operator dispatch, owned batch transactions, typed state access and the undo-log pattern are available. P4 does not alter rolling numerical results or routing in this step.
- **P3:** runtime-owned copy-on-write state pages, dirty-page bitmaps, frozen epoch references, background persistence and manifest v4 are available. P4/P5 declare state through that interface and do not implement operator-local `checkpoint()` encoders.
- **P0:** exact reference oracles, fixed fixtures and baseline builds are sealed before measurement. Historical numbers in §1 do not substitute for P0 builds.

P4 precedes P5 because P5 consumes P4's keyed core and typed run/state conventions. P6 starts only after P5 state has explicit key ownership and no serial routing/copy layer that would negate partitioning.

### 5.2 Execution and cancellation

All migrated data, progress and finite kernels are synchronous CPU-local operations. There is no `.await`, `yield_now`, `consume_budget`, Tokio task hop or per-edge charge inside keying, state update, probing, eviction or selection. DataFusion remains the SQL/expression planner and evaluates approved scalar expressions; it never executes Join's equality probe.

A compute morsel is at most **4,096 input rows or selected output pairs**. Variable key/payload work also checks at least every **64 KiB** traversed. Cancellation is checked before starting and at every morsel boundary. A cancelled or failed pre-emission attempt restores the exact prior state and releases transient ownership. As today, output already accepted by a downstream collector cannot be withdrawn; a later emit failure terminates the live attempt/job and recovery starts at the last durable cut.

Input envelopes are admitted once against ingress budgets and emitted envelopes once against egress budgets. Internal synchronous calls do not perform async accounting. Checkpoint-off and checkpoint-on jobs execute the same compute path; checkpoint-off uses the P3 no-op persistence dirty-bit interface, not a second operator implementation.

## 6. Shared keyed-state core (P4)

### 6.1 Key schema, identity and equality

**FR-K1 — One compiled key schema.** Each consumer declares an ordered, exact Arrow key schema and an input null policy at plan time. The common supported key types are Boolean, signed/unsigned 8/16/32/64-bit integers, Utf8/LargeUtf8, Date32/Date64 and all timestamp units with exact timezone metadata. No implicit cast, dictionary decoding rule or timezone normalization is introduced.

- SQL grouping treats null key components as groupable values according to existing DataFusion semantics.
- Join drops any row with a null key component before key birth and preserves its existing counters/precedence.
- ASOF rejects null keys under its existing exact-schema rules.

**FR-K2 — Vectorized routing.** For each morsel, the core hashes the typed key columns directly into one row-to-group-ID vector. Common fixed-width, Boolean, single-string and composite keys have plan-selected typed paths. Hash-table lookup compares the complete typed key against the columnar key store on a collision; a hash alone never establishes identity. Hash iteration order, random seed and table capacity are unobservable.

**FR-K3 — Birth and canonical bytes.** A first-seen key receives the next monotonically increasing dense group ID in accepted input order. At that birth only, the core stores each key component in typed append-only columns and constructs one canonical, type-tagged, length-delimited byte identity. Existing rows hash and compare without canonical framing. Canonical bytes are immutable and reused for state output, checkpoint encoding and equality audits; no per-row or per-checkpoint re-framing is permitted.

SQL group IDs remain append-only. Join/ASOF leases may retire a key when no live row, pending identity, undo entry or frozen page references it. A retired ID is never reused in the same lineage. State-table compaction may reclaim retired key payload/hash storage but preserves every retained ID and rewrites no live group identity. Restore preserves the persisted IDs and next-ID frontier; it never reassigns IDs in hash or canonical-key order.

**FR-K4 — State-table ownership.** Key columns, hash-table storage, group-ID vectors and lease metadata are runtime-owned state pages with arithmetic capacity charges. There is no per-key `MemoryReservation`. A new-key transaction preflights key-column, hash and page growth before mutation. Undo includes key births and lease changes so a rejected batch cannot consume a visible group or leak a key.

### 6.2 Determinism and recovery

**FR-K5 — Deterministic assignment.** Given the same ordered input trace, group IDs and canonical bytes are identical across process restarts, hash seeds and input RecordBatch boundaries. Multi-input Join/ASOF IDs depend on the concrete legal arrival schedule; no cross-source order is invented. Checkpoint state persists live keys, canonical identities, leases and the next-ID frontier. Restore validates key schema, canonical uniqueness, ID/reference bijection and all charges before installation.

**FR-K6 — Fingerprint.** The shared state-layout fingerprint covers a versioned layout ID, exact key schema and null policy, canonical encoding version, typed hash/equality contract version, consumer kind, consumer semantic specification and engine/state-layout identities. It excludes hash-table capacity, morsel position, runtime tuning and checkpoint interval. Any mismatch fails before source replay output or state installation.

## 7. P4 typed grouped aggregates

### 7.1 State and update transaction

**FR-A1 — Typed SoA state.** Replace `Group`, per-group scalar vectors, aggregate enums and per-group `Box<dyn Accumulator>` ownership with contiguous group-indexed columns. Each declared aggregate owns typed state fields and validity/initialized columns; key values live only in the shared key store. The single global aggregate uses reserved group slot zero. Plan-time specialization selects the aggregate/dtype/null path, not per-row enum dispatch.

Batch-local `GroupsAccumulator` or typed partial columns may process a morsel, but no per-group heap object becomes persistent state. Integer SUM retains current wrapping/null behavior. Existing sequential grouped-float and ordered-LAST_VALUE policies retain their current numerical and tie-breaking semantics; P4 does not import P2 rolling arithmetic into SQL.

**FR-A2 — Undo, not candidates.** An input batch runs as one transaction:

1. Validate the complete input, compile/prove the plan, evaluate predicates/arguments, route keys and preflight all group, page, undo, output and budget growth.
2. Record old typed fields once for each touched existing group and record births for new groups. Apply updates in original row order within each group, checking cancellation only at morsel boundaries.
3. Finalize only touched groups and construct the selected snapshot or changelog output.
4. After successful emission, atomically install state, key births, dirty-page marks, counters, output caches and next output sequence. Before emission, failure/cancellation undoes every touched slot and birth.

The undo log is required with checkpoints both on and off because it provides batch atomicity. Its live capacity and any pages pinned by P3 are charged arithmetically. A checkpoint freezes state-page references; it never clones all touched groups and never encodes operator-specific group objects.

**FR-A3 — Admission.** Every query shape admitted compact before P4 remains compact with equivalent results and budgets. In addition, admit non-distinct built-in `MIN`/`MAX` over Boolean, Date32, Date64, Time32/Time64, Timestamp and Duration arguments, including grouped/global forms, direct or already-allowlisted deterministic scalar expressions, `WHERE`, aggregate `FILTER`, `HAVING` and final projection. Null and ordering behavior must match DataFusion 55. No new UDF, distinct, ordered aggregate or window-function shape is admitted.

An unsupported shape without an explicit compact requirement retains the existing retained-input DataFusion fallback until P7. An explicit `sql_compact_state_budget` or changelog declaration rejects unsupported compact execution; it never silently switches to retained execution after state exists.

### 7.2 Default full snapshots

**FR-A4 — Dirty output chunks.** Group IDs partition output into 8,192-group chunks. Cache complete projected key/aggregate output arrays or RecordBatches per chunk, not only keys. A batch marks chunks containing touched/new groups dirty; only those chunks are rebuilt. Clean chunks are sliced/reused in group-ID order. An empty input or an update with no visible state change reuses the prior complete snapshot while retaining the existing rule that every accepted input batch produces a full cumulative snapshot.

The complete default snapshot remains one logical output envelope and must fit the effective edge budget whole; it is not split into replacement fragments. `HAVING`, projection and any already-supported outer `ORDER BY`/`LIMIT` retain their current semantics. If outer order/limit requires a global rebuild, charge and perform it once per snapshot; clean chunk reuse upstream of that step remains required.

### 7.3 Opt-in changelog output

**FR-C1 — Semantic declaration.** Add Rust `SqlAggregateOutputMode::{Snapshot, Changelog}` with `Snapshot` as default and a `SqlOperator` setter for the mode. Add symbolic Python `cf.changelog_sql(query, /, **tables)` and `TableExpr.changelog_sql(query, /)` as the explicit high-level opt-ins; existing `cf.sql`, `TableExpr.sql` and `snapshot_sql` are unchanged. Project-v3 SQL nodes carry optional `"aggregate_output": "snapshot" | "changelog"`, defaulting to `"snapshot"` when absent.

The mode is part of the node configuration, output schema and semantic/state fingerprints; it is not a runtime tuning field. Existing project documents without the field retain their meaning and fingerprint. Batch mode, `latest_relation`, multi-alias stream SQL, UDF SQL and non-aggregate SQL reject `Changelog` before execution.

**FR-C2 — Admission and schema.** Changelog requires compact aggregate execution, no outer `ORDER BY` or `LIMIT`, and a visible group identity: every `GROUP BY` expression must appear exactly once in the final projection as that same typed expression or an alias, without a cast. A global aggregate has the singleton empty key. A user output field named `_cf_op` is rejected for this mode.

The output schema is the ordinary projected SQL schema followed by non-null Utf8 `_cf_op`. Its values are exactly `"upsert"` and `"retract"`. Both operations carry the complete projected row; a retract carries the previously visible values, not only keys. Group ID and canonical bytes are implementation details and are not exposed.

**FR-C3 — Baseline and diff.** The logical baseline is the last successfully emitted cumulative visible relation, initially empty. For each accepted input batch, compare the pre-update and post-update visible row for every affected group, including `HAVING` transitions. Equality includes nulls and exact typed values, including floating-point bit patterns. A changed group produces, in this order as applicable:

1. `"retract"` with the old row if it was visible;
2. `"upsert"` with the new row if it is visible.

A newly visible group emits only upsert; a group leaving `HAVING` emits only retract; a touched group with identical visible output emits nothing. There is no output for a completely unchanged relation. Applying the operations in order by group key must reproduce the default snapshot relation after every batch. For a global aggregate the singleton key is used.

**FR-C4 — Ordering and flush.** Changes are ordered by ascending internal group ID, with retract before upsert within a group. One accepted data Batch flushes one ordered change sequence at the same boundary where default mode emits its snapshot. Output may be split into edge-budgeted envelopes only between operations; chunking never reorders a group's retract/upsert pair. Watermark, idle and EOF emit no synthetic baseline or extra diff.

**FR-C5 — Recovery and delivery.** The restored committed aggregate state is the restored baseline; no separate shadow row-object map is allowed. A checkpoint captures the state only after its corresponding output was accepted. Restore emits no bootstrap upserts and computes the next diff from restored state. If a downstream failure follows partial acceptance, the live job fails; durable replay may repeat operations according to the existing at-least-once sink contract. Changelog does not itself provide exactly-once delivery or consumer deduplication.

### 7.4 Aggregate budgets and state identity

`sql_compact_state_budget.max_rows` continues to count live groups. `max_bytes` covers key/hash pages, typed state capacity, undo ownership, cached output chunks, P3-pinned state and retained immutable checkpoint handles using checked column-capacity arithmetic. Expression/input buffers and transient output gathering retain their documented separate reservations. Admission preflights old state plus new/undo/cache overlap; a one-byte-short budget rejects the whole batch with no output or state change.

SQL's new state fingerprint includes the query/UDF references, exact schemas, aggregate output mode, compiled numerical policy, shared-key layout and P3 layout/engine identities. Layouts 3/4/5 and their encoders are deleted for migrated compact state; unsupported retained SQL remains a separate current path until P7. Old SQL compact checkpoints fail closed rather than being decoded into the new state.

## 8. P5 Join and interval Join

### 8.1 Columnar runs and admission

**FR-J1 — Run state.** Each side stores accepted retained rows as immutable columnar runs. A run references one projected Arrow batch slice and carries parallel group-ID, event-time and physical row-ID columns, a run ID, live range/prefix metadata and ownership/accounting handles. Probe-only rows remain in the temporary admitted batch and never allocate persistent row objects. The incoming retain bitmap skips all persistent payload, expiration and lease work for false bits.

Each physical input row consumes its per-side row ID, including dropped rows, preserving existing identity/order behavior. Null time/key, lateness and complete-envelope validation occur before key/state commit. Per-key indexes contain run/row locators ordered by `(event_time, row_id)`; an out-of-order admission run stores a sorted locator vector without rewriting its Arrow payload. No `StoredRow`, `LocalIntern` → persistent `Dictionary` → `NativeIndex` identity chain remains.

**FR-J2 — Native selection-vector probe.** Route incoming key columns once through the shared core. For every admitted incoming row, probe only the opposite bucket with that group ID and evaluate the inclusive interval `[left_time - before, left_time + after]` in checked `i128` arithmetic. Count the complete match set and enforce `max_matches_per_input_batch` before output allocation or emission.

The probe emits two parallel selection vectors in exact `(incoming admitted position, opposite event_time, opposite row_id)` order. Time bounds may reuse a cursor for monotonic same-key probes and binary/galloping bounds otherwise; both paths must produce identical selections. The opposite bucket's ordered run locators make result multiplicity, inclusive boundaries and on-time out-of-order rows deterministic. No DataFusion SQL equality query, retained-state key materialization for SQL, or SQL retry/fallback tier is permitted.

### 8.2 Materialization and budgets

**FR-J3 — One selection-driven materialization.** Translate the selection vectors into incoming/opposite source locations and gather each selected output column once. Use Arrow `take` for a contiguous/single-owner source and `interleave` for multiple source runs; concatenate ranges in selection order. Never construct per-pair row records. Existing output field order/prefixes, nullability, direct-consumer projection and dictionary/nested payload behavior remain unchanged.

Preflight the match cap, all output chunks, sequence range and effective edge budget before the first emission. Fixed-width schemas use row counts and plan-computed width/bitmap arithmetic per output chunk rather than per-pair byte accumulation. Variable-width/nested/dictionary schemas compute exact selected visible charges from source ranges/offsets; unreferenced dictionary values do not count. A single oversized row, overflow or one-byte-short budget fails before emission and before state commit. Each emitted chunk owns only its materialized payload and does not pin the complete fan-out buffer.

**FR-J4 — Accounting.** `max_state_rows_per_side` counts live retained rows. `max_state_bytes_per_side` is the new versioned logical charge for that side's live run metadata, retained visible Arrow payloads, group/time/row-ID capacities and expiration indexes. Shared key-store pages are charged once at operator level; per-side charges do not depend on which side happened to create a shared key. Fixed-width run charges are computed by batch/column arithmetic, not per-row reservations. All shared owners are charged once per actual owner, including overlap during prefix replacement and frozen P3 pages.

The new accounting/layout identity is part of the state fingerprint. Existing limit names, failure reasons and prospective all-or-nothing admission remain; documentation must state that old retained-byte values and old checkpoints do not carry over to this layout.

### 8.3 Eviction, progress and recovery

**FR-J5 — Prefix eviction.** On opposite progress or end, preserve the exact expiry predicate `row_time + extension < opposite_watermark`, with end expiring all. Each run's time-ordered locators permit a single partition point and prefix cut. Update run heads, per-key locators, row/byte gauges and key leases in aggregate per run/key; release fully retired runs without visiting each payload row. Never evict an equality boundary row early.

Join emits only append-only matched pairs; eviction emits no output or retraction. Output frontier, idle/ended behavior, late counters, terminal cleanup and next output sequence retain current semantics. Finite execution uses the same run/probe/materialization kernels with fresh run-local state and no stream checkpoint.

**FR-J6 — State and recovery.** Runs, indexes, expiration prefixes, key leases, row-ID frontiers, metrics, progress and output sequence are declared P3 state. Undo covers admissions and prefix cuts. A checkpoint freezes dirty run/index/key pages and persists canonical keys and live run references; capture cost is proportional to dirty pages, not all rows. Restore validates complete run/reference coverage, row-ID uniqueness/frontiers, per-key ordering, interval configuration, schemas, limits and segment integrity before installation.

Join v1/v2 snapshots, pending/tombstone logs, operator-local compaction and their magic/layout encoders are deleted in the Join migration commit. An old lineage receives the P3 explicit incompatible-lineage error.

## 9. P5 ASOF migration

**FR-S1 — Shared keys and columnar pending/right state.** Replace ASOF's private key encodings/dictionaries/eight shards with the shared key core. Pending left rows and right history use columnar runs referencing Arrow payloads. Right per-key indexes order candidates by `(event_time, typed sequence tuple)`. Pending left output order remains `(left_time, left key tuple, left sequence tuple)` using typed key ordering, never group-ID order. Complete identity remains `(side, key tuple, event_time, sequence tuple)`.

**FR-S2 — Selection and finality.** For each ready left row, select the greatest `(right_time, right_sequence)` within inclusive `[left_time - tolerance, left_time]`, or no candidate. Strict finality still requires both unfinished watermarks to be strictly greater than the left time; idle preserves its watermark, equality does not close, and EOF drains before EOF forwarding. Late policy, duplicate precedence, identity-only history, unmatched left preservation and all exact input type rules are unchanged.

**FR-S3 — Output and eviction.** Finalization produces left/right selection vectors in canonical left order, with explicit right validity for unmatched rows. Materialize each selected column once using `take`/`interleave`, including existing direct-consumer projection. Right payload and identity-only expiration use run-prefix cuts under the current conservative rule: `right_time + tolerance < min(left_future_time_bound, earliest_pending_left_time)`; equality remains eligible. Right EOF alone does not discard useful history.

**FR-S4 — Limits and recovery.** The shared `max_state_rows` still counts pending left, retained right and required identity-only entries once each. `max_state_bytes` covers new run/key/index/page capacities and owners; the independent workspace ceiling still bounds admission, sorting, probing and materialization. Resource refusal never selects a worse candidate or discards useful history. Accepted output prefixes, remaining pending rows and sequence/progress commit exactly as the current contract requires.

All ASOF state uses P3 pages and shared-key identities. Restore validates candidate ordering, identity uniqueness, pending/output frontiers, charges and payload references, then continues bitwise. ASOF v1/v2, existing native row-log layouts and source-replay-specific operator encodings are removed for this rebuilt operator in the same commit; old snapshots are rejected under the approved no-backward-compatibility decision. Finite ASOF uses the same selection/materialization kernels over complete fresh inputs and must retain the documented finite-vs-stream final-relation oracle.

## 10. Required tests and independent oracles

Every behavior change starts with a focused failing test recording the expected failure. Tests exercise public behavior and private work counters where necessary to prove that deleted work is actually absent.

### 10.1 Shared core and P4

1. **Key differential:** for every supported key type, compare group assignments against a naive typed/canonical map over randomized rows, forced hash collisions, composite/null keys, empty/Unicode/NUL/long strings, signed extremes and timestamp unit/timezone variants. Repeat with arbitrary RecordBatch splits and after new-format restore. Assert identical canonical bytes and deterministic group assignment.
2. **Birth-work counter:** N rows with D distinct keys construct exactly D canonical identities; re-probes and unchanged checkpoint captures construct zero. Cancellation at every morsel and long-key byte boundary leaves no key, lease, charge or ID mutation.
3. **Aggregate differential:** compare default compact snapshots with the pre-P4 compact oracle for every previously admitted shape and with DataFusion 55 for the newly admitted temporal MIN/MAX shapes. Cover nulls, filters, HAVING, projection, ordered output, global/grouped forms, integer wrapping, sequential floats and ordered-LAST_VALUE ties. Do not weaken existing float or tie rules to make a generic finite comparison pass.
4. **Snapshot chunk oracle:** retain all snapshots; check complete schema, group uniqueness, values and order. Work counters prove that only chunks containing touched groups rebuild, clean key/aggregate arrays are reused, and no full-state candidate clone occurs. Include sparse updates, new groups crossing an 8,192-group boundary and empty input.
5. **Undo/fault matrix:** inject cancellation, allocation refusal, aggregate finalization failure, output-budget refusal and checkpoint freeze/persist failure at every declared phase. Before output acceptance, state bytes/values, key IDs, caches and counters equal the pre-call state; after partial downstream acceptance, verify documented terminal recovery rather than attempting an impossible sink rollback.
6. **Changelog fold oracle:** independently fold `(group key, retract/upsert, full row)` events into a map and compare after every input with the default snapshot relation. Cover first baseline, update, unchanged visible value, null key, global singleton, HAVING enter/exit/stay-hidden, retract/upsert split across output envelopes, empty input, EOF and restore. Assert schema, `_cf_op` domain, ascending group order and retract-before-upsert.
7. **API/project/recovery:** test Rust defaults/setter, Python declarations and validation, generated project schema, omission/default equivalence, fingerprint differences, reserved `_cf_op`, missing visible keys, rejected ORDER/LIMIT and unsupported compact fallback rules. New-format checkpoint/resume must emit no bootstrap diff and must continue both output modes bitwise.
8. **Deletion/resource tests:** old compact layouts fail closed; no per-group reservation/object remains; one-byte-short group/state/undo/output-cache budgets reject atomically; checkpoint-off and -on share the same compute counters.

### 10.2 P5 Join and ASOF

1. **Join nested-loop oracle:** for each concrete arrival schedule, independently enumerate equal non-null keys and inclusive time bounds, then compare exact pair values, multiplicity and order `(incoming admitted position, opposite time, opposite row ID)`. Cover both incoming sides, duplicate pairs at both boundaries, dropped rows consuming IDs, late/null precedence, on-time disorder, multiple runs per key and direct projection.
2. **Join payload/materialization oracle:** compare selection-driven output with explicit row-wise expected arrays for flat, nullable, variable-width, dictionary, nested, sliced and multi-owner inputs. Check output chunk sequences, single-row/whole-batch byte limits, unreferenced dictionaries and no full fan-out backing retention.
3. **Eviction oracle:** maintain a naive live-row set and compare after every watermark/end. Cover strict equality at expiry, run-prefix cuts, out-of-order runs, complete-side flush, restored leases and no-op progress. Work counters must scale with affected runs/keys and expired prefixes, not all retained rows.
4. **No-SQL-probe proof:** exercise every supported Join key type and composite/null/collision cases through native probes. A test DataFusion hook must observe zero equality-probe queries; source/module inventory confirms removal of `equality_query`, SQL retry tables and the three-layer key identity.
5. **ASOF naive oracle:** independently choose maximum `(right_time, right_sequence)` in inclusive tolerance for every left row and compare canonical output, unmatched nulls and right nullability. Cover ties, composite typed sequences, empty sides, out-of-order input, late error/drop, duplicates, identity-only expiration and tolerance boundary equality.
6. **Progress/finite oracle:** verify strict dual-watermark closure, idle/reactivation, forwarded frontier, EOF drain and status invariants. Compare uninterrupted finite and stream final relations across batch splits and legal arrival schedules without requiring intermediate chunk equality.
7. **Recovery/fault oracle:** for Join and ASOF, replay generated traces with checkpoint cuts after admissions, partial outputs, progress and evictions. Resume from exact source cursors and compare all post-cut outputs/state fingerprints with an uninterrupted new-format run. Test corrupt references, duplicate IDs, wrong ordering, changed specs/schemas, incompatible old layouts, cancellation and one-byte-short state/workspace/output limits before installation.
8. **Synchronous-work proof:** migrated kernels contain no inner async quantum/yield and perform cancellation at the declared morsel boundaries. Deletion tests cover Join v1/v2, ASOF old encodings/shards and operator-local checkpoint journals; `unsafe_code = "forbid"` remains unchanged.

## 11. Pre-registered performance protocols

No performance measurement is part of writing this specification. Implementation uses sealed default-feature release builds, identical dependencies/harnesses, matching workload fingerprints and complete untimed correctness oracles. Each case has **two rounds of ten adjacent alternating AB/BA pairs**, reversing the leading arm in round two, plus one excluded warmup per arm: 20 formal observations per arm per case. Builds are estimated/reported separately.

Each step has at most five cases and one **600-second total wall-clock budget**, including fixture creation, startup, preflight, warmup, timing, complete correctness/lifecycle checks and cleanup. Stop at 570 seconds for requests, reserving 30 seconds for owned cleanup. No extra case, sample, favorable retry or budget extension is allowed. Partial/inconclusive results remain explicit. Absolute gates use candidate p50 across the 20 formal observations; paired comparisons also retain `scripts.benchmark_suite.report.comparison` and its maintained 5% verdict. A missed absolute gate blocks the phase even if the paired verdict says no regression.

### 11.1 P4 — five cases

Baseline is the sealed P3 head in PR #619; candidate is the P4 head. All stream cases use 100 × 100 groups, one DataFusion partition, checkpoints off and default full snapshots unless stated otherwise. Reuse `grouped_sum_state_benchmark` and `scripts.benchmark_suite.sum_state`; the 640-row control is a declared workload migration, not a silent geometry change.

| Case | Geometry and affinity | Purpose / acceptance |
|------|-----------------------|----------------------|
| P4-1 dense 1 CPU | 10k seeds + 10M business rows, 1M-row source batches, CPU0/Tokio1 | Principal gate: candidate p50 ≤0.5 s; every cumulative snapshot checked |
| P4-2 dense 8 CPU | Same geometry, CPUs0–7/Tokio8, one partition | Pre-P6 regression control; no scaling credit |
| P4-3 sparse 1 CPU | 10k seeds + 20 × 100 updates, CPU0/Tokio1 | Dirty-chunk/sparse-update control with all 21 snapshots checked |
| P4-4 finite control | 10k seeds + 10M rows, 1M-row batches, CPUs0–7, eight partitions | Finite SQL control; one final result, no stream snapshot comparison |
| P4-5 small append 1 CPU | 10k seeds + 20 × 640 business rows, CPU0/Tokio1 | Small-batch keyed/dirty-output control; no absolute gate added |

P4-1 must satisfy the absolute gate. Every case must complete its exact oracle; P4-2 through P4-5 report maintained comparison verdicts without claiming equivalence from an inconclusive result. Changelog correctness is mandatory in focused tests but adds no sixth performance case.

### 11.2 P5 — five cases

Baseline is the sealed P4 head; candidate is the P5 head. Reuse the maintained `EngineCase`/Join harness and `ready-enqueue-to-arrow` boundary. Large inputs contain 1M rows; Join has 1M quote rows and 64 static dimension rows, while ASOF/interval replay 1M rows per side. Interval has 64 keys, inclusive ±5 s bounds and exactly 10,998,080 expected pairs. Checkpoints are off for this kernel acquisition.

| Case | Geometry and affinity | Purpose / acceptance |
|------|-----------------------|----------------------|
| P5-1 Join 1 CPU | 1M × 64, 1M source batch, CPU0/Tokio1 | Principal gate: candidate p50 ≤30 ms; exact sequence/value oracle |
| P5-2 ASOF 1 CPU | 1M per side, 1M source batches, CPU0/Tokio1 | Shared-key ASOF migration control; exact canonical result |
| P5-3 interval 1 CPU | 1M per side, 1M source batches, CPU0/Tokio1 | Principal gate: candidate p50 ≤350 ms; exact 10,998,080-pair multiset and per-probe order |
| P5-4 interval 32 CPU | Same geometry, CPUs0–31/Tokio32 | Pre-P6 regression control, not a scaling verdict |
| P5-5 Join segmentation control | 1M × 64, 64k source batch, CPU0/Tokio1 | Run/selection behavior across transport segmentation |

P5-1 and P5-3 are absolute blocking gates. Cross-input global interval order is not an oracle; exact pair multiset and guaranteed per-probe/run order are. ASOF and regression controls do not acquire new absolute targets.

P3-inherited checkpoint gates still apply to migrated group-by and Join at final integrated acceptance: enabled-but-untriggered overhead ≤3%, 1-second checkpoint throughput ≥90% of off, and 100 ms throughput ≥75% of off. They use the maintained checkpoint performance task/suite rather than expanding these two five-case kernel acquisitions. Pending scheduled evidence is not recorded as a passing gate.

## 12. Commit sequence, deletions and handoffs

All commits accumulate on the integrated PR #619 branch. The following order is mandatory; test red/green evidence accompanies each implementation commit.

1. **P4-1 — shared keyed core and inline grouped SUM.** Introduce the crate-private key schema/store, vectorized routing, collision equality, canonical birth bytes, leases, morsel cancellation and P3 page declarations together with the grouped Int64/UInt64 SUM consumer. Replace candidate cloning for that consumer with undo. Do not add unused window/distinct scaffolding.
2. **P4-2 — all compact aggregates and dirty snapshots.** Move every previously compact shape onto typed state columns, add the specified temporal MIN/MAX admission, cache complete output chunks and migrate compact state to P3 pages. Delete the old compact `Group`/candidate/container layout and old SQL compact encodings at this phase boundary. Unsupported retained SQL remains explicitly separate.
3. **P4-3 — opt-in changelog.** Add Rust/Python/project declarations, schema inference, diff/flush/recovery semantics and fold oracles. Run the pre-registered P4 acquisition after the complete P4 behavior exists; archive its exact evidence before P5.
4. **P5-1 — Join and interval Join.** Migrate admission, run state, native probes, selection vectors, take/interleave output, prefix eviction and P3 recovery together. Delete Join v1/v2, SQL probe tables/retries, `StoredRow`, the three-layer key identity, pending/tombstone checkpoint logs and inner async quanta in the same commit.
5. **P5-2 — ASOF.** Move shared keys, left/right runs, candidate selection, output and prefix retirement onto the same core. Delete old ASOF key shards/dictionaries, native row-log/source-replay operator encodings and local journal/capture machinery in the same commit.
6. **P5-3 — integrated evidence and docs.** Run the P5 acquisition, record all gates/limitations, reconcile public docs/CHANGELOG through `cf-doc-writer`, and obtain final specialist review. No commit or evidence is pushed by the specification author.

`cf-api-designer` must review the public changelog declaration/schema/fingerprint contract and `cf-critic` must review transaction, changelog baseline/recovery, eviction and measurement semantics before implementation. The orchestrator schedules those steps; this specification does not authorize implementation or a plan-gate change by itself.

## 13. Acceptance criteria

- [ ] Shared key differential and birth-work tests prove typed collision-safe identity, D canonical births for D distinct keys, deterministic restore and bounded synchronous cancellation.
- [ ] All previously compact SQL shapes pass their existing semantic oracles on typed state; newly admitted temporal MIN/MAX shapes match DataFusion 55, and explicit compact requirements never fall back silently.
- [ ] Default aggregate output remains a complete snapshot after every accepted input; only dirty 8,192-group chunks rebuild and all snapshot oracles pass.
- [ ] Changelog is opt-in, default project/API behavior is unchanged, and an independent fold after every batch equals the default cumulative relation, including HAVING and recovery.
- [ ] Grouped SUM P4-1 completes the exact 10M/1M-batch single-CPU protocol with candidate p50 ≤0.5 s and complete snapshots; all other P4 cases and limitations are reported without expansion.
- [ ] Join emits exactly the nested-loop oracle's pairs and order without any SQL probe; selection-driven take/interleave output, prefix eviction, budgets and new-format recovery pass all generated/fault cases.
- [ ] ASOF preserves candidate choice, canonical output, strict finality, identity/duplicate/late behavior, bounded eviction and finite/stream equivalence on shared keyed run state.
- [ ] P5-1 Join candidate p50 ≤30 ms and P5-3 interval candidate p50 ≤350 ms under the exact single-CPU protocol; all five P5 cases complete their correctness oracles or the phase remains incomplete.
- [ ] P3-inherited checkpoint overhead/throughput gates for migrated group-by and Join have passing final evidence; old SQL/Join/ASOF layouts fail closed and no retired production path remains at its phase boundary.
- [ ] Focused Rust/Python suites, generated project-contract checks, `git diff --check`, necessary formatting/lint/rustdoc checks and final specialist review pass. Full regression, 90% Rust line coverage and required cross-platform checks remain CI responsibilities; pending CI is not merge readiness.
- [ ] Published docs, `_native.pyi`/generated schemas where applicable, capabilities and CHANGELOG describe the new output mode, accounting identities and no-old-checkpoint rule without claiming unmeasured performance.

## 14. Open questions

None. The public API shape and the high-risk recovery/changelog semantics still require the named API and critic reviews, but this specification supplies one concrete contract and does not offer alternative architectures or alter approved gates.
