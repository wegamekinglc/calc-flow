# Continuous Streaming 3.0 — M4 State and Window Critique

| Field             | Value                                                                  |
| ----------------- | ---------------------------------------------------------------------- |
| Status            | **Approved for documentation merge; implementation remains pending**   |
| Review type       | Current-main reconciliation and adversarial design review              |
| Baseline          | `main@6275599ca3bb872ab480b677f13bcd9698b144f0`                        |
| Spec input        | `specs/m4-state-window.md`, D1–D12 and AC-01–AC-58                     |
| API input         | `api-notes/m4-state-window.md`                                         |
| Artifact slug     | `m4-state-window`                                                      |
| Reviewer verdict  | **Zero document blockers; no implementation evidence claimed**         |

## 1. Review target and method

The review compared the proposed M4 specification and API note against:

1. current `origin/main` at the baseline commit;
2. the Arroyo/RisingWave research document;
3. the detailed continuous-streaming plan;
4. the approved M0 runtime specification and API note;
5. the M3 delta package and PR #86 implementation;
6. current checkpoint, stream-operator, compiler, runtime-task, metrics,
   progress, and soak source.

The review attacked five boundaries:

- whether M4 can merge without silently implementing half of M5;
- whether state publication and cleanup remain recoverable at every crash
  point;
- whether incremental window state can stay outside bounded JSON;
- whether window/late semantics are deterministic across Arrow batch splits,
  replay, and compaction;
- whether the proposed types can actually fit the current Rust module and task
  ownership structure.

Reviewed input hashes:

- specification SHA-256:
  `a417f98917c7a19580a6fbe38d4872f9f245d103867346a4409d3ff07da5f553`;
- API note SHA-256:
  `f3bde952eedd88d5b7f544e5a89043651083db01849a6768049da6dc58577ad1`.

“Approved” means the documents are coherent, implementable, and testable. It
does not mean code, tests, benchmarks, soak, CI, Codacy, or review evidence
exists.

## 2. Executive verdict

The first-pass plan exposed fifteen material risks. All fifteen now have explicit
resolutions in the specification/API note. No blocker, major, or minor
document issue remains.

| Severity                    | Open count |
| --------------------------- | ---------- |
| Blocker                     | 0          |
| Major                       | 0          |
| Minor                       | 0          |
| Implementation advisories   | 7          |

The strongest decisions are:

- M4 finalizes the v3 manifest model but does not replace or drive the
  checkpoint store;
- segment operations require a backend-neutral lineage-scoped session;
- built-in window state is immutable Arrow IPC delta/base state with explicit
  tombstones and bounded-inventory compaction;
- late classification occurs only after concrete window assignment;
- aggregate and float semantics are fully enumerated rather than delegated to
  DataFusion defaults;
- persistent metrics are task-owned rather than context-owned;
- M4 has an operator/state round-trip, not a false durable-job-recovery claim.

## 3. Findings and resolutions

### F1 — The historical file list is impossible beside `checkpoint.rs`

**Risk.** The detailed plan asks M4.1 to create
`crates/calc-flow/src/checkpoint/model.rs`, but current main still compiles the
flat module `src/checkpoint.rs`. Rust cannot resolve both layouts under one
module name without converting the module now, which would pull M5.1's public
v2-to-v3 replacement into M4.

**Resolution.** Specification D1 and API §2 place the final v3 model in
`state/manifest.rs`. M5 later moves/re-exports that exact model while replacing
the store. M4 never edits the checkpoint module layout.

**Verdict.** Resolved.

### F2 — M4 acceptance language accidentally claimed M5 crash recovery

**Risk.** Historical M4.2 says a crash selects the latest committed manifest,
while M4.4 asks for “checkpoint recovery.” Current runtime rejects Barrier and
has no coordinator, store-v3 selection, source reopening, or sink recovery.
Implementing those words literally either expands M4 into M5 or produces a
misleading recovery claim.

**Resolution.** Specification §4 and D1/D12 distinguish a private commit-fault
harness, operator snapshot/restore, and state-inventory restore from a complete
running-job recovery. M5 alone integrates barrier, manifest selection, source
cursor, sink pre-commit, and job restart.

**Verdict.** Resolved.

### F3 — The original `StateBackend` sketch cannot enforce lineage locking

**Risk.** M0's flat trait exposes segment methods on `&self`, while A6 requires
an exclusive lease across the state backend and checkpoint store. A future
runner receives `Arc<dyn StateBackend>` and cannot safely downcast to
`LocalStateBackend::acquire_lineage`. Cleanup/publication could therefore run
without the required cross-process lease.

**Resolution.** API §4 changes `StateBackend` into a backend-neutral factory
for a non-cloneable `StateLineageBackend` session. The returned session owns
the lease and is the only object exposing stage/load/publish/collect methods.
This is an explicit M4 amendment to the earlier illustrative flat trait.

**Verdict.** Resolved; the implementation must add compile tests using a mock
backend to prove the runner-facing seam is not Local-only.

### F4 — “Manifest last” was underspecified without segment publication

**Risk.** Writing and validating a staged segment is insufficient. A manifest
that references a committed path cannot publish before the staged-to-committed
rename and parent-directory synchronization. This exact gap blocked the first
M0 critique round.

**Resolution.** Specification D3 repeats all eight ordered operations,
including segment rename and directory synchronization before manifest write.
AC-09 injects at every boundary.

**Verdict.** Resolved.

### F5 — Incremental deltas can make a “bounded” manifest grow forever

**Risk.** “Flush only dirty keys” naturally accumulates one handle per epoch.
Retaining only two manifests does not bound the handle count of the latest
manifest. Without a compaction trigger tied to predicted manifest size, a
large number of small epochs eventually exceeds the 10 MiB document limit.

**Resolution.** D5 introduces base-plus-delta inventory, deterministic
last-operation-wins restore, and mandatory compaction before predicted
manifest overflow. Old segments remain reachable through retained historical
manifests. AC-14/AC-15 cover both sides.

**Verdict.** Resolved.

### F6 — The state backend cannot require Arrow IPC for custom operators

**Risk.** The approved public `OperatorStateSnapshot` lets custom operators own
arbitrary segment bytes. Enforcing Arrow IPC in the generic backend would
contradict that existing surface, while treating built-in window bytes as
opaque would fail the M4 state-format decision.

**Resolution.** D5 makes the backend an opaque length/checksum envelope and
requires Arrow IPC specifically for built-in window state. Restore validates
the built-in Arrow schema before installation. Custom bytes retain their
approved operator-owned format.

**Verdict.** Resolved.

### F7 — Synchronous `checkpoint()` can copy or encode all state on Tokio

**Risk.** `OperatorStateSnapshot` contains `Vec<u8>` values. A naive
implementation serializes every dirty accumulator during barrier handling or
clones prebuilt buffers, making control latency proportional to dirty bytes and
blocking an executor thread.

**Resolution.** D6 requires incremental preparation/owned handoff and moves
bulk encode, checksum, I/O, restore decode, and compaction off the executor.
AC-52 is a paused-time responsiveness test, not a wall-clock assertion.

**Verdict.** Resolved at contract level. The implementation must report copied
bytes in benchmark evidence rather than assuming `Vec` handoff is cheap.

### F8 — Final-only cleanup appears to require a missing commit callback

**Risk.** The runtime spec says closed state is tombstoned only after output is
durable, but the frozen `StreamOperator` trait has no checkpoint-complete
callback. Adding one casually would change a public trait and still not help
M4, which has no durable coordinator.

**Resolution.** D12 uses an emitted-but-not-checkpointed set. The next
operator snapshot carries the state transition/tombstones. If M5 later fails
to publish that snapshot, the job terminates and the previous manifest replays
the old state. No callback is required; a failed epoch never continues.

**Verdict.** Resolved. M5 fault tests must confirm the assumed terminal-on-
publication-failure behavior.

### F9 — Current late metrics disappear between handler calls and can wrap

**Risk.** `StreamOperatorContext::new` creates a new `LateRowRecorder` for each
data/watermark/EOF call. Its atomic `fetch_add` also wraps silently. M4 could
appear to record late rows in unit tests while every runtime status remains
zero or regresses after overflow.

**Resolution.** D11 and API §8 move metric ownership to `OperatorProgress`,
pass one shared sink to every context, stage one transactional delta per input
batch, and use checked counters. The existing public context constructor stays
source-compatible; a crate-private task constructor supplies the shared sink
and output budget. AC-43–AC-46 cover persistence, affected-batch cardinality,
overflow, and redaction.

**Verdict.** Resolved.

### F10 — A row-level late predicate breaks hopping and normal disorder

**Risk.** `event_time <= WM_in` drops valid rows whenever watermark delay is
smaller than window size and cannot represent a hopping row with both closed
and open assignments.

**Resolution.** D11 requires assignment expansion first and tests only
`assignment.window_end <= WM_in`. AC-40–AC-42 include the canonical 10:15/
10:30 case and mixed hopping assignments.

**Verdict.** Resolved.

### F11 — “Deterministic aggregate” is false without bit-level float rules

**Risk.** Default hash/equality, NaN propagation, signed zero, batch-local
partial sums, integer widening, Decimal rules, and DataFusion aggregate
choices can all change output across batch partition or compaction.

**Resolution.** D8/D9 fix group-key byte equality, 64 KiB bound, the complete
first-version type matrix, checked integer outputs, sequential FIFO float
accumulation, canonical NaN, total-order min/max, and explicit Decimal
rejection. AC-20/AC-21 and AC-30–AC-35 make every branch observable.

**Verdict.** Resolved. A future parallel accumulator would require a spec
amendment because reassociation is currently forbidden.

### F12 — A close event can exceed the bounded edge budget

**Risk.** `StreamCollector` applies backpressure but does not give a public
operator the edge budget. Emitting every newly closed key as one Batch can
produce an oversize envelope; staging every row before multiple emits can also
create an unbounded internal buffer.

**Resolution.** D10 and API §§7.3/8 add a crate-private effective-output-budget
context seam. The built-in window operator emits deterministic contiguous
chunks and fails before enqueue when one row alone exceeds the byte limit.

**Verdict.** Resolved. The implementation must use incremental Arrow builders
or bounded chunks; collecting the complete close event in a `Vec<RecordBatch>`
would violate the resolution.

### F13 — Watermark-triggered output has no inherited BatchMetadata

**Risk.** A data handler can preserve input metadata, but a window emits from
`on_watermark` or `on_end`, where no input `Batch` exists. Leaving metadata to
implementation convenience can duplicate sequence values, use unstable source
labels, or produce different chunks after restore.

**Resolution.** D10 and API §7.3 give each window operator a checked output
sequence starting at zero, use stable operator ID as source, keep attributes
empty, and capture/restore the next sequence. AC-36 covers row/chunk order and
metadata together.

**Verdict.** Resolved.

### F14 — The owned compiled enum cannot reach checkpoint/restore

**Risk.** `StreamOperator` exposes `checkpoint` and `restore`, but current
`CompiledStreamOperator` dispatches only reset, data, watermark, and end. A
window implementation could pass direct trait tests while M5 has no owned
runtime path to invoke its state lifecycle.

**Resolution.** WP3 and API §6.2 require checkpoint/restore dispatch for every
compiled variant. M4 tests the seam without accepting Barrier in
`operator_task`; M5 later calls it after alignment.

**Verdict.** Resolved.

### F15 — Hopping overlap can amplify one row without bound

**Risk.** The approved integral-overlap rule guarantees exactly
`size / slide` assignments but does not cap that quotient. A tiny slide and
large size can make one otherwise bounded input row allocate or process
billions of window updates, bypassing the edge row budget.

**Resolution.** D7 and API §6.1 set `MAX_WINDOW_OVERLAP = 1_024` and reject a
larger quotient at compile time. AC-17 covers divisibility and the overlap
bound.

**Verdict.** Resolved. Raising the limit later is a reviewed resource-limit
change, not an unchecked runtime knob.

## 4. Decision and acceptance closure

| Decision | Covered by                                            | Review result |
| -------- | ----------------------------------------------------- | ------------- |
| D1       | AC-02–AC-05, AC-55                                    | Conformant    |
| D2       | AC-01, AC-04, AC-12                                   | Conformant    |
| D3       | AC-06–AC-10                                           | Conformant    |
| D4       | AC-10–AC-12, AC-14                                    | Conformant    |
| D5       | AC-08, AC-13–AC-15, AC-47, AC-51                      | Conformant    |
| D6       | AC-52, AC-56                                          | Conformant    |
| D7       | AC-16–AC-18, AC-26–AC-29                              | Conformant    |
| D8       | AC-23, AC-32, AC-33, AC-36                            | Conformant    |
| D9       | AC-20, AC-21, AC-30–AC-35                             | Conformant    |
| D10      | AC-19, AC-23, AC-24, AC-36–AC-39                      | Conformant    |
| D11      | AC-40–AC-46                                           | Conformant    |
| D12      | AC-37–AC-39, AC-47–AC-51, AC-53–AC-58                 | Conformant    |

All 12 decisions have executable acceptance targets. All 58 acceptance IDs are
unique and assigned to a work package in the specification.

## 5. Implementation advisories

These advisories do not alter the contract:

1. Prefer pure functions for timestamp assignment, group encoding, scalar
   update, manifest validation, and state folding; confine mutation to the
   owned window state, lineage session, and metric recorder.
2. Reuse DataFusion's re-exported Arrow crate so one Arrow version owns arrays,
   schemas, and IPC; do not add a second direct Arrow version accidentally.
3. Use `BTreeMap`/`BTreeSet` for observed ordering and serialization. Hash maps
   may be used only for non-observed caches with an explicit deterministic
   projection before output.
4. Keep fault injection behind test-only seams. Production code should expose
   ordered primitives, not a public “crash here” API.
5. Measure cold and warm restore separately. The first decode/cache fill is not
   comparable to an already-cached restore.
6. Keep `tests/fixtures/v1/`, generated schemas, OpenAPI, and TypeScript clients
   unchanged; AC-55 should assert an empty diff for those paths.
7. Treat the advisory-lock crate as a supply-chain change: pin it through the
   workspace dependency table and include it in audit/deny plus release SBOM
   evidence.

## 6. Final verdict

The documentation package is suitable for a docs-only PR. It corrects stale
main-plan status, resolves the impossible checkpoint-module file list, defines
the M4/M5 durability boundary, and supplies an implementable state/window
contract with named evidence gates.

The future implementation must still provide:

- focused RED/GREEN evidence for every behavior change;
- AC-01–AC-58 mapping;
- full repository verification and 90% Rust line coverage;
- paired benchmarks with confidence intervals;
- the exact 20-minute soak on the final implementation SHA;
- resolved Copilot/Codacy/human review and exact-head CI evidence.

**BLOCKS REMAINING: 0**
