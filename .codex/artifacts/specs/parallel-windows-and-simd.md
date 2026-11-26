# P6/P7 Parallel Execution, Window Panes, and SIMD - Specification

## Source

- Delivery: the single integrated re-architecture PR #619; P6/P7 are commit
  sequences in that PR, not separate PRs.
- Approved plan:
  [2026-11-15-streaming-core-rearchitecture](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/docs/plans/2026-11-15-streaming-core-rearchitecture.md) sections 2, 3.4,
  4.4-4.5, 6, 7.7-7.9, 8, 9, and 10.
- Kernel and determinism dependency:
  `.codex/artifacts/specs/rolling-kernel-v2.md` sections 4.5, 5, and 6.
- Earlier pane constraints:
  [2026-11-08-ta-lib-arroyo-performance](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/docs/plans/2026-11-08-ta-lib-arroyo-performance.md) section 5.2.
- Current behavior:
  `docs/relational-frontends.md#fixed-utc-event-windows`,
  `docs/streaming-guide.md#event-time-windows`, and
  `docs/runtime-envelope.md#operator-tasks-and-barrier-alignment`.
- Current implementation anchors:
  - `runtime/streaming/gather_work.rs:65-99` and
    `runtime/streaming/entity_work.rs:363-438` own separate CPU-work systems;
    the latter launches rolling lanes through `spawn_blocking`.
  - `operator/rolling/kernel.rs:70` and
    `operator/rolling/kernel/entity_parallel.rs:30` set the old 64,000-row and
    16,000-row-per-lane thresholds.
  - `operator/window.rs:335-400` stores per-window row objects;
    `operator/window.rs:1793-1902` closes and emits the ordered prefix;
    `operator/window.rs:2001-2148` contains the async stream/checkpoint path.
  - `Cargo.toml:48-60` forbids unsafe code and has no target-CPU override;
    `pyproject.toml:49-53` builds the single `calc_flow._native` module.
  - `.github/workflows/release.yml:35-129` and `:210-278` build, verify, and
    publish the existing portable wheels; `AGENTS.md#release-invariants`
    requires one native module in a core wheel.

## Problem Statement

Streaming execution still has one logical owner per operator and several
operator-specific ways to outsource CPU work. The withdrawn parallel SUM and
sharding attempts paid serial routing, copying, and merging costs on every
batch, while the measured 45-88 us join dispatch and flat 1-to-32-CPU join
scaling show that more workers alone do not help. Window aggregation expands a
hopping row into as many as 1,024 open-window accumulators. P6/P7 must replace
these mechanisms with partitioned synchronous ownership, bounded stateless
morsel execution, pane state, and safe SIMD delivery, then remove the legacy
paths rather than preserve another permanent fallback.

## Fixed Decisions

1. Keep `unsafe_code = "forbid"`. SIMD comes from autovectorizable safe Rust
   loops and x86-64-v3 wheel builds, not intrinsics or an unsafe kernels crate.
2. Use N fixed, key-partitioned synchronous pipeline instances for admitted
   large-batch work. Route once at ingress and merge deterministically at the
   output boundary. The small-batch N=1 path runs on the caller thread.
3. Use one unified, bounded morsel pool only for stateless large-batch CPU
   work. Tokio owns asynchronous source, sink, store, and persistence I/O, not
   inline CPU execution.
4. Publish a separate, explicitly installed `calc-flow-python-v3` wheel
   distribution. Keep `calc-flow-python` as the portable default. Do not put
   baseline and v3 native modules in one wheel and do not auto-select a native
   module at import time.
5. Implement slide-aligned panes on the P4 keyed-state core. Land COUNT, MIN,
   and MAX first. Admit SUM and AVG only after the merge-order and error
   equivalence requirements below are proven.
6. Finish with one synchronous execution path and the manifest-v4/state-table
   format. Old checkpoints and old operator codecs are not readable.
7. Do not change any approved P0-P5 gate, the P6 four-core target, the P7
   hopping-improvement target, or the separate Arroyo acceptance gate.

## Goals

- Scale an admitted keyed 10M-row workload to at least 3x its N=1 throughput
  when both arms use the same four-CPU resource envelope and the candidate uses
  four partitions.
- Preserve the small-batch caller-thread path and show no confirmed regression
  under the maintained paired regression rule.
- Replace `gather_work`, `entity_work`, rolling lanes, and scattered CPU
  `spawn_blocking` with the unified pool without changing control or recovery
  semantics.
- Reduce high-overlap hopping work with pane state while preserving window
  membership, lateness, output order, budgets, and recovery.
- Deliver opt-in x86-64-v3 wheels without weakening portable-wheel support or
  the single-native-module invariant.
- Delete the legacy async operator paths and all superseded state codecs in
  the same integrated PR.

## Non-Goals

- Dynamic resharding, changing a live checkpoint's partition count, distributed
  execution, or a second repartition/exchange inside a pipeline.
- Implicit data broadcast to make an otherwise non-partitionable plan parallel.
- Session windows, new aggregate functions, early window output, retractions,
  multi-level panes, or a new `BatchKind`.
- Runtime CPU dispatch between two native modules, a universal v3-only wheel,
  removal of portable wheels, or a v3 source distribution that forces v3 on
  source builds.
- Backward-readable manifest v3 or legacy operator state, compatibility shims,
  or preservation of `tests/fixtures/v1/` as a runtime path.
- Changes to source cursors, sink delivery proof, at-least-once duplicate
  rules, project-v3 UDF references, or the public window declaration syntax.

## Functional Requirements

### Partitioned synchronous execution

- **FR1 - Fixed ownership and admission.** An execution instance has a fixed
  partition count N and router identity before it admits state. Every admitted
  stateful node is partitioned by the same plan-time key contract: rolling by
  entity, grouped aggregation/window/distinct by group key, equijoin by its
  compatible join keys on every input, and ASOF by entity key. Every input of
  a multi-input node maps the same logical key to the same partition. Global
  aggregates, ungrouped windows, cross-section, global order, snapshot SQL,
  unknown key lineage, volatile or otherwise order-sensitive work, and plans
  requiring another exchange use N=1. N=1 is the same synchronous core, not a
  legacy executor.
- **FR2 - No live topology change.** N and the partitioner version do not
  change after state is admitted. Batch size may suppress parallel dispatch,
  but may not move keys between owners or turn an existing partitioned state
  into a different topology. `StreamSession` retains its N=1 caller-thread
  contract. The default small-batch path performs no worker launch, future
  poll, Tokio entry, or cross-thread wait and retains the P2 allocation gates.
- **FR3 - One-time ingress routing.** A data row is hashed and assigned at the
  pipeline ingress exactly once. Per-partition input carries its original
  ingress ordinal for merging. Internal operators consume the owner's existing
  key/group id; they do not re-route rows, rebuild a global key dictionary, or
  clone the full input for every partition. Multi-input joins route each input
  once with the same partitioner. Routing, output materialization, and merging
  are inside every performance timing boundary.
- **FR4 - Deterministic ordered merge.** For the same compiled declaration,
  canonical input messages, and seed, the observable output is identical at
  N=1 and N>1 and independent of worker completion order. Equality includes
  output name, message/chunk boundaries, schemas, batch metadata and sequence,
  row order, null bitmaps, string bytes, integer values, and floating values
  compared by bit pattern. Existing row-origin outputs retain canonical input
  row order; aggregate/window outputs retain their canonical key order; join
  and ASOF retain the P5 order contract. Worker scheduling cannot change the
  first visible error position or output prefix.
- **FR5 - Bounded backpressure.** Ingress mailboxes, per-partition work,
  outboxes, merge buffers, and morsel queues have explicit row/byte/count
  bounds. A full owner or blocked output prevents further ingress admission;
  no unbounded per-partition staging is allowed. Existing envelope admission
  and output chunk limits apply before and after partitioning respectively.
  Backpressure releases after capacity returns and does not reorder messages.
- **FR6 - Cancellation and failure settlement.** Cancellation is checked at
  message entry and at least every 8,192 rows or morsel. Cancellation or the
  first failure stops new work, signals running owners/morsels, joins every
  launched worker, releases reservations and frozen-page pins, and only then
  completes job/session teardown. No later watermark, barrier, EOF, manifest,
  or source acknowledgement is emitted for the failed cut. Already accepted
  sink output retains the existing delivery contract; no new cross-partition
  atomic-delivery promise is added.

### Control alignment

- **FR7 - Watermarks.** The runtime broadcasts the same typed watermark to
  every partition after all preceding data has been processed and merged. Each
  partition runs its synchronous watermark callbacks and emits closed results;
  the merger emits those results in FR4 order before forwarding one aligned
  watermark. Multi-ingress minimum/idle rules and strict-advance errors remain
  unchanged. A fast partition cannot advance the global frontier alone.
- **FR8 - Barriers.** A barrier for epoch E follows all pre-E data in every
  partition. Each owner freezes its state-table pages, dirty bitmap, progress,
  and merger cursor for E before acknowledging the aligned cut. The coordinator
  forwards exactly one barrier only after all required partitions acknowledge;
  persistence then proceeds under P3's single-in-flight-epoch rule. Post-E data
  cannot overtake E at the merged boundary. Capture, persistence, checksum, or
  publication failure publishes no manifest for E and converges under P3's
  failure model.
- **FR9 - Idle and EOF.** Idle preserves the existing progress-driver semantics
  and is never treated as a watermark or barrier. EOF is broadcast only after
  every required ingress has explicitly ended; every operator instance runs
  `on_end` exactly once, all final outputs are merged in FR4 order, and one EOF
  is forwarded per output. A closed channel without explicit EOF remains a
  failure, not synthetic completion.

### Unified morsel pool and I/O separation

- **FR10 - One job-scoped compute budget.** A fixed, bounded pool serves all
  admitted stateless large-batch CPU morsels for a job. A morsel owns its input
  or holds safe immutable/shared references and returns owned output; it cannot
  mutate operator/keyed state, perform I/O, call Python, or wait on nested pool
  work. Small or ineligible work executes inline in the owning synchronous
  pipeline. Pool admission accounts scratch and output memory before launch and
  refunds it on every exit path.
- **FR11 - Replacement and separation.** Migrate join/ASOF/order/SQL gather and
  data-work, rolling lanes, and CPU-side checkpoint encode/hash work to FR10,
  then remove `gather_work`, `entity_work`, operator-local worker systems, and
  their metrics/fallback counters. Replace direct blocking filesystem wrappers
  with async I/O rather than sending I/O to the compute pool. Tokio source,
  sink, store, and persistence tasks never execute large CPU loops inline, and
  compute threads never perform connector or filesystem I/O. No direct
  `tokio::task::spawn_blocking` CPU call sites remain in the core.

### Checkpoint capture and restore

- **FR12 - Per-partition v4 capture.** Manifest v4 records N, partitioner
  version/identity, plan and state-layout identities, and one set of state-page
  handles per partition. Capture includes each owner's key store, keyed state,
  undo/transaction boundary, input/output progress, terminal flags, and merge
  sequence state. Frozen pages are immutable and hashed/persisted once; the
  manifest is published only after every partition handle and all integrity
  checks complete. Sink commit and source acknowledgement retain P3 ordering.
- **FR13 - Restore and incompatibility.** Restore validates the complete v4
  identity and every partition before opening sources or acknowledging output.
  It requires the stored N and router identity; a mismatch fails closed with a
  checkpoint mismatch and does not silently reshard. Restoring and continuing
  produces the same output bits/order and subsequent checkpoint bytes as an
  uninterrupted run. Manifest v3, legacy operator layouts, and unknown v4
  partition/layout versions fail closed with an actionable new-lineage error.

### SIMD and wheel packaging

- **FR14 - Safe vectorizable kernels.** The shared P2/P4 accumulator loops keep
  contiguous typed columns, plan-time specialization, and branch-free fast
  paths so release builds can autovectorize. No per-row enum dispatch, unsafe
  block, target-feature intrinsic, or `target-cpu=native` is introduced. Only
  v3 wheel build jobs set `target-cpu=x86-64-v3`; the repository and portable
  builds retain their existing target settings.
- **FR15 - Separate v3 distribution.** `calc-flow-python-v3` has the same
  calendar version and `calc_flow` import package as `calc-flow-python` and is
  explicitly installed in a clean environment. Build v3 wheels for the three
  existing x86-64 targets (Linux, macOS, and Windows), with the same six Python
  tags and two abi3 build tiers as the portable release. AArch64 and all
  `calc-flow-python` wheels remain unchanged and portable. No v3 sdist is
  published. The two distributions are mutually exclusive; importing an
  environment in which both distributions are installed raises an actionable
  error instead of silently using overwritten files.
- **FR16 - Artifact contract.** Every portable or v3 wheel contains the
  `calc_flow` package, exactly one `calc_flow._native` abi3 module, metadata/
  SBOM, and license, with no second native module or runtime selector. The
  release workflow verifies both matrices and their hashes before publishing
  either distribution. The v3 artifact inspection records the v3 target and
  confirms vector instructions in at least one designated P2/P4 numeric loop;
  the portable artifact has no v3 target override. A partial PyPI upload follows
  the existing immutable-version/new-version procedure.

### Pane-based window aggregation

- **FR17 - Pane geometry and update.** For window size W, slide S, and overlap
  k=W/S, pane p is the half-open epoch-aligned interval
  `[p*S, (p+1)*S)`, using mathematical floor for negative timestamps. A valid
  input row updates exactly one `(group, pane)` per declared aggregate instead
  of k per-window accumulators. A window `[w, w+W)` merges exactly panes
  `w/S` through `w/S + k - 1`, in increasing pane-start order. Rows within a
  pane retain canonical input order. Pane existence is based on a valid group
  row, not on a non-null aggregate operand, so all-null groups and empty-window
  rules remain unchanged.
- **FR18 - Membership, close, and output.** Each target window retains its own
  late test `end <= current_input_watermark`; a late row may still update panes
  needed by other open windows and late metrics count the same dropped
  assignments as before. A real watermark closes and emits every nonempty
  window with `end <= watermark`; EOF drains all remaining windows. Output
  columns, nullability, types, chunking, and ordering by window start, window
  end, then native stable group-key encoding are unchanged. No early output or
  retraction is added.
- **FR19 - Retirement and budget.** Pane p remains available until every window
  it can contribute to has closed, including the latest target window starting
  at `p*S` with end `p*S+W`; failed output/close atomically restores the exact
  pre-call state under the state-table transaction rules. `StateBudget.rows`
  continues to count logical open `(window bounds, group)` pairs, not physical
  panes. `StateBudget.bytes` covers actual pane/key storage and merge scratch;
  admission preflights both logical and physical usage and never silently
  rescales or weakens the configured budget because the representation changed.
- **FR20 - COUNT, MIN, and MAX first.** COUNT ignores null operands and uses
  checked u64 addition during both update and merge. MIN/MAX ignore nulls and
  preserve the existing total comparison behavior for every supported type,
  including NaN, infinities, signed zero, strings, and temporal values. Pane
  close and N=1/N>1 execution produce the same values and bits as the pre-pane
  sequential reference, with identical output order and errors.
- **FR21 - SUM/AVG admission proof.** SUM and AVG use the shared accumulator
  module and the fixed order in FR17. Before either is enabled on pane state,
  focused differential/property evidence and the accompanying numeric argument
  must prove, for every supported input type:
  - signed/unsigned integer SUM and the AVG numerator retain widened exact
    arithmetic, output-range checks, intermediate-overflow behavior, and the
    same first error at the same canonical row and callback;
  - floating SUM/AVG retain the current sequential window operator's output
    bits and null/NaN/infinity/signed-zero/overflow behavior, rather than merely
    approximately agreeing after reassociation;
  - count overflow, all-null operands, empty panes, merge/readout order, and
    checkpoint continuation are covered;
  - results are independent of batch boundaries, N, worker schedule, and
    checkpoint/restore.

  If this proof fails, stop that admission and escalate the numeric-contract
  conflict; do not silently change window numeric output, retain an async
  fallback, or claim P7 completion.
- **FR22 - One window core and state format.** Finite and streaming windows use
  the same pane kernels and keyed state; finite execution remains run-local and
  finalizes the complete input. Streaming panes reside in per-partition state
  tables and follow FR12/FR13. `WindowKey`/`AccumulatorRow` row storage,
  `StateOperationRow`, `WindowSnapshotMetadata`, retained delta/base segments,
  and window layouts 1/2 are deleted rather than kept as read-only recovery.

### Final migration and deletion

- **FR23 - Remaining operators.** Cross-section, distinct, bounded order, and
  snapshot SQL move to the synchronous operator trait and runtime-owned state
  tables. Their global or non-keyed semantics use N=1 where FR1 requires it.
  Batch and stream drivers share operator kernels; this is not a second state
  format or an async per-operator execution path.
- **FR24 - Legacy removal.** By the final P7 commit, remove the async
  `StreamOperator` execution trait and adapter, obsolete `CompiledStreamOperator`
  variants and operator-task graph, `gather_work`, `entity_work`, rolling lane
  machinery, remaining operator-specific work pools/codecs, and all old state
  layout decoders. Sources, sinks, checkpoint persistence, and connector I/O
  remain async. No removed path is retained behind a configuration flag.
- **FR25 - Documentation and compatibility.** Update published runtime,
  streaming/window, checkpoint/recovery, installation, and release documentation
  plus the changelog in PR #619. State the new-lineage requirement, the fixed-N
  restore restriction, the separate v3 installation and CPU requirement, and
  the final single-path/single-format boundary. Project-v3 declarations and the
  existing public window syntax remain unchanged.

## Inputs and Outputs

| Name                    | Type / surface                         | Units              | Range / constraints                                             |
|-------------------------|----------------------------------------|--------------------|-----------------------------------------------------------------|
| `N`                     | Internal execution-partition count     | partitions         | Fixed before state admission; N=1 for small/non-keyed/session   |
| Keyed input             | Immutable table `Batch`                | rows / bytes       | Existing exact schema and ingress envelope budgets              |
| Watermark               | Typed `EventTime`                      | microseconds (UTC) | Existing strict advance and multi-ingress alignment             |
| Barrier                 | Typed `Epoch`                          | epoch              | One aligned cut; at most one in-flight persistence epoch        |
| Window size `W`         | Existing `WindowSpec`                  | microseconds       | Positive exact integer; existing overflow rules                 |
| Window slide `S`        | Existing `WindowSpec`                  | microseconds       | Positive; `W % S == 0`; `1 <= W/S <= 1024`                      |
| Window output           | Table `Batch`                          | rows / bytes       | Existing schema, order, nullability, chunk and delivery rules   |
| Portable wheel          | `calc-flow-python`                     | artifact           | Existing targets; one native module; no v3 target override      |
| SIMD wheel              | `calc-flow-python-v3`                  | artifact           | x86-64 only; explicit install; one v3 native module             |
| Recovered state         | Manifest v4 plus per-partition pages   | bytes / epoch      | Exact N/router/layout match; old formats rejected               |

## Delivery Sequence in PR #619

Each behavior-changing step begins with the focused failing tests required by
`AGENTS.md`. These are commits in one PR, not independently merged stage PRs.

1. Add N=1 partition-executor equivalence tests, then fixed ownership,
   one-time routing, deterministic merge, and control alignment.
2. Add pool cancellation/backpressure/settlement tests, then the unified morsel
   pool and compute/I/O separation; migrate and delete old CPU-work systems.
3. Add per-partition v4 capture/restore and incompatibility tests, then connect
   partition ownership to P3 persistence.
4. Add artifact/coexistence/vector-inspection failures, then safe SIMD build
   delivery and the separate v3 wheel matrix.
5. Add COUNT/MIN/MAX pane differential, late, budget, close, and recovery
   tests, then pane state on the keyed core.
6. Add the complete FR21 SUM/AVG proof corpus before admitting those pane
   merges; stop on any numeric or first-error mismatch.
7. Migrate the remaining operators, delete legacy async paths/codecs, and run
   the final single-path and documentation checks.

## Acceptance Criteria

### Correctness, lifecycle, and recovery

- [ ] AC1: Given admitted rolling, grouped SUM, window, join, and ASOF graphs,
      routing counters and ownership assertions show one ingress routing per
  row, one owner per key, no per-node re-routing, and no full-input clone per
  partition; ineligible graphs execute through the same synchronous core at
  N=1.
- [ ] AC2: Differential tests compare complete outputs at N=1, N=2, and N=4
  under forced out-of-order worker completion. FR4 equality passes for values,
  float bits, nulls, rows, chunks, metadata, output/control order, and first
  error; repeated runs are byte-for-byte deterministic after canonical
  serialization.
- [ ] AC3: A small-batch N=1 session/pipeline test proves caller-thread
  execution, zero worker/future/Tokio activity, unchanged P2 allocation
  budgets, and identical results before and after P6.
- [ ] AC4: Watermark tests with uneven partition work prove that all preceding
  data and all closed results precede exactly one aligned watermark and that no
  partition can independently advance the output frontier.
- [ ] AC5: Barrier tests prove one per-partition capture at the same epoch,
  one forwarded barrier, correct FIFO around post-epoch data, and no manifest
  when any capture, write, checksum, or publication step fails. A second
  barrier backpressures while the prior epoch is unfinished.
- [ ] AC6: EOF tests cover partial-input EOF, explicit all-input EOF, and a
  dropped channel. `on_end` and final output occur exactly once only in the
  explicit all-input case; no synthetic EOF follows a dropped channel.
- [ ] AC7: Cancellation tests at ingress, inside an 8,192-row morsel, during
  barrier capture, and during background persistence stop new work, join all
  owners/morsels, release memory/page pins, and emit no later control,
  acknowledgement, or manifest. A blocked-sink test proves bounded queues and
  clean resumption without reordering.
- [ ] AC8: Injected worker panic and ordinary error tests settle every sibling,
  preserve the canonical first failure/prefix, leave no running worker or
  reservation, and recover only from the last published manifest.
- [ ] AC9: Checkpoint round trips at N>1 cover data/watermark/barrier/EOF
  progress and resume with FR4-identical output and identical subsequent v4
  checkpoint bytes. Changed N, router identity, layout, corrupt/missing
  partition pages, manifest v3, and every deleted operator codec fail before
  source open or external acknowledgement.
- [ ] AC10: Structural runtime tests show that CPU loops never occupy Tokio
  workers, compute workers perform no I/O, nested pool submission cannot
  deadlock, all queue/reservation bounds hold, and no legacy pool or direct CPU
  `spawn_blocking` path remains.

### Window and final migration

- [ ] AC11: Pane assignment tests cover tumbling, hopping overlap 2 and 1024,
  negative event times, null times, null groups, all-null operands, empty input,
  and event-time arithmetic overflow. Each valid row updates one pane and each
  emitted window merges exactly its k ascending panes; no empty window is
  invented.
- [ ] AC12: Late/watermark/EOF tests match the pre-pane operator's dropped
  assignments and metrics, including a row with both closed and open target
  windows. Close order, output schemas/types/chunks, EOF drain, and rejected-
  output rollback are unchanged.
- [ ] AC13: COUNT/MIN/MAX differential tests match the pre-pane sequential
  reference bit-for-bit across batch splits, N, and checkpoint/restore, with
  explicit NaN, infinities, signed zero, strings, temporal values, and count
  overflow cases.
- [ ] AC14: SUM/AVG are admitted only with the complete FR21 evidence. Integer
  intermediate overflow and first-error location, floating output bits,
  null/NaN/infinity/signed-zero behavior, AVG readout, and checkpoint
  continuation all match the current sequential reference. Any mismatch blocks
  P7 completion.
- [ ] AC15: Budget tests show that logical open-window row counts are unchanged
  by pane representation, physical pane and merge-scratch bytes are admitted
  and refunded, and budget failure leaves no partial state. Pane retirement is
  correct at equality watermarks and after failed emission.
- [ ] AC16: Finite and streaming window executions share the pane kernel and
  satisfy the same oracle; finite state remains run-local. Cross-section,
  distinct, order, and snapshot SQL run only through the synchronous/state-table
  core after migration.
- [ ] AC17: The final integrated diff contains no async stream-operator
  execution path, old runner operator-task graph, `gather_work`, `entity_work`,
  rolling lanes, operator-specific compute pool, or superseded state decoder.
  Old checkpoints fail with the documented new-lineage error;
  `tests/fixtures/v1/` is unchanged.

### Packaging, documentation, and verification gates

- [ ] AC18: Artifact inspection accepts both distributions only with exactly
  one native module per wheel, matching calendar versions, expected Python/
  platform tags, metadata/SBOM/license, and no runtime selector. Installing both
  distributions produces the specified coexistence error.
- [ ] AC19: A v3 wheel passes the installed-wheel smoke on x86-64-v3 hardware
  and vector inspection finds the designated autovectorized loop. The portable
  `calc-flow-python` wheel passes `scripts/smoke_wheel.py` in a clean x86-64
  environment without AVX2; an unavailable non-v3 environment leaves this gate
  unresolved rather than proving portability from a v3 host.
- [ ] AC20: Published docs and changelog cover all FR25 items. Focused Rust/
  Python/release tests and necessary formatting/lint/type checks pass; full
  regression, the 90% Rust coverage floor, and required cross-platform release
  checks run in CI. Pending CI is not merge readiness.

## Pre-Registered Measurement and Stop Rules

Freeze exact commit SHAs, wheel/native hashes, harness identities, machine and
CPU affinity before execution. The five cases below cannot be replaced or
expanded after results are seen. Each timed case uses one excluded warmup per
worker and two rounds of ten alternating AB/BA pairs, retaining the maintained
full-output oracle, resource, provenance, and paired-interval rules. All five
cases share 600 seconds including fixtures, startup, correctness checks,
restore where needed, and cleanup; release builds are estimated separately.

| Case                    | Baseline / candidate                                      | Fixed workload and boundary                                      | Acceptance                                                   |
|-------------------------|-----------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------|
| M1 keyed scaling        | Same P6 build and wheel, N=1 / N=4, both pinned to 4 CPUs | 10M rows, 100x100 groups; ingress through route, state, merge    | Paired median N=1/N=4 wall-time ratio >=3.0 in each round    |
| M2 small-batch control  | Sealed pre-P6 integrated commit / P6 N=1 candidate        | 640-row grouped SUM; complete synchronous call and returned output | No confirmed regression under the maintained +5% rule        |
| M3 dense hopping panes  | Sealed pre-P7 commit / P7 pane candidate, both N=1        | 1M rows, 64 groups, W=60s, S=1s; all five aggregates through close | Maintained verdict `improved` and exact required oracle      |
| M4 low-overlap control  | Sealed pre-P7 commit / P7 pane candidate, both N=1        | 1M rows, 64 groups, W=2s, S=1s; all five aggregates through close  | No confirmed regression under the maintained +5% rule        |
| M5 portable wheel smoke | Existing portable release / new portable release          | One clean non-AVX2 x86-64 install and `scripts/smoke_wheel.py`   | Exit 0 and total `[3]`; no v3 wheel or native module present |

M1 includes all routing and ordered-merge costs; a kernel-only speedup does not
satisfy it. M3 may claim improvement only after SUM/AVG satisfy FR21; otherwise
report the incomplete admission and stop. Inconclusive timing does not prove a
speedup, equivalence, or regression. Stop on deadline, host interference,
identity mismatch, or oracle failure; settle owned processes, preserve partial
evidence, and do not add samples or retry for a favorable verdict. Scheduled
broad benchmarks and the separate Arroyo gate retain their own scope.

## Open Questions

None for the P6/P7 contract. P3 owns the detailed persistence failure model
that FR8/FR12 consume. Measurement thresholds and admission constants must be
recorded before execution and tuned only within the fixed five-case budget;
they are not permission to change the approved gates.
