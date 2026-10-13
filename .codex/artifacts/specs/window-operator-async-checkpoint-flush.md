# Window Operator Asynchronous Checkpoint Flush

| Field             | Value                                                          |
| ----------------- | -------------------------------------------------------------- |
| Status            | DRAFT - requires review before any implementation              |
| Artifact slug     | `window-operator-async-checkpoint-flush`                       |
| Project format    | `3` (unchanged)                                                |
| Native dependency | existing window delta checkpointing and v3 manifest recovery   |

## 1. Background and evidence

Profiling of the streaming `window_sum` workload (10M rows, shared WSL2 host,
checkpoint trigger every 200 ms / 1 s) shows the checkpoint flush blocking the
data path:

- Off: 1,010 ms (100%). Enabled, not triggered: 1,616 ms (62%).
- Every 1 s, state on disk: 1,930 ms (52%). Every 200 ms, on disk:
  2,719–3,034 ms (33–37%). Every 200 ms, state on tmpfs: 1,675 ms (60%).

The tmpfs/disk gap isolates storage as the bottleneck: each checkpoint stalls
the stream for 40–240 ms while segments are written, fsynced, validated, and
renamed. The engineering reference target (set for joins in
`.codex/artifacts/analysis/issue406-join-hotpaths.md`) is sustained throughput
with checkpoints of at least 90% of disabled at 1 s and 75% at 100 ms.

## 2. Current protocol (status quo)

`complete_barrier_if_ready` in
`crates/calc-flow/src/runtime/streaming/operator_task.rs` runs strictly in
order: capture snapshot → encode envelope → **stage operator state** (write +
fsync + validate + rename + directory sync, awaited inline at
`stage_operator_snapshot`) → ack coordinator → forward barrier downstream.
While staging, the operator task observes only job cancellation; ingresses are
not read.

Executable pins of this order:

- `operator_task/tests/segment_publication_tests.rs::multi_segment_publication_must_finish_before_operator_ack_or_barrier`
- `operator_task.rs::barrier_alignment_blocks_only_arrived_ingress_until_snapshot_ack`
- `runtime/streaming/runner/tests.rs::manual_checkpoint_returns_only_after_durable_manifest_and_sink_commit`
  and the AC14 fault-matrix recovery tests.

Documentation pins: `docs/runtime-envelope.md` ("the task awaits checkpoint
preparation, captures and stages operator state, acknowledges the coordinator,
then forwards the barrier"; "These operations complete before the participant
acknowledges the checkpoint") and `docs/design.md` ("Checkpoint transaction",
steps 3 and 5).

## 3. Proposed change

Split the operator's synchronous duty at the barrier into two phases:

1. **Bytes-ready phase (inline)**: after `prepare_checkpoint_async` and
   `checkpoint(epoch)` have produced immutable segments (the window operator
   already hands over fully encoded bytes), the operator task submits the
   segments to the manifest transaction for background staging and immediately
   acks + forwards the barrier. The ack carries a durability handle instead of
   a completed `OperatorManifestEntry`.
2. **Durability phase (background)**: the transaction finishes write/fsync/
   validate/rename on blocking workers and resolves the handle. Epoch manifest
   publication awaits every participant's resolved handle, so the durable
   commit point (manifest last, then sink commit and source cursor ack) is
   unchanged.

Exactly-once rests on the manifest being the commit point; that ordering is
preserved. What changes is only *when the operator task unblocks*, not what
must be durable before the epoch completes.

## 4. Semantics and risk analysis

- **Recovery**: a crash between ack and durability resolution loses the
  in-flight epoch, which never publishes a manifest — recovery replays from
  the last durable epoch, exactly as a failed checkpoint today.
- **Cancellation**: background staging must settle under `owner_settled`
  semantics before lineage ownership is released, matching
  `state/transaction.rs` guarantees; a cancelled job still awaits admitted
  writes.
- **Memory**: `WorkingStatePins` (or equivalent) must pin segment bytes until
  durability resolves; backpressure budgets need to account for one extra
  in-flight flush per operator.
- **Failure propagation**: a background flush failure must fail the job before
  manifest publication and must surface the same error classification
  (`ManifestPublication::Installed` recovery intent) as the synchronous path.
- **Protocol surface**: the ack payload shape changes; coordinator admission
  of unresolved acks, duplicate/conflicting ack rules, and the terminal
  checkpoint path need explicit transitions.

## 5. Required updates if approved

- `docs/runtime-envelope.md` barrier-alignment and manifest sections;
  `docs/design.md` checkpoint transaction steps.
- `StreamOperator` rustdoc (`operator/stream.rs` prepare/capture contract).
- New executable pins replacing the tests in §2: ack may precede durability,
  manifest publication never precedes durability resolution, background flush
  failure fails the epoch and recovers exactly-once.

## 6. Pre-implementation measurement gate

WSL2 directory sync is known-slow. Before prioritizing implementation, rerun
the 100 ms-interval case once on the target production storage class and
confirm the on-disk/tmpfs gap reproduces there. If the gap does not reproduce,
this proposal drops in priority.

## 7. Adjacent follow-up (separate work item)

`RollingOperator` and `CrossSectionOperator` still encode a full-state base
segment at every checkpoint (`rolling.rs`, `cross_section.rs`). Adopting the
window operator's mark-and-defer net-delta encoding (and, for cross-section,
the per-row bucket-division cache) would bring their checkpoint-on overhead
in line with window and join. This requires its own state-layout design and
is intentionally out of scope here.
