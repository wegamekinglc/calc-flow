# Continuous Streaming 3.0 — M5 Epoch Checkpoint Critique

| Field         | Value                                                                  |
| ------------- | ---------------------------------------------------------------------- |
| Status        | **Reviewed — no open specification blocker**                           |
| Baseline      | `main@a5fc2c395e347041f8d16384be99af7e23d2ebff`                        |
| Review target | M5 specification and API note for `m5-epoch-checkpoint`                |
| Review style  | adversarial design, recovery, ownership, boundedness, scope audit      |

## 1. Review target and method

This critique reviews:

- `.codex/artifacts/specs/m5-epoch-checkpoint.md`;
- `.codex/artifacts/api-notes/m5-epoch-checkpoint.md`;
- the historical M5 tasks in the continuous-streaming v3 plan;
- the current M3 progress and M4 state/window implementation;
- the legacy public checkpoint and runner surfaces.

The review attacks transaction ordering, replay cuts, barrier ownership,
multi-input alignment, terminal processing, partial sink commit, recovery
identity, boundedness, cancellation, public API leakage, and evidence quality.
“Resolved” means the planning package now contains a testable rule; it does not
claim the M5 implementation already exists.

## 2. Executive verdict

The historical M5 plan had enough direction to name the major components, but
it was not safe to implement literally against current `main`. M4 finalized
the manifest in a different module while keeping public v2 surfaces, M3 moved
source edge ownership into a live progress coordinator, and final-only windows
made a post-end barrier illegal. The old task list also left progress
durability, partial multi-sink commit, and runtime-configuration mismatch
ambiguous.

The M5 delta resolves those conflicts without widening scope. Its central
invariants are now:

1. the M4 manifest is the sole durable truth;
2. the global source cut is serialized through the progress edge owner;
3. operators block only ingresses that reached the epoch barrier;
4. sink pre-commit precedes manifest and external commit follows it;
5. post-manifest failure completes forward on recovery;
6. terminal output is checkpointed out of band without a post-end barrier;
7. M5 remains crate-private and public v2 remains intact until A6.

No specification blocker remains. Implementation review must still reject any
shortcut that violates the findings below.

## 3. Findings and resolutions

### F1 — “Replace checkpoint v2” would break the approved public firewall

**Risk.** The historical M5.1 file list says to replace `checkpoint.rs`, but M4
deliberately finalized v3 under `state/manifest.rs` while public v2 runners
still depend on `CheckpointStore`. Literal replacement would either break
public compatibility or force unreviewed A6 integration into M5.

**Resolution.** Specification D1 and API §§1–3 require reuse of the M4 model
behind an explicit legacy-v2 firewall. Production manifest operations are
added privately; public v2 replacement remains post-M5.

**Verdict.** Resolved.

### F2 — Runtime configuration was accidentally part of recovery identity

**Risk.** Current `ManifestExpectation::validate_identity` rejects a changed
`runtime_config_hash`. Operational changes such as checkpoint interval,
timeout, or bounded queue sizing would make otherwise compatible durable state
unrecoverable, contradicting the runtime envelope.

**Resolution.** D1 and API §3.1 retain the hash in canonical metadata but move
comparison to a non-failing diagnostic. Pipeline fingerprint, participant
sets, epoch, checksums, and handle ownership remain strict identity.

**Verdict.** Resolved. AC-05 requires a successful restore plus observable
mismatch, preventing an implementation from silently dropping the field.

### F3 — Serializing the M3 snapshot would persist unbounded history and an invalid clock

**Risk.** The exact M3 in-process snapshot contains execution trace, receipt
state, admission gates, and monotonic timer coordinates. Serializing it would
grow with runtime history and restore process-local time into a new process.

**Resolution.** D3 and API §4 define a bounded semantic projection using
existing source/operator manifest fields. Restore creates new trace and
receipt namespaces and re-arms timers from a new origin using full configured
delays while preserving watermark monotonicity.

**Verdict.** Resolved. No serializable `StreamProgressDriverSnapshot` is
permitted.

### F4 — A source task cannot safely inject a barrier after M3

**Risk.** `LiveProgressCoordinator` owns every source output sender and can
emit timer-driven controls. Allowing the source task or checkpoint coordinator
to send a barrier directly creates multiple writers and permits watermarks or
data to overtake the cursor cut.

**Resolution.** D5 and API §6 route settlement and fan-out through the progress
owner's existing serial drive boundary. Sources pause polling and provide
positions; the global coordinator never writes edges.

**Verdict.** Resolved. AC-11/AC-12 require adversarial timer and prefetch tests.

### F5 — A per-source cut is not necessarily a globally ordered progress cut

**Risk.** Pausing each source independently can still leave accepted raw
events, pending progress receipts, or ready timers inside the job-scoped
driver. A barrier emitted as soon as one source pauses would not describe a
replayable prefix across all bindings.

**Resolution.** D5 requires all pre-cut submissions to settle at one
progress-owner cut before any barrier fan-out. Ack cursor promotion happens
only after successful cut dispatch.

**Verdict.** Resolved. The implementation must not approximate “settled” with
an empty source-local prefetch slot.

### F6 — Barrier alignment can become an unbounded buffer

**Risk.** A common alignment implementation keeps reading the fast, already
barriered ingress and buffers its post-barrier messages while waiting for a
slow ingress. This bypasses the bounded edge budget and can reorder data.

**Resolution.** D6 and API §7 remove blocked ingresses from receive selection.
Backpressure remains in the existing bounded channel; no auxiliary alignment
buffer exists. Unblocked ingresses continue normally.

**Verdict.** Resolved. AC-19 and AC-26 make the ownership and memory bound
observable.

### F7 — Waiting for global manifest publication before forwarding multiplies alignment latency

**Risk.** Holding all operator inputs blocked through downstream pre-commit,
manifest fsync, and sink commit serializes alignment across the graph and can
turn checkpoint latency into sustained backpressure.

**Resolution.** D6 explicitly selects immediate barrier forwarding after the
operator's local snapshot/staging succeeds. Global failure later cancels the
job and uses abort/recovery rules.

**Verdict.** Resolved. The alternative is not left to implementation comments.

### F8 — Manifest publication after external sink commit creates unrecoverable output

**Risk.** If a sink commits externally before the recovery intent is durable,
a crash can expose data that no selected checkpoint records. Replaying the
previous epoch then duplicates output.

**Resolution.** D7 fixes ordering: all pre-commits, manifest durable, then
external commits. Before-manifest failure aborts; after-manifest failure
preserves and completes forward.

**Verdict.** Resolved. AC-29–AC-33 cover both sides of the rename boundary.

### F9 — Multi-sink commit cannot be atomically rolled back

**Risk.** After sink A commits and sink B fails, rolling A back may be
unsupported while replaying the whole epoch can duplicate A. A separate
“completion” file would introduce competing truths and its own crash window.

**Resolution.** D7 defines idempotent forward recovery from the one durable
manifest. Every sink can determine and complete epoch commit by stable
pipeline/sink/epoch identity. There is no second durable completion marker.

**Verdict.** Resolved. A sink unable to meet that recovery contract is not an
exactly-once sink.

### F10 — End-of-input makes a normal final barrier illegal

**Risk.** M3 forbids any message after committed `EndOfInput`, and M4 windows
emit final-only results from `on_end`. Injecting a final barrier after end
violates the data-plane protocol; checkpointing before end loses final output
and post-end operator state.

**Resolution.** D8 and API §12 define a coordinator-owned terminal epoch after
operators and sinks acknowledge the data-plane end cut. The manifest captures
post-`on_end` state and final pre-commit metadata without emitting another
edge message.

**Verdict.** Resolved. AC-37–AC-40 distinguish terminal and periodic paths.

### F11 — Recovery order can leak source or sink side effects

**Risk.** Current private runner resets operators and opens connectors before
there is manifest selection or restore. If source polling or sink writes begin
before all state is validated, a later restore failure corrupts the externally
visible prefix.

**Resolution.** D10 and API §11 define gated recovery: select and validate,
load segments, restore progress/operators, open sources paused and seek,
recover sinks, then spawn/release. Failures close in reverse ownership order.

**Verdict.** Resolved. AC-04 and AC-41 cover pre-side-effect validation and
cleanup.

### F12 — “Exactly-once sink” alone is not an exactly-once output proof

**Risk.** A transactional sink cannot compensate for a lossy/non-seekable
source, nondeterministic operator, volatile UDF, lossy edge, or operator that
cannot restore. A whole-job boolean also mislabels unrelated outputs.

**Resolution.** D9 and API §9 derive capability per requested output over the
complete reachable subgraph before lifecycle work. Ordinary output remains
explicitly at-least-once.

**Verdict.** Resolved. The first incompatible stable component appears in the
validation error.

### F13 — M5 could accidentally absorb production connector work

**Risk.** Testing transaction semantics against Kafka or object storage would
couple M5 correctness to M6 client dependencies, secrets, feature flags, and
external availability. Conversely, a memory-only fake cannot demonstrate
rename/restart crash boundaries.

**Resolution.** D9/D12 and API §8.2 use a deterministic filesystem-backed test
transactional sink with durable epoch staging and idempotent commit markers.
It remains private test support and makes no production connector claim.

**Verdict.** Resolved.

### F14 — Synchronous state capture can block Tokio during alignment

**Risk.** A locally aligned operator may serialize or hash a large state set on
the executor, stalling cancellation, other barriers, and sink progress. M4
already forbids bulk encode in `checkpoint()`, but production publication adds
more I/O and hashing.

**Resolution.** D2/D6/D11 and API §3.2 preserve metadata-only synchronous
capture and place filesystem, hashing, serialization, and large decode work on
true async I/O or owned blocking workers. Coordinator deadlines remain active.

**Verdict.** Resolved. Benchmark and paused-time responsiveness evidence is
required.

### F15 — Retention can delete state needed by an in-flight or retained epoch

**Risk.** Cleanup concurrent with publication may observe a segment before its
manifest, or delete an older base still referenced by a retained delta chain.
A broad retry after a link/delete error increases the blast radius.

**Resolution.** D2 reuses the M4 reachability and lineage lease rules,
serializes retention with publication, preserves retained/in-flight handles,
and fails closed on links or unexpected files.

**Verdict.** Resolved. AC-10 and the fault matrix cover retention/compaction
interleavings.

### F16 — Checkpoint status can leak connector payload or create cardinality explosion

**Risk.** Cursor and pre-commit metadata may contain operational identifiers or
secret-adjacent values. Using epoch/participant as labels creates unbounded
metrics cardinality during long jobs.

**Resolution.** D11 restricts status to phase, counts, timing, stable IDs,
failure category, and configuration mismatch. Payloads, paths, rows, and state
bytes are prohibited. Epoch is a status value, not a metric label.

**Verdict.** Resolved.

### F17 — Crate-private implementation can tempt a public test-only API

**Risk.** The easiest way to integration-test M5 would be to export source
bindings, checkpoint trigger, or private runner controls prematurely. That
would create an unsupported A6 surface and lock in incomplete semantics.

**Resolution.** Scope and API §14 put coordination tests beside private modules
and retain public integration tests only for already public M4 state values.
No provisional export is allowed.

**Verdict.** Resolved.

### F18 — Pre-push CI or a short happy-path run is not merge evidence

**Risk.** A final review fix changes the head after CI, Codacy, benchmark, or
soak evidence. A soak without restart, raw SHA, per-minute samples, and output
reconciliation cannot establish durable behavior.

**Resolution.** D12 requires all evidence on the exact final remote SHA and
invalidates it after any push. The 20-minute soak has 120 samples, deterministic
restart, zero duplicate/missing output, bounded resources, and terminal zero.

**Verdict.** Resolved. AC-43–AC-47 make merge readiness independently
checkable.

### F19 — The M5 introduction has no honest private main baseline

**Risk.** The exact introduction base
`main@972964413d328dfeabd4597088396bfe4516e5a3` lacks the private M5
checkpoint coordinator, production manifest transaction, and transactional
checkpoint paths. Treating missing base cases as zero, substituting a test
double, or reporting candidate absolute time as a private regression would
manufacture a favorable comparison. Conversely, allowing one shared-path
comparison to emit an M5-wide pass would overstate what the evidence proves.
The former phrase “120 one-minute samples in 20 minutes” was also internally
impossible and obscured the implemented process-restart schedule.

**Resolution.** Specification D12 amendment M5-D12-E1 and API §15 restrict
`B1 -> C1 -> B2 -> C2` to the named production edge-channel path present on
both refs and scope its result to `shared_edge_result`. The 12 private M5 cases
run as exact-final-candidate optimized P1/P2 `absolute_only`
characterizations, with candidate enabled/disabled cost labeled as
candidate-only self-overhead. Same-ref executables, repeatability, noise, host
stability, full provenance, immutable artifact coverage, and inconclusive WSL
or unstable-host handling are mandatory. There is no M5-wide `overall_pass`,
and the exact M5 merge SHA becomes the required baseline for later changes.

AC-46 now matches the process contract: one parent launches three distinct OS
children for global ranges `0..40`, `40..80`, and `80..120`; only filesystem
evidence crosses generations; and 120 10-second observations span 20 minutes
and may be aggregated into 20 one-minute intervals. The exception waives none
of the fault, correctness, coverage, CI, review, Copilot, Codacy, or soak gates,
does not widen the public A6/Python/Studio firewall, and does not claim the
final performance or soak runs have occurred.

**Verdict.** Resolved and approved as the one-time M5-D12-E1 introduction
baseline amendment. Reuse after the M5 merge is prohibited.

## 4. Decision and acceptance closure

| Decision area                     | Controlling rule | Observable closure                          |
| --------------------------------- | ---------------- | ------------------------------------------- |
| manifest reuse and v2 firewall    | D1               | AC-01–AC-05                                 |
| publication, selection, retention | D2               | AC-06–AC-10                                 |
| durable progress                  | D3               | AC-16–AC-18                                 |
| epoch coordinator                 | D4               | AC-14–AC-15                                 |
| exact source cut                  | D5               | AC-11–AC-13                                 |
| operator alignment                | D6               | AC-19–AC-26                                 |
| sink transaction                  | D7               | AC-29–AC-36                                 |
| terminal checkpoint               | D8               | AC-37–AC-40                                 |
| capability proof                  | D9               | AC-27–AC-28, AC-35–AC-36                    |
| recovery ordering                 | D10              | AC-04, AC-08–AC-09, AC-39, AC-41            |
| bounded lifecycle and diagnostics | D11              | AC-26, AC-41–AC-42                          |
| fault, benchmark, and soak gates  | D12/M5-D12-E1    | AC-43–AC-47                                 |
| milestone firewall                | Scope/D1/D9      | AC-02, AC-48                                |

Every controlling decision has at least one acceptance criterion and named
test/evidence placement. No criterion depends on an unapproved public API.

## 5. Implementation advisories

- Keep coordinator state transitions pure where possible; confine mutation to
  the one owned task and canonical `BTreeMap` participant sets.
- Use explicit bounded test gates at transaction boundaries. Do not use timing
  sleeps to infer that a source paused or a manifest renamed.
- Treat a manifest-durable commit failure as a recovery scenario, not as an
  excuse to delete the manifest or abort external state.
- Preserve original error sources and stable participant paths while redacting
  connector JSON and filesystem details from status.
- Measure the normal no-checkpoint path as well as checkpoint latency; a
  correct protocol that permanently regresses streaming throughput is not
  merge-ready.
- Keep work-package commits narrow and independently reviewable, but do not
  merge a partially enabled checkpoint protocol into `main`.

## 6. Final verdict

**APPROVE THE M5 PLAN FOR IMPLEMENTATION AFTER THIS DOCUMENTATION PR IS
REVIEWED.**

The delta is specific enough to drive test-first work and closes the known
current-main contradictions. M5-D12-E1 is approved only for the introduction
baseline and does not convert scoped or absolute performance evidence into an
M5-wide pass. Approval does not authorize a partial public runner or a
production connector claim. The final implementation still requires the
complete exact-head fault, benchmark, CI, review, Codacy, and 20-minute soak
gates; those final performance and soak artifacts are not yet claimed here.

**BLOCKS REMAINING: 0**
