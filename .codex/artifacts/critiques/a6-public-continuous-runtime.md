# Public Continuous Streaming A6 - Critic Critique

## Target

- PR snapshot: `#113@d1d9f4673aa6f62d3fd4c0a45597b8a49c0d7b46`
- Baseline: `main@aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`
- Spec: `.codex/artifacts/specs/a6-public-continuous-runtime.md`
- Reviewed spec SHA-256:
  `d2730be9dd135ad95e57db052b87e515ec722d468b4c4bc9b34dab26b9735948`
- API note: `.codex/artifacts/api-notes/a6-public-continuous-runtime.md`
- Reviewed API-note SHA-256:
  `faae44001e1d85a579fa85ae6c16c15c353d770d0bb0238c60e1e611c779d40e`

The review also checked the governing streaming and compatibility claims in
`docs/introduction.md`, the private runtime analogues and tests in
`runtime/streaming/{runner,job,source_task,sink_task}.rs`, the #108 checkpoint
coordinator, and the manifest selector/publication transaction. This verdict
is for the exact snapshot and hashes above, not for an unreviewed rewrite.

## Verdict

**Proceed with caveats**

All six requested axes have zero design blockers. The revised pair accurately
projects the post-#108/#110 runtime while making the public owner, completion,
recovery, capability, redaction, and cutover contracts testable across Rust,
PyO3, and Python. It is ready for the documentation-only PR correction and
final review; it is not runtime-implementation evidence or merge approval.

## Six-Axis Conclusions

- **Ownership — Zero blockers.** Rust exposes one consuming
  `StreamingRunner::start(self)` and one non-cloneable `StreamingJob`; Python
  enforces the same one-shot transfer and defines cancellation/GC at every
  observer boundary. `ManagedCheckpointRuntime::new(managed_root)` is the
  only owner constructor, derives `state` and `manifests`, retains both the
  complete-root and lineage leases until settled cleanup, and has no erased
  backend, child-root, two-root, PyO3, or Python escape signature. The
  start-gate and reaper requirements extend the private ownership tests in
  `crates/calc-flow/src/runtime/streaming/runner.rs` without creating a second
  lifecycle owner.
- **Checkpoint completion — Zero blockers.** The note preserves #108's one
  bounded FIFO and its actual completion seam: the manual waiter resolves at
  `CheckpointAdvancement::Completed`, after `ManifestDurable` and every
  expected sink commit acknowledgement
  (`runtime/streaming/checkpoint/coordinator.rs`). `absent`, installed with
  unknown durability, and `durable` are not conflated; a parent-sync failure
  after rename returns the epoch-bearing
  `checkpoint_publication_unknown` error, advances no completed epoch, commits
  or aborts no preserved intent, and requires recovery. Success remains
  irreversible before later retention or compaction.
- **Recovery — Zero blockers.** Automatic start distinguishes empty lineage
  from resume, validates every canonical candidate in ascending order, fails
  closed on an invalid lower or higher candidate, and selects the highest only
  when all candidates validate. That matches the selector loop and sparse/
  corrupt candidate tests in `crates/calc-flow/src/state/transaction.rs`.
  Terminal resume does not poll or re-emit, and each connector receives a
  fresh sink-scoped recovery value rather than the baseline's complete
  `CheckpointManifest`; the cross-process Rust/Python test plan verifies that
  this projection does not become a second recovery truth.
- **Capability proof — Zero blockers.** Cursor ownership is bound to the exact
  source ID on configured, restored, and emitted paths. The complete source
  descriptor is sampled once before open and fingerprinted; public exact
  replay requires explicit `exact_pause_report_and_seek` plus lossless
  delivery, with no boolean fallback. Proof is output-scoped over reachable
  sources, operators, edges, and sinks, and includes adversarial foreign-
  cursor, descriptor-drift, ignored-restore, lossy-source, volatile-UDF, and
  bounded-retention cases. This tightens, rather than contradicts, the #110
  sampling and identity logic in `runtime/streaming/source_task.rs` and
  `runtime/streaming/progress/prepare.rs`.
- **Redaction — Zero blockers.** Every fallible A6 owner/runner/job operation
  has exactly one public Rust failure shape,
  `CalcFlowError::Streaming(StreamingError)`. Construction is private, the
  category mapping is exhaustive, normal/alternate `Debug`, `Display`, safe
  serialization, and complete `source()` chains are specified, and Python
  receives only `StreamingRuntimeError` or the publication-unknown subclass
  with no retained `__cause__` or `__context__`. Status, outcomes, logs,
  metrics, diagnostics, manifests, and two-sink recovery each have explicit
  positive and negative censuses, so a harmless-looking display sanitizer
  cannot hide a raw-bearing alternate path.
- **Cutover — Zero blockers.** The replacement inventory is atomic across
  crate exports, native registration, Python exports and `_native.pyi`, the
  Studio-private v2 document store, examples, benchmarks, active docs, clean
  package/release smoke, and removed-symbol negative tests. It preserves the
  batch plan, project v2, manifest v3, Studio `/api/v2` routes, OpenAPI, and
  generated TypeScript while removing legacy runner/checkpoint names rather
  than leaving aliases. The 48 named fault cases, bidirectional restart E2E,
  and exact-head soak cannot be satisfied by enumeration or stubs.

The proposed connector surface continues to move or borrow immutable `Batch`
envelopes as required by `docs/introduction.md`'s Batch contract. It neither
moves table calculation outside DataFusion nor introduces a Python expression
evaluation path.

## Findings

### Blocking Issues

None.

### Significant Concerns

- **The exact PR head still contains stale approval-state prose.** At
  `d1d9f467…`, the spec's `Open Questions` still says the refreshed API note
  is stale, while the plan header and `docs/runtime-envelope.md` already call
  the old-baseline critique zero-blocker approval. These are delivery-state
  errors, not defects in the reviewed behavioral/signature pair. Replace the
  old critique with this artifact and make those three statements describe
  the same review state before resolving the four actionable PR threads. Any
  change to a controlling decision or public signature invalidates this
  verdict; deleting stale review bookkeeping does not.
- **Final merge must use a new exact-head review.** During this review,
  `main` advanced to `c93b2cf0fe2a9d6afa82c54102a4396ce93efc45` through PR
  #111. Its private sink changes fix capability selection in binding
  constructors and make sink IDs globally stable, which agrees with this API
  note's `SinkBinding::{ordinary, transactional, epoch_idempotent}` design and
  creates no blocker for the requested `aa3bbf0…` snapshot. It still means
  this critique is not evidence that a later PR head is mergeable or fully
  reviewed: the final reviewer must inspect the rebased/merged exact diff,
  unresolved actionable threads, review decision, directed documentation
  checks, and all required checks before exercising the user's merge
  authorization.

### Minor / Style Notes

- **Test hierarchical managed-root contention.** Fixed child names prevent a
  caller from supplying overlapping state and manifest roots to one owner,
  but two independent callers can still choose nested managed roots, such as
  `R` and `R/manifests`. The complete-root lease must reject an ancestor/
  descendant overlap before the nested files reach the baseline's fail-closed
  manifest scanner. Add an in-process and subprocess case beside the required
  same-root and symlink-alias tests. The existing exclusive-complete-root
  contract is sufficient; this does not require another API revision.

## Counter-Proposals

None. A single local namespace owner, one consuming runner-to-job transfer,
and one safe error seam are the smallest coherent public projection of the
implemented runtime.

## Questions for the Author

None.
