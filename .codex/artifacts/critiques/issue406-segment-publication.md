# Issue 406 Segment Publication - Critic Critique

## Target

- Spec: [issue406-segment-publication.md](../specs/issue406-segment-publication.md).
- Baseline: `53f44fad`; review is of the proposal, not an implementation review.
- Domain: [basic vocabulary](../../../docs/introduction.md#the-basic-vocabulary),
  [manifest publication and recovery](../../../docs/runtime-envelope.md#manifest-publication-and-recovery),
  and [cancellation and ownership](../../../docs/runtime-envelope.md#cancellation-and-ownership).

## Verdict

**Proceed with caveats.** The revised specification resolves both blocking
issues and the cancellation-contract concern below. Implementation may proceed;
the caveats are the specified failure-injection, compatibility, and ownership
tests, not an additional design or approval gate. This is specification
approval only, not validation of an implementation or a performance result.

Final disposition after rereading the revised specification:

- FR2/FR5 explicitly scope filesystem durability to Local. The one-slice
  default preserves existing committed-read semantics, delegates publication
  only on `NotFound`, and propagates other errors. AC6 requires a strict legacy
  backend rather than assuming repeated single publication is accepted.
- The public surface requires whole-input conflict checks before mutation and
  exact deduplication in first-appearance order. AC6 asserts zero publication
  calls for a conflict occurring after otherwise valid handles.
- FR8 explicitly settles the entire admitted batch, including lock waits. AC5
  checks pre-admission cancellation, pending-worker ownership, second-lineage
  exclusion, and publication serialization.
- Creation-sync retry covers managed paths touched by publication and preserves
  their ancestor boundary without expanding into unrelated root initialization.

## Findings

### Blocking Issues

None remain. The following records the resolved findings and their rationale.

- **Resolved: a provided method does not establish existing-backend idempotency.** The
  original default called `publish_segment` for every handle, including a
  matching committed handle with no remaining staging file. The existing
  public contract only describes publishing a previously validated segment
  (`state/backend.rs:222`); repeat publication is demonstrated for Local
  (`tests/local_state.rs:104–105`), not guaranteed for arbitrary implementations.
  The existing managed visible-file branch accepts a successful committed load
  without publishing again (`state/transaction.rs:614–641`). A backend that
  consumes staging during publication and rejects a second publication conforms
  to that old sequence but fails the proposed default sequence. Compilation
  compatibility alone is insufficient evidence of behavioral compatibility.
  - **Suggested fix:** Preserve the single borrowed-slice surface, but have its
    default confirm already committed handles using the existing committed
    verification operation and call `publish_segment` only on `NotFound`.
    Propagate all other errors. Keep Local's override responsible for the
    stronger filesystem confirmation, including directory sync for visible
    retry files. State expressly that the default retains the backend's
    existing committed-read semantics; it does not independently establish a
    filesystem durability boundary. Do not retroactively require arbitrary
    existing `publish_segment` implementations to be idempotent across sessions.
    AC6 should use an old-style backend whose single publish requires validated
    staging and fails on a repeated call, rather than only a permissive mock.

- **Resolved: conflict rejection must precede the default's first publication.** The
  original plain sequential default could publish the first handle and discover
  a contradictory handle for the same path later. That violates the surface's
  explicit pre-mutation rejection guarantee. `StateHandle::new` validates a
  handle individually (`state/backend.rs:28–54`); it does not establish the
  consistency of a collection.
  - **Suggested fix:** Apply batch-wide same-path identity/content conflict
    validation before delegation in both the default and Local override.
    Exact duplicates can be visited once, preserving first-occurrence order.
    Assert that a contradictory batch performs zero single-publication calls
    on the old-style backend, not merely that it eventually returns an error.

### Significant Concerns

- **Resolved contract concern: cancellation settlement changes from a segment to a batch.** Existing
  `publish_staged_segments` wraps each segment separately
  (`state/transaction.rs:662–675`). `owner_settled` awaits its whole future
  once cancellation wins (`state/transaction.rs:1655–1671`). Wrapping one batch
  therefore permits the remaining admitted batch work to finish after
  cancellation. This is safe under FR8's ownership requirement, but increases
  the possible drain work. The revised spec clarifies that admission is the managed cancellation
  check before invoking the batch future; after admission, waiting for the
  backend lock and the entire batch remain owned settlement. Do not promise
  that a token observed while an already-admitted backend future waits for its
  lock prevents that future from later starting its worker. The existing trait
  has no token parameter that could enforce that stronger promise.
  - **Suggested fix:** Keep the admitted-batch semantics and extend AC5 to
    verify that another lineage open and a conflicting publication cannot
    succeed while the worker is blocked and cancellation is pending. Test a
    token already cancelled before admission separately. A gate merely proving
    that the outer future stays pending does not prove that the lease and
    serialization guard remain held.

### Minor / Style Notes

None. The review does not request unrelated style or architecture changes.

## Axis Review

- **Correctness:** Local retry durability, three-state managed classification,
  all-batch cache advancement, per-snapshot ACK boundary, and unchanged
  manifest/retention ordering are specified adequately. Caller payload and
  handle immutability are preserved. The introduction's consistent
  source-position/state recovery contract is respected.
- **Hidden assumptions:** The default-backend idempotency assumption identified
  above is removed. Local's existing matching-file early return
  (`state/local.rs:175–176`) and managed early cache insertion must both change;
  the spec correctly identifies both.
- **Missing edge cases:** Partial rename, repeated directory-sync failure,
  failed directory creation, duplicates, conflicting handles, multiple target
  directories, and successful carry reuse are covered. Test lease exclusion
  during cancellation as described above. Zero-length payloads retain their
  existing length/checksum rules; no new payload-type behavior is introduced.
- **Backwards compatibility:** No project, manifest, Python, or Studio schema
  change is proposed. The additive trait method is source-compatible; the
  revised default preserves the old backend's committed-read/publication
  sequence, with an additional sequential verification attempt.
- **Performance:** Scope and budget are appropriate. Deterministic syscall
  reduction is distinct from wall-time improvement. Default-backend extra
  verification reads must be disclosed; Local can override without paying
  that additional delegation cost. No aggregate payload buffering is needed.
- **Surface and ergonomics:** The real public operation trait is named
  correctly. One borrowed slice remains sufficient; separate new/visible
  slices would add classification and overlap rules without improving the
  optimized Local path. Preserve existing error families and path context.
- **Test plan:** The failure injections, real green-path syncs, restored
  bytes/state, ACK boundary, and checkpoint-off control provide useful
  behavioral evidence. Add the strict old-style backend and zero-mutation
  conflict assertions above. The 20-epoch claim is correctly separated from
  finite paired comparisons.
- **Risk and scope:** The durability and cancellation edges justify focused
  adversarial tests. Keep file/staging/ancestor/manifest syncs out of the
  batching reduction; retrying a failed ancestor sync must not rely only on
  directory existence (`state/local.rs:961–982`). Unix durability and the
  existing non-Unix visibility-only limit remain explicit.

## Counter-Proposals

Keep the one-slice API. Its provided implementation can preserve the old
backend's successful committed-read behavior and otherwise delegate to the
old single-publication operation, after deterministic collection validation.
Local overrides that method to supply the new batched directory confirmation.
This is smaller than introducing a public classification type or two borrowed
slices with cross-slice overlap rules. It adds serial verification work only
to the generic fallback, not the optimized Local path.

## Questions for the Author

None remain after the revision. Implement and review the resulting code against
the corrected contract and AC1–AC8; do not infer completed acceptance from this
specification review. No user-level decision is required. No builds or tests
were run for this critique.
