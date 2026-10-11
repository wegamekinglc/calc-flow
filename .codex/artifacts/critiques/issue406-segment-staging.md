# Issue 406 segment staging: contract and API review

## Target

- [Specification](../specs/issue406-segment-staging.md).
- Source baseline: `b9a12255fe80ed86d5a23da4a21ab28c94ce74e5`.
- Reviewed `StateLineageBackend`, Local staging/publication, managed snapshot
  collection, session updates, and `owner_settled` against
  [managed durability](../../../docs/runtime-envelope.md#manifest-publication-and-recovery)
  and [ownership](../../../docs/runtime-envelope.md#cancellation-and-ownership).

## Verdict

**Approve for implementation.** No blocking contract or API issues remain.
This approves the specified change and focused acceptance plan; it does not
establish implementation correctness, performance improvement, or merge readiness.

## Contract findings

- The provided borrowed-slice method preserves existing implementations and
  delegates in first-appearance order. Whole-input metadata checks precede
  mutation, exact duplicates require equal bytes, and Local additionally
  rejects distinct committed handles with the same staging target.
- Local retains every new file sync and the separate ancestor-creation,
  committed-directory, and manifest boundaries. Only repeated confirmation
  of a staging owner directory within one completed batch is coalesced.
- Retry must confirm a visible staging file's directory even when its bytes
  match. Fresh-session creation retries cannot treat directory existence as
  evidence of completed parent synchronization. These requirements address
  concrete gaps in the current early-return and creation-confirmation paths.
- Managed classification completes before staging begins. Carries retain
  their existing handles; visible committed files still require publication
  confirmation. Failure cannot add carried/verified entries, pins, or ACKs.
- One admitted batch includes the backend-lock wait and every remaining
  file. Cancellation waits for this finite unit while ownership and
  serialization remain held. The public trait receives no stronger guarantee
  for arbitrary future abandonment.
- Partial failure may leave valid staged orphans. That is compatible with the
  existing cleanup contract and does not permit publication or checkpoint
  success. Unix and non-Unix durability limits remain explicit.

## Implementation review points

- Apply FR7 to duplicate byte comparisons as well as ordinary checksum
  validation. Use bounded cooperative work or an appropriate bounded owned
  worker; an unbounded synchronous payload comparison on a Tokio executor
  would violate the approved resource contract. Do not buffer all payloads.
- Preserve one-file payload ownership while holding publication serialization
  for the complete batch, including its final directory confirmations.
- Exercise real worker and lock-wait cancellation, lineage exclusion, a later
  file/directory failure, persistent retry failure, fresh-session ancestor
  confirmation, and the strict legacy default. The AC1–AC4 tests are required
  evidence, not assumptions supplied by this approval.

## Evidence and scope

No tests, native builds, measurements, or CI queries were run for this review.
The specification separates deterministic sync-count reduction from measured
wall time and preserves the
pinned main/current comparisons and lifecycle-invalid baseline labels. The
implementation review must verify the normative documentation, compiling
rustdoc example, focused test evidence, and unchanged generated contracts.

The performance inventory and remaining budget originally reviewed here were
superseded by the user's complete eight-workload target table on 2026-10-11.
They do not authorize running the old final comparison. The API and durability
approval remains applicable; the replacement measurement protocol requires its
own review. The subsequent operator cancellation RED exposed an existing outer
dispatch boundary that can abandon admitted staging, so implementation approval
was withheld until the complete lifecycle satisfied AC4.

Final source review approved the implementation after the outer cancellation
and ASOF post-capture status REDs passed. Verification covers 85 distinct focused
test entries, the new API doctest, core library/tests all-features Clippy,
formatting, whitespace and generated-contract checks. This approves the local
source commit and release-build preparation; the complete replacement
performance protocol and its results still require separate review.
