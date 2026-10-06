# Stream Join and ASOF acceleration - Critic Critique

## Target

- Spec: [stream-join-asof-acceleration.md](../specs/stream-join-asof-acceleration.md)
- Proposal: [Issue #363](https://github.com/wegamekinglc/calc-flow/issues/363),
  especially D1, J2b, and A4.
- Review scope: the proposed contracts and high-risk design gates; no build,
  tests, or review of the concurrent implementation was performed.

## Verdict

**Revise** the ASOF version wording and make the two compatibility gates
explicit. Low-risk progress/safety work, J1, and A1/A2 can proceed. The narrow
Join v1 migration direction is viable without changing the global capability
decoder or semantic fingerprint. A4 must remain gated by exact allocation,
inventory, and wire parity rather than output parity alone.

## Findings

### Blocking Issues

- **FR3 names the wrong ASOF layout/accounting versions.** The spec says
  "Preserve state/layout/accounting version 3." Current ASOF metadata has
  semantic `state_version = 3`, but current row-log `layout_version = 10` and
  `accounting_version = 10`. Restore accepts only that pair, and the wire magic
  is `CFASRW10`; historical layout 3/4 snapshots are explicitly rejected.
  Evidence: [checkpoint.rs](../../../crates/calc-flow/src/operator/asof/checkpoint.rs)
  (`capture_metadata`, `validate_metadata`),
  [log.rs](../../../crates/calc-flow/src/operator/asof/checkpoint/index_v3/log.rs)
  (`MAGIC`), and
  [current_format_tests.rs](../../../crates/calc-flow/src/operator/asof/checkpoint/current_format_tests.rs)
  (`current_restore_rejects_layout_three_and_four_before_workspace`). This
  conflation would turn an acceleration into an incompatible format change.
  - **Suggested fix:** Freeze semantic state version 3 and the actual current
    row-log layout/accounting 10 separately, including framing and canonical
    bytes. Do not introduce layout 3 or 11 as part of this optimization.

### Significant Concerns

- **D1 should retain the existing semantic capability, not introduce a
  compatibility bypass.** Join declares capability `state_version = 1` in
  [pipeline/mod.rs](../../../crates/calc-flow/src/pipeline/mod.rs). That value
  is inserted into managed snapshots, checked for exact equality in
  [pipeline/stream.rs](../../../crates/calc-flow/src/pipeline/stream.rs), and
  included in the fingerprint by
  [pipeline/compile.rs](../../../crates/calc-flow/src/pipeline/compile.rs).
  Join's private `layout_version` in
  [join.rs](../../../crates/calc-flow/src/operator/join.rs) is a distinct
  field. ASOF already demonstrates semantic version 3 with a separate physical
  layout 10. A representation-only Join layout v2 can therefore keep capability
  1 and the exact unchanged-graph fingerprint, with v1/v2 decoding confined to
  Join. The existing mismatch/fingerprint tests must keep passing unchanged.
  - **Suggested fix:** State this narrow mechanism in FR12 and prohibit global
    fingerprint relaxation, version normalization, and family-wide fallback.
    Require a managed manifest captured by the prior release, continuation,
    v2 recheckpoint, and another restart. Also reject altered graph, bindings,
    schema, UDF versions, lineage, and reserved capability metadata. Old
    binaries need not read new v2 snapshots; document that forward direction.

- **A4 must preserve actual owner topology and capacities.** Equality of key
  bytes does not imply equality of charged state. `Encoding::Batch` retains
  an `EncodedBatch` whose values and offset capacities plus metadata are
  charged; `EncodingOwners` tracks owner lifetime and owner wire length.
  Evidence: [state.rs](../../../crates/calc-flow/src/operator/asof/state.rs)
  (`EncodedBatch::allocation_bytes`) and
  [ownership.rs](../../../crates/calc-flow/src/operator/asof/state/ownership.rs).
  Replacing one encoded batch with compact per-distinct-key owners can change
  state bytes, resource failure thresholds, and checkpoint owner records,
  despite identical output. Likewise, `ChunkData::prepare` deliberately copies
  timestamps because external Arrow buffers can hide larger backing owners;
  see [left.rs](../../../crates/calc-flow/src/operator/asof/state/left.rs).
  The current checkpoint log records column and index capacities, not only
  live lengths; see
  [model.rs](../../../crates/calc-flow/src/operator/asof/checkpoint/index_v3/log/model.rs).
  - **Suggested fix:** Add explicit same-input differential admission tests
    for actual capacity/owner inventory, journal, and checkpoint bytes, then
    every finalized/evicted prefix, including partially referenced shared
    owners. Include tight limits immediately below/at the legacy admission
    charge, external/sliced Arrow backing, and cancellation/reset with a live
    worker. If direct Arrow sharing or a distinct-key representation cannot
    prove parity, keep the existing funded owned-copy/encoding path for that
    shape. Do not simulate capacity charges while silently changing the
    owner wire inventory.

### Minor / Style Notes

- [runtime-envelope.md](../../../docs/runtime-envelope.md), "Stream plan compilation",
  describes the fingerprint as freezing a state-layout version. When
  documenting J2b, distinguish the capability's semantic compatibility version
  from private physical layouts. Its existing exact compatibility claim must
  remain intact.

## Axis audit

- **Correctness:** Covered by FR1–FR4, subject to the version and owner fixes.
  Input immutability and sink-accepted prefix commit match
  [introduction.md](../../../docs/introduction.md), "The basic vocabulary",
  and [asof-join-guide.md](../../../docs/asof-join-guide.md), "Bounded state
  and workspace".
- **Hidden assumptions:** The order proof and committed-state boundary must
  also work after restore and prior out-of-order admission; canonical key order
  cannot be replaced with dictionary-ID order. FR16 already requires fallback.
- **Missing edge cases:** Tight capacity thresholds, hidden backing owners,
  partial owner release, and second restart after migration need the explicit
  gate above. Existing shared-owner and sliced-backing tests provide analogues.
- **Backwards compatibility:** Join read-v1/write-v2 is viable with capability
  1 unchanged; ASOF physical format 10 must remain unchanged.
- **Performance:** Targets are planning goals, not acceptance assertions of
  measured gains. Two-round paired measurements, retained/interval coverage,
  checkpoint activity, small batches, allocations, and worker RSS are adequate.
- **Surface and ergonomics:** Additive Join status with full-range Studio
  timestamp strings is consistent with the existing ASOF status convention.
- **Test plan:** Order-sensitive property tests and pre-change durable fixtures
  are appropriate. Stub output or layout-only unit tests cannot establish
  managed recovery or resource parity; require the differential gates above.
- **Risk and scope:** Preserve the staged PR dependencies. J2b and A4 have
  explicit gates; they should not delay independent low-risk work.

## Counter-Proposals

Keep the existing exact semantic-capability and graph-identity gate. Confine
Join's representation migration to its private reader/writer. For A4, restrict
eligibility to shapes whose legacy owner/capacity inventory can be reproduced;
fall back before admission for all other shapes. These constraints preserve
the issue's scope without broad runtime migration infrastructure.

## Questions for the Author

No additional user decision is required for D1–D6. Amend the version wording
and the two proof gates above before J2b/A4, then route the actual implementation
and migration/resource evidence to the final specialist reviewer.
