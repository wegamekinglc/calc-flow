# Issue 406: retained Join payload projection

Status: requirements/protocol accepted for implementation with the caveats in
`.codex/artifacts/critiques/issue406-join-state-projection.md`; no remaining
specification blocker. Implementation and focused local verification are
complete; final specialist implementation review is pending.
Base: reviewed local `3cfe6c49f8f962d9fa7ae7476b9f2f3e2f2c2a56`.
Phase 2.1 output projection is implemented and locally reviewed. Phase 2.2 is
awaiting final local review; all original eight targets and other acceleration phases remain
active. No performance acquisition or retry belongs to this draft.

## Required behavior

For a Join whose consumers all qualify for physical output projection, retain
only the selected payload dependencies plus every column the operator itself
needs for identity, matching, event time, expiry and recovery. Derive sorted
per-side logical indices once at compilation; keep complete logical input ports
and validate caller batches against those declarations. Unconsumed/full-input
producers and selected-UDF graphs keep full retained payloads.

Admission must classify keys/time and compute the existing V1 logical row charge
from the original full input before discarding payload fields. Preserve authentic
execution's state/status charges, limits, row IDs, matching order, null/late/error
priority, cancellation and preflight-before-emission. Necessary identity/time
fields can remain retained even if no expression consumer reads them.

Actual state owners, sparse compaction and checkpoint IPC must omit unused
payload arrays. Merely projecting emitted records, wrapping full parents, hiding
columns behind a view or dropping fields only after checkpoint decoding does not
satisfy steady-state retention. Temporary admission may share selected caller
arrays read-only until owned retention copies them; state retains independent,
funded selected buffers with existing dictionary compaction guarantees.

Compose output mappings with retained physical indices for both incoming sides.
Incoming prepared rows and resident opposite rows must use the same per-side
physical schemas before gathering; deriving the side offset from a full incoming
row against a narrowed opposite side is incorrect. Separate logical admission
indices from physical retained-key indices, including schemas with unused fields
before keys/time. Preserve selected flat/nested budgets and zero-column output
row counts from phase 2.1.

## Recovery and format proposal

The managed checkpoint manifest stays v3 and project-v3 remains unchanged.
Join's logical checkpoint capability stays `state_version = 1`; changing that
capability also changes the graph fingerprint and would defeat legacy managed
recovery. This capability is separate from the inner operator payload layout.
Logical configuration/fingerprint and exported declarations remain data-only and
unchanged. Retained physical schema belongs in operator checkpoint metadata.

Current Join layout 2 readers/writers require exact metadata fields, version 2
and full payload schemas. Adding optional projection fields to that format would
silently change its contract. Proposed direction: a strict inner
`layout_version = 3` projected envelope carrying exact retained logical indices
and both logical/physical schema digests, while reusing appropriate immutable V2
payload/frame machinery.
Do not weaken the existing layout 1/2 decoders or historical fixtures.

Before implementation, the critique must settle the projected envelope shape
and its interaction with managed preflight, writer history, payload inventory,
restore admission and rebase. Only actually narrowed retained schemas need the
new representation; ordinary full-retention execution keeps its current format.

The exact projected root contains the existing V2 summary/inventory fields plus
`charge_mode: "v1_admission"` and `retained_payloads`; its `layout_version` is 3.
The retained descriptor has exactly `columns`, `logical_schema_digests` and
`schema_digests`, each containing left/right entries. Indices are sorted/unique,
and all values match the compiled plan exactly; reject unknown/missing fields.
Full logical digests bind names/types/metadata of omitted columns as well, since
standalone restore has no managed graph fingerprint to supply that proof.
Only explicit legacy migration accepts an old full representation; projected
descriptors do not implicitly adapt broader or narrower projected checkpoints.

A fresh projected base is useful as an internal protocol-validation stage,
but base-only rewriting is not the final phase 2.2 deliverable. Keep projected
incremental checkpoints and rebase: history identity includes the retained
descriptor and charge mode, and legacy migration forces a new narrow base before
subsequent narrow deltas. Cover history reset/OFF, empty/terminal state and failed
restore explicitly. Do not turn dirty-row capture into a full-retained-state
rewrite whenever the compiler enables projection, or defer this existing
complexity contract as an optional follow-up. Compiler enablement waits for the
coherent checkpoint and runtime integration.

Required compatibility:

- A matching projected checkpoint cold-restores into the same physical plan.
- A checkpoint omitting any required retained field is rejected before state
  installation, even if logical input schemas/fingerprint match. Unknown,
  duplicate, reordered/out-of-range descriptor indices and digest mismatches
  fail explicitly; do not infer missing columns or fabricate null payloads.
- Authentic legacy full layout 1/2 checkpoints remain readable by a newly
  projected plan. Validate their original schema/charges before discarding
  unused fields. The next projected checkpoint must not preserve full old
  payload segments through reused history; establish a funded projected base.
- Full-retention plans keep their legacy recovery behavior and reject projected
  checkpoints that cannot supply their full state. Test-only V1 producers must
  not emit narrowed payloads while labeling them legacy layout 1.
- Cancelled, refused or invalid restore/migration leaves the previous installed
  state and owners unchanged; all temporary leases drain/refund correctly.

There is an accounting proof decision to resolve explicitly: after omitting
payload columns, a decoder cannot recompute the original full-row V1 charge from
remaining arrays alone. Proposed new-format semantics carry the authentic
admission charge per row, validate its representability, a V1 logical-cell lower
bound including omitted-field minima, aggregate metrics and limits. Such carried
charges are authoritative historical values in the new format. Checksums verify
integrity against a manifest; they are not signatures proving original admission
history. This is not equivalent to recomputing a full V1
charge from a full legacy record. Critique must either justify and document this
new-format contract or propose another coherent solution; do not claim both
omitted source payloads and complete recomputation from those absent values.

The proposed lower bound uses existing V1 accounting, not Arrow allocation or
GC bytes: state-row overhead plus encoded-key length, the existing 16-byte term,
retained logical-cell charges and at least one byte per omitted logical column.
Compute omitted width from trusted compiled schemas/indices, not a supplied
counter. Keep exact full-record charge validation on legacy layout 1/2 paths.

## Focused verification

Observed integration points for the implementation:

- `side_plan`, SQL probe/state-key scratch and native lookups currently share
  logical `CompiledJoin` indices and declared input schemas. Physical retained
  indices need an explicit separate route, rather than changing admission's
  full-input indices globally.
- `owned_copy::prepare_payload_copy` initializes funding from `input_schema`
  and full `payload_schema_bytes`. Both copy columns and funded schema metadata
  must become selected; `ChunkRow.bytes` tracks actual physical GC bytes and
  must not be confused with `StoredRow.charge`'s historical logical charge.
- `finish_prepared` calls `retained_rows` after matching. Computing its charge
  from an already projected RowPayload would silently reduce V1 charges. The
  original input and `admitted_source_rows` remain available during preparation;
  preserve a coherent original-row charging route without reinterning keys.
- The V2 metadata writer's byte model enumerates exact JSON fields. Any new
  projected envelope/descriptor needs its own funded metadata model, as well as
  recognition in synchronous and managed restore/preflight.

Start with recorded RED proving a projected Join still retains an unused large
payload column. Reuse existing registrations, then test both incoming sides,
identity fields shifted by preceding unused columns, aliases/arithmetic/filter
consumers, constant outputs and multiple consumers. Compare full-input matching,
status/limit/error behavior and untouched caller buffers against the baseline.

Prove selected-copy work and stored schemas omit unused fields; cover sliced and
multiple parents, nullable/variable/nested/dictionary fields, owned worker refusal,
cancellation/abandonment and sparse compaction. Avoid a Cartesian case matrix.

Cover projected cold continuation, legacy V1/V2 migration, projected-base rewrite,
missing/broader/narrower/forged descriptors, charge bounds and malformed inventory,
budget refusal and actual owner cleanup. Adjust the phase 2.1 recovery controls
only where this explicitly versioned state contract changes their assumptions;
preserve their original output and charge parity controls.

Use focused core tests, necessary scoped Clippy/format/CCN checks, generated
schema no-drift and whitespace checks. Final specialist review is required.
Normative streaming/state documentation must explain the resolved projection and
accounting/recovery semantics. No performance, combined 90% coverage, remote CI or
complete issue verdict follows from a source implementation alone.
