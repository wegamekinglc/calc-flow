# Retained Join payload projection — protocol critique

## Target

- Specification: `.codex/artifacts/specs/issue406-join-state-projection.md`.
- Branch: `feature/issue406-join-state-projection`, base `3cfe6c49`.
- Scope: phase 2.2 requirements and protocol decisions, including revisions
  accepted during this critique, before product implementation. Phase 2.1 is
  not being reopened.
- Reviewer ownership: this critique only. Product, tests and specification are
  read-only; no builds, tests, measurements, CI queries or remote actions run.

## Verdict

**Proceed with caveats — the revised specification is coherent.** The initial
critique required separate version layers, complete logical/physical schema
binding, an explicit carried-charge contract and an identity-bound migration/
history transition. The owner incorporates those decisions in the specification
and accepts the state-machine guardrails below. No unresolved specification
blocker remains; implementation correctness and verification are still open.

The complete delivery must include projected dirty-row checkpoints, restore,
history reuse and rebase. Base-only snapshots are suitable for an internal
implementation/test stage; they are not an acceptable final automatically
enabled optimization because every checkpoint would regress to scanning and
rewriting all retained rows.

## Load-bearing decisions resolved in the revised specification

### 1. Bind a new envelope to both logical and physical schemas

`checkpoint_v2::metadata::Metadata` denies unknown fields, and
`checkpoint_v2::inventory::{validate_metadata, validated_base_epoch}` require
exact layout 2/codec 2 metadata. `checkpoint_v2::payload::OwnedPayload::rebind`
and the IPC validator require the full expected input schema. Adding optional
projection fields or feeding projected payloads into the old semantic decoder
would weaken an existing protocol rather than extend it safely.

There is another version layer: `PipelineBuilder::add_node` currently declares
Join's managed **operator state capability version 1**, even though Join writes
inner layout 2. `OperatorCheckpointCapability::decode_snapshot` requires exact
capability equality; `compile::graph_fingerprint` includes that capability.
Bumping it would change the fingerprint and prevent the requested managed
legacy restore before the Join migration decoder runs.

**Resolved decision:** use a separately dispatched inner projected layout, with
**layout 3** as the concrete proposal, while retaining capability 1, manifest
v3 and the logical graph fingerprint. Document those separate version domains.
Keep legacy layout 1/2 semantic decoders strict. An old reader must reject the
unknown inner layout; it must never interpret projected data as full V1/V2.

The new envelope needs an exact retained descriptor for both sides, the charge
mode, and binding to the **full logical input schemas as well as the physical
schemas**. A physical digest alone loses the old full-schema check: changing
only an omitted column's type can leave the projected schema, indices and Join
spec unchanged. Managed graph fingerprints help, but standalone operator
restore has no equivalent manifest guard. Require canonical logical-schema
digests, sorted unique retained indices and exact physical-schema digests.

Compare the descriptor to the compiled plan exactly, including empty/terminal
snapshots. Reject broader or narrower projected descriptors in this first
contract; legacy-full migration is the explicit exception, not general
projection-to-projection inference. Missing, duplicate, reordered, unknown or
out-of-range indices and malformed digests fail before state installation.
The existing ASOF `payload_projection::validate_retained_descriptor` is a
useful exact-match analogue, not a reusable Join charge proof. The revised
specification now requires the complete logical-schema binding and exact plan
comparison, including rejecting implicit broader/narrower adaptation.

The minimum complete layout-3 root has the existing V2 summary/inventory fields
plus `charge_mode = "v1_admission"` and `retained_payloads`. Its exact root keys
are:

```text
layout_version, spec, next_left_row_id, next_right_row_id,
next_output_sequence, metrics, ended, epoch, v2_inventory,
charge_mode, retained_payloads
```

The retained descriptor has exactly `columns`, `logical_schema_digests` and
`schema_digests`, each containing two side entries. Indices are sorted logical
indices; the first digests bind complete input schemas and the second bind the
selected physical schemas. Use a canonical schema-digest definition covering
field order, types, nullability and metadata; compare every entry with the
compiled plan. The reused frame/inventory codec can remain 2, while the outer
Join layout selects new semantic validation. New metadata/descriptor copies
require their own funded byte model.

### 2. Make carried charges authoritative, with a specific lower bound

`checkpoint_v2::row::validate_charge` currently recomputes every historical
row's V1 charge from its full payload and requires equality, including history
rows that are no longer live. That equality cannot survive omission of source
values. Checksum/lineage integrity does not reconstruct them and does not
cryptographically attest an arbitrary standalone snapshot's admission history.

The proposed carried-charge contract is implementable, but its semantics must
be explicit: a layout-3 row stores the historical admission charge as
**authoritative accounting metadata**, not as a charge derivable completely
from its projected IPC. Authentic new admission and fully validated legacy
migration supply that value. New-format restore validates its bounds and
consistency, rather than claiming legacy-equivalent recomputation.

There is a concrete lower bound using the frozen implementation in
`join::{state_columns_charge_with_key, logical_cell_charge}`:

```text
V1 row overhead + encoded-key length + 16
+ sum(V1 logical_cell_charge of retained columns)
+ (full logical column count - retained column count)
```

Every omitted logical cell costs at least one validity byte, including null
cells. This bound uses the unchanged logical cell table and preserved values/
types; it is not Arrow buffer capacity, IPC size, scratch credit or the sparse
chunk's physical occupancy estimate. Calculate every conversion and addition
with checked arithmetic. Require each historical charge to satisfy the bound
and the immutable per-side state-byte limit; sum live charges exactly into the
retained metrics and reapply row/byte limits. Preserve row-ID, key, event-time,
inventory and checksum checks for historical/dead rows too.

**Resolved decision:** name/version this charge mode in the new envelope and
apply it consistently in standalone restore, managed restore and rebase. Actual
decode/copy allocations remain independently funded from verified physical
geometry; a carried logical charge is not allocation admission credit.

Tests can reject under-bound, over-limit, overflowed and aggregate-inconsistent
charges. They cannot promise to reject every plausible changed charge when an
entire new-format snapshot and its hashes/metrics have been made self-consistent.
If that stronger guarantee is required, the omitted payload or independently
trusted charge evidence must remain available; the present proposal supplies
neither. Do not describe the weaker proof as full V1 recomputation.

### 3. Migration must replace owners and history, not just row views

`checkpoint_v2::candidate::prepare` currently builds a restored writer with
`WriterState::restored`, and `writer::encode::restored` carries the original
base, deltas and payload segments. Retaining that writer after narrowing live
rows preserves full old payloads in memory and in later checkpoints.
Furthermore, selecting Arrow columns from a decoded full IPC record can keep
the entire old IPC allocation alive through buffer owners. A projected schema
alone therefore proves neither steady-state memory savings nor migration.

**Resolved decision:** legacy restore first performs the existing complete
schema/key/time/charge/history validation. It then constructs an independently
funded deep copy of selected live payloads, remaps physical indices and creates
a new writer candidate with no carried full history. Install state, ownership,
layout identity and migration-required writer state together, after validation
and the final cancellation check. Never install full state and compact it later
as the migration's correctness mechanism. Temporary full decode is permitted
and must be funded; failure leaves the previously installed state/writer intact.

The first subsequent projected checkpoint must establish a **fresh narrow
base**. Until that succeeds, old full history cannot be reused or relabeled,
even on a clean checkpoint, empty state or terminal state. Later checkpoints
must resume narrow dirty-row deltas and narrow history/rebase, rather than
rewriting a full base every time.

Bind writer history, prepared work/results and cut reuse to a layout identity
containing both retained descriptors and charge mode. Existing
`history_trusted`, `migration_required` and tracker ownership are useful gates,
but none currently establishes that identity. A descriptor/mode mismatch or
OFF mutation must force a funded fresh base, never a same-count/same-charge
history shortcut. Reset, cancellation/refusal, stale prepared results and
failed restore must not clear those gates prematurely.

## Significant concerns and required invariants

- **Logical versus physical indices:** derive retained sets and inverse maps
  once from complete input declarations. Split a logical output index using
  the logical left width, then map within that side and apply the physical
  left width. Keep original admission indices separate from physical probe/
  recovery indices. Both incoming prepared rows and resident rows must already
  have the agreed physical schema when gathering; keys/time shifted by omitted
  preceding fields are the critical regression. ASOF eligibility stays unchanged.
- **Actual owners:** selected-only state applies to normal retention, declined
  owned-copy fallback, restored rows, sparse repair, pending upserts, checkpoint
  work and writer payload history. A `RowPayload::Rowed` full parent or a
  projected view into full IPC does not satisfy D5. Copy selected recursive
  buffers and compact dictionaries independently, with credit through drain.
- **Three accounting domains:** preserve `StoredRow::charge`, state/status
  retained bytes and admission limits from original full inputs. Recompute
  selected-column occupancy for sparse/GC bookkeeping. Independently fund
  actual schemas, indices, copies, IPC and temporary migration owners. Physical
  savings must not be inferred from the deliberately unchanged status counter.
- **Admission/error order:** compute authentic charges from full source rows
  at the same point in the existing classification/admission flow. Do not start
  charging rejected null/late rows or discarded fields earlier and thereby
  expose errors or allocation failures that the original ordering avoided.
- **Managed and direct paths:** extend explicit format dispatch in
  `checkpoint_v2::managed` and `StreamJoinOperator::restore`, construction,
  preload funding, writer metadata/empty-base capture and rebase validation.
  Existing managed Join preload already measures generic metadata/segment
  ownership, but new descriptor copies/workspace still need admission. A
  layout-1 test producer must reject narrowed state instead of mislabeling it.
- **Checkpoint preparation:** define behavior while the migration base is
  unprepared or refused. Managed prepare/capture must publish no new checkpoint
  until a valid narrow base exists; the prior durable checkpoint remains
  untouched. Successful restore with migration pending and a later checkpoint
  failure is distinct from a failed restore that must preserve old installed
  state. Preserve epoch, source progress and output-sequence contracts.

## Minimal coherent counter-proposal

Retain one immutable per-side layout plan containing sorted logical indices,
physical schemas, remapped identity/time indices and output composition. Keep
logical ports, exported declarations and fingerprint construction unchanged.

Give projected layout 3 its own exact envelope/semantic validation and writer
identity. Reuse V2's immutable frame/IPC geometry where the byte contract is
unchanged, but do not pass narrowed payloads through the V2 full-charge semantic
decoder or relax its format checks. Layout 3 validates selected key/time values
and the carried-charge contract; layouts 1/2 retain their old complete proof.

Use these explicit transitions: new projected state starts with a narrow base;
matching layout-3 restore resumes narrow validated history; legacy restore
validates full data then installs independent narrow state with migration
pending; one fresh narrow base completes that transition; subsequent deltas and
rebase stay narrow and identity-bound. OFF history invalidation requires another
fresh narrow base before reuse. Full-retention execution remains on the existing
legacy format and rejects a descriptor that cannot supply its complete state.

This is one runtime compatibility unit. Split its work into protocol/spec
review and bounded implementation reviews, not into an automatically enabled
state-pruning change followed by a later recovery fix. Base-only projected work
may be an internal test stage, but incremental/history/rebase and their latency
scope remain required before this increment is called complete. No performance
acceptance follows from this design choice.

## Focused acceptance additions

The draft's direct retained-column RED is appropriate and does not authorize
protocol implementation before these decisions are resolved. Add bounded
controls for the following contracts without a Cartesian matrix:

1. Both incoming sides with unused fields before keys/time; constant-only
   consumers still retain identity/time and preserve output row count. Prove
   selected copy work, caller immutability and no retained full-parent/IPC owner.
2. Legacy V1 and V2 with dirty/dead history: validate full charges, migrate,
   drop full history, write a narrow base, then write/restore a narrow dirty
   delta and rebase. Compare authentic logical charges/status/limits and output
   continuation with full execution throughout.
3. Unknown layout/charge mode; altered omitted logical field; noncanonical or
   mismatched descriptors; under-bound/overflowed/over-budget charges; aggregate
   mismatch; stale/dangling payload locator and malformed dead history. Reject
   before installation and leave old state/funding intact.
4. OFF same-charge replacement, reset, initially empty and terminal state;
   clean checkpoint immediately after migration; stale prepared work; restore/
   copy/base admission refusal, cancellation and abandonment. All leases stay
   owned until actual drain; checkpoint bytes never mix full and narrow layouts.

The schema/charge preservation controls and final local specialist review are
separate from required CI/coverage, phase-wide allocation/RSS and all-eight
performance acceptance. Retain previous STOPPED evidence without retry.

## Other critique axes

- **Correctness and public surface:** compatible with immutable Arrow-backed
  `Batch`, data-only projects and compatible recovery in
  `docs/introduction.md`, once the three protocol findings are locked. No new
  public API, executable configuration or provider path is needed.
- **Hidden assumptions:** exact full schemas, per-side layout identity and
  sufficient temporary migration funding must be explicit. A checksum is
  integrity evidence, not omitted-value reconstruction.
- **Backward compatibility:** preserve legacy reader semantics and fixtures;
  schema descriptors govern projected compatibility. Do not change capability
  version or logical fingerprint casually to hide an inner-format migration.
- **Performance and scope:** retain dirty-row behavior and bounded async work;
  no whole-state rewrite at every projected checkpoint. Physical savings and
  handler latency need their own later evidence.
- **Documentation:** update `docs/runtime-envelope.md` and the streaming
  guidance with version separation, authoritative carried accounting, legacy
  migration and exact physical-plan compatibility. No manifest/project-v3
  format change is implied.

## Handoff

The revised specification incorporates the projected envelope, complete schema
binding, explicit accounting mode and required identity-bound incremental
migration. Proceed with implementation against those decisions and this state
machine; final specialist implementation review remains required. This critique
approves neither an implementation nor issue-wide acceptance.
