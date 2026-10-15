# Issue 406: retained Join projection evidence

Base: locally reviewed/committed phase 2.1 `3cfe6c49`. The phase 2.2 candidate
implements independent selected-state ownership, strict layout-3 recovery,
authentic V1/V2 migration, identity-bound incremental history and OFF rebuild.
Focused runtime/recovery, nested-dictionary, actual native sparse and generic
CopyWork lifecycle controls pass. Core lib/test Clippy passes. The doc writer
has reconciled both normative guides and the qualifying changelog. Final local
specialist implementation review approves commit e884adc362ac8f8d03c1364d0dfdb4e205407696;
see the [review report](../reviews/issue406-join-state-projection.md).

Current Rust source is frozen in `complexity-phase22-review.json`: 50 Rust
files, 293 changed/new functions, maximum CCN 8, no changed/new parser omissions.
Sorted file/sha256/LF source digest:
`706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302`.
The five parser omissions are exact unchanged HEAD inventory.rs excerpts.

The latest selected-state registration passes in
`selected-copy-worker-corrected.log` (1m21s compilation, 0.18s execution),
and `clippy-selected-copy-worker-corrected.log` passes in 56.81s. The recorded
RED/GREEN and bounded intermediate evidence below show which source/contract
each earlier observation covers; older running/pending notes are superseded by
their later results. No full matrix or current-source release build ran. The
Python inventory tests used a compatible baseline native import, not the new
native implementation. Required remote CI, combined 90% coverage and all eight
original issue targets remain unproven. Other acceleration phases remain active;
no performance acquisition/retry or remote mutation belongs to this increment.

## Recorded RED

From this worktree, with the root `target/issue406-join-hotpaths-build/`, two
Cargo jobs and serial execution:

```bash
cargo test -p calc-flow --lib \
  test_charge_fast_path_falls_back_when_the_bound_exceeds_the_budget \
  -- --test-threads=1 --nocapture
```

Root `target/issue406-join-state-projection-20261014/retained-projection-red.log`
records compilation in 3m54s, then one expected failure in 0.01s. Before the
failure, the new helper compares full/projected authentic state charges and
caller input equality, drops both operators, awaits job cleanup and asserts the
pool returns to zero. The final state-schema assertion fails:

```text
actual:   [key, time, value, unused]
required: [key, time, value]
```

The unused field is a large UTF-8 payload. The helper is invoked through an
existing registration, preserving the 2,033 default libtests. The owned command
terminated with expected exit 101 and its handle is closed. This is RED evidence,
not a GREEN or a retention/performance acceptance result.

## Implementation handoff

Specialist critique concludes **Proceed with caveats**, with no remaining
specification blocker. The accepted specification
separates logical checkpoint capability from the operator's inner projected
layout, requires safe legacy migration and projected incremental/rebase history,
and explicitly identifies the historical-charge proof change caused by omitting
source values. A base-only projected writer is an internal implementation stage,
not a replacement for the final incremental contract.

The strict inner layout-3 envelope binds full logical and selected physical
schemas, retains logical capability version 1, and names authoritative carried
charges as `v1_admission`. Checksums do not sign historical admission values.
History/prepared work binds the descriptor and charge mode. Fully validated
legacy migration constructs independent funded narrow state with no full history,
then one fresh narrow base precedes subsequent narrow dirty deltas/rebase.

Selected admission/state ownership, key/output remaps, funding and new-format
cold recovery are implemented as WIP. Legacy compatibility, writer identity,
malformed-checkpoint and lifecycle controls, scoped checks, normative
documentation and final specialist review are still required.

## Runtime WIP evidence

The implementation keeps full logical admission/matching and V1 row charging,
then independently copies selected columns and remaps the output/state-key paths.
Native matching continues to consume original encoded keys. Flat physical-validity
eligibility applies to the selected columns; other payloads use an observed,
funded owned worker with cancellation and retirement through settlement.

All logs are under the root evidence directory above. Initial runtime attempts
`runtime-first-green.log` and `runtime-corrected-green.log` are compile failures
in the new cleanup-control call (fallibility, then explicit output type), not
passing tests. `runtime-selected-cases.log` records an invalid Float64-key fixture;
that type is not a supported Join key. It was corrected to Int8, exercising SQL
fallback with shifted physical identity columns.

`runtime-fixture-corrected.log`: one existing registration PASS in 0.02s after
52.33s compilation. Its helper covers the original large unread UTF-8 field,
nullable retained Int64 with an Int8 SQL key/left-first input, and retained List
with an Int64 native key/right-first input. Sliced/multiple parents, output/status
parity, caller immutability, independent recursive buffers and zero-credit drain
are checked. Default registration count remains 2,033.

The bounded specialist runtime review found three actual issues: unsupported/
nullable flat-copy qualification, unbounded parent/filter scans on Tokio, and
resource rejection classified as checkpoint corruption. Those issues were
corrected. It also established that Arrow concat for view arrays shares source
variadic backing; wrapping that backing in a new funded owner does not copy it.

`view-ownership-red.log`: expected FAIL in 0.02s after 52.29s compilation, from
retained view backing overlapping caller storage. The test releases operators
and awaits job drain before that assertion. Direct long views and Struct<view>
are now recursively rebuilt from selected values into independent funded
buffers. `view-ownership-green.log` PASS in 0.03s after 1m15s compilation;
`runtime-refactor-green.log` PASS in 0.05s after 1m25s compilation, after
complexity extraction and explicit fallible view append/owned-block length
validation. The latter checks u32::MAX boundary rejection without allocating
4 GiB. These are source/runtime controls, not performance evidence.

The specialist source recheck resolves the three original runtime findings;
the view extreme-length follow-up was subsequently corrected. Actual owned-copy
refusal/cancellation/abandonment tests, final scoped lint/complexity provenance,
protocol compatibility and final specialist approval remain pending.

## Checkpoint WIP evidence

`projected-checkpoint-red.log`: expected integration failure in 0.04s after
53.79s compilation. The existing full-schema writer indexes a missing fourth
column from the narrowed three-column state, producing a supervised worker
TaskPanicked error. The isolated test awaits owned cleanup before reporting it.
This establishes the missing checkpoint integration; it is not an acceptable
finished state.

The writer WIP now uses retained schemas and a layout-3 envelope with the exact
descriptor/charge mode. The first attempt (`projected-checkpoint-first-green.log`) is a compile failure
because Arrow Schema is not Serde Serialize; the corrected encoding uses the
Arrow IPC schema representation. `projected-checkpoint-corrected-green.log`
then PASSed in 0.03s after 2m03s compilation. Specialist source review found its
IPC schema digest was not injective over complete Arrow types: Timestamp None
and Some("") normalize identically, as can a directly nested Dictionary layer.
That intermediate IPC digest has been replaced; it is not the current contract.

`logical-schema-digest-red.log` records the Timestamp collision (expected FAIL,
0.01s after 53.06s compilation). Current digests use the explicit
`calc-flow/join/retained-schema/v1` equality encoding: fixed-width little-endian
counts/parameters, length-framed strings, sorted field/schema metadata, recursive
field/type identities and exhaustive pinned Arrow DataType variants. Scalar
types without fields use their complete scalar Debug form, including temporal
Option values. IPC-only dictionary ID/ordering properties are intentionally
omitted, matching Arrow Field equality. A fixed golden digest protects the v1
wire identity. Both omitted Timestamp and nested Dictionary controls PASS in
`logical-schema-digest-green.log` (52.84s compile, 0.04s execution). The bounded
specialist source recheck resolves this finding, without a phase-wide approval.

`projected-cold-restore-red.log` records strict legacy parsing rejecting the new
charge_mode field (53.97s compile, 0.05s expected FAIL). The separate projected
metadata reader preserves exact root/descriptor fields and legacy readers;
physical key/time indices and carried-charge lower bounds are selected only for
layout 3. `projected-cold-restore-first-green.log` PASSed (2m37s compile, 0.03s).

`clippy-runtime-layout3.log` is a scoped Clippy failure with five style findings
(iteration, single-pattern match, lossy view-length cast, and needless Vec
ownership). These were corrected without allowances. The affected registration
plus fixed digest PASSed in `runtime-layout3-lint-corrections-green.log`
(1m02s compile, 0.03s). The corrected scoped Clippy subsequently passed; the latest result is recorded below.

`projected-managed-delta-rebase.log` PASSed (1m09s compile, 0.05s). The focused
helper now captures an initial narrow checkpoint, four dirty deltas and a
rebase; selected cuts restore through both direct and owned managed paths.
Status/charges remain equal and destination/source pools drain to zero. This is
new-format continuation evidence, not legacy migration or performance proof.

## Legacy V2 migration WIP

`legacy-v2-projection-red.log` records authentic legacy V2 restore retaining the
unused column (1m18s compile, 0.04s expected FAIL after charge parity and zero
credit drain). The new migration runs after full history/schema/charge
validation, copies selected live payloads independently, preserves logical
charges/IDs/keys, and installs an empty-history migration-required writer.
`legacy-v2-projection-first-green.log` PASSed (2m28s compile, 0.05s).

Managed migration shares a dedicated retirement guard, without old payload
owners, between new resident copies. Workspace pays the initial shared Arc;
each resident seed conservatively covers it after workspace retirement. Source
review found the last Arc control must be freed before the final resident
refund. Funding::drop now extracts the last Arc with Arc::into_inner and returns
its guard to the original Direct drop order. Direct construction adds no Arc
allocation. The stronger managed/escaping-buffer/fresh-base control PASSed in
`legacy-v2-managed-owner-and-base.log` (2m06s compile, 0.06s execution).
A sole escaping buffer keeps resident credit and job retirement alive; releasing
it permits the job drain and all three pools to reach zero.

Complete cold/managed layout-3 malformed-input validation, identity-bound
incremental history, legacy V1 and historical/dead-row migration, mandatory
one-time fresh narrow base and its refusal/cancellation controls,
empty/terminal/reset controls and normative docs
are still required. No phase 2.2 commit, remote push, CI result or final review is
claimed by this partial runtime/writer stage.

## Selected-range accounting evidence

`contiguous-selected-copy-budget-red.log` records expected rejection of 4,096
selected rows under a 1 MiB pool (2m50s compile, 0.02s execution). Per-row slice
controls exhausted the pool even though the selected payload fits. The copier
now creates temporary slices per contiguous source run, retaining input order
and duplicates. Resident backing remains funded by selected rows, bytes and
recursive geometry. Cancellation checks cover both scans.

`contiguous-selected-copy-budget-green.log` PASSed (1m04s compile, 0.06s).
`selected-ranges-and-v2-migration-final.log` PASSed (2m18s compile, 0.05s), adding
a sparse, reordered, duplicated selection `[2, 0, 1, 1, 7, 8]` against independent
Arrow take, with distinct source keys and zero-credit drain. This registration
also retains the runtime/new-format/V2 migration controls described above.

`clippy-legacy-v2.log` and `clippy-selected-ranges.log` record style failures,
corrected without allowances (used underscore field, needless value passing and
large test future). `clippy-selected-ranges-corrected.log` PASSed in 1m59s.
Formatting, generated-schema no-drift and whitespace checks also passed before
the latest test fixture change. The older complexity manifest predates these
changes and is not final provenance. No performance inference follows.

Legacy V1 migration now has an authentic full-payload test fixture under the
same registration; its RED observation is in progress. Product V1 integration,
writer identity, adversarial/lifecycle controls and final documentation/review
remain pending.

## Legacy V1 migration WIP

`legacy-v1-projection-red.log`: expected FAIL after 1m57s compilation/0.05s
execution, exposing unfunded escaping legacy buffers. The cleanup-corrected
fixture in `legacy-v1-projection-cleanup-red.log` (57.99s/0.08s) drains owners
before the expected full-versus-selected schema failure.

The implementation now routes projected layout 1 through the funded owned
decoder. Borrowed framing validates every historical upsert against the complete
input schema, exact V1 key/time/charge, identity counters and segment inventory
before independently copying selected live rows. It installs no full historical
payload segments; the migration writer requires one new narrow base.

`legacy-v1-projection-first-green.log` is a compile failure (inventory borrow
versus shared-retirement construction), corrected by reordering and bounded
helper extraction. `legacy-v1-projection-corrected-green.log` compiled in 2m06s
then failed in 0.07s because the test-only V1 producer's segments have no paid
owner. Production managed LocalStateBackend preloading supplies owners for both
formats. The fixture now attaches real capacity/control/SHA-string credit only
to ownerless segments, keeping the raw direct fixture and existing V2 owners.

Specialist review found two actual V1 issues: a local shared Arc surviving the
consume-self assemble failure boundary, and rejection of authentic empty V1
history after on_end. The Arc is explicitly dropped while decoder workspace is
still alive, before assemble. Empty inventory now requires all four retained
gauges to be zero; historical counters, ended and epoch remain preserved.

`legacy-v1-empty-history-red.log` reached the authentic empty fixture after
nonempty direct/managed migration and fresh-base checks, but its cleanup check
ran before dropping the job. `legacy-v1-empty-history-cleanup-red.log` corrects
that fixture order and records the expected empty-inventory rejection after
zero-credit drain (1m09s/0.06s). `legacy-v1-empty-history-green.log` PASSed (57.82s compilation, 0.05s
execution), including nonempty direct/managed migration, fresh narrow bases,
escaping-buffer retirement and authentic empty history. No phase-wide approval,
performance or CI acceptance is implied.

## Pending writer consistency correction

Read-only specialist review identified that invalidate_history only sets trust
false: cached fresh preparation can still match after OFF mutations. Existing
rebase_trust_capture_tests explicitly expect stale cached captures and do not
compare their cold restore to the changed live state. A discriminating RED must
cover fresh/base -> OFF retention or eviction -> ON preparation/capture and cold
parity, including unchanged gauges with changed payloads.

Use a cached private fixed-size HistoryIdentity, hashing framed full logical
and retained physical schema digests, selected column indices and charge mode
with an independent domain and full/projected tag. Bind WriterState history,
Cut/prepared/submitted work and rebase inputs to it; compare at reuse, merge,
installation and capture. Empty history is unbound; strict matching restore
binds validated identity; legacy migration remains unbound until its new base.
OFF invalidation must checked-advance revision and force a live fresh base before
capture; normal trusted dirty state retains incremental delta/rebase behavior.
Commit history identity/trust/pending clearing only after cut/cancellation
validation. Keep fixed fields under existing size_of funding, wire contracts,
capability and fingerprint unchanged. This is a reviewed design, not implemented
or verified behavior. Final approval remains pending.

`legacy-v1-producer-label-red.log` records the expected failure when a narrowed
plan's test-only producer emits layout 1 (55.98s compile, 0.01s execution). Both
test-only preparation and capture now reject a retained projected layout before
mutating state or encoding. Authentic full V1 producers remain available.

`legacy-v1-label-and-tail-green.log` is a compile failure from an unnecessarily
qualified test accounting helper, corrected without lint allowances. The
corrected tail/dead-history control PASSed in
`legacy-v1-tail-dead-history-green.log` (1m19s compile, 0.06s execution). It injects cancellation at the
last three decoder checks and the pre-install check with real retirement guards,
asserts prior state/epoch and resident credit remain unchanged, and drains the
separate job. The authentic dead-history fixture carries prior full upserts plus
on_end tombstones; a changed historical charge must fail even though live state
is empty. The empty-history fixture also rejects each missing nonzero retained
gauge without modifying installed state. These are focused protocol/lifecycle
results, not final feature or performance acceptance.

`clippy-legacy-v1.log` found two style errors (unused self and negative if/else),
corrected without allowances. `clippy-legacy-v1-corrected.log` PASSed in 39.81s.
The intermediate complexity manifest `complexity-legacy-v1.json` caught the
touched V1 test producer at CCN 10 (its base was 9); pending-delta capture has
been extracted to a CCN-5 helper, reducing checkpoint_v1 to 7. The expanded
prepare-and-capture guard/refactor test is running. This manifest is therefore
intermediate, not final source provenance.

`legacy-v1-guard-refactor-green.log` PASSed (1m14s compilation, 0.06s),
including explicit refusal before test-only V1 preparation and capture.
`legacy-full-empty-compatibility.log` PASSed the existing full-plan authentic
empty-V1 compatibility registration (0.15s cached build, 0.01s execution).
`clippy-legacy-v1-refactor.log` PASSed in 36.02s on the refactored source.
Bounded specialist review closes both V1 findings and finds no new blocker in
this scope; it does not approve the whole phase.

The OFF cold-parity/capture tests now replace two old stale-cache expectations
under their existing registrations. `off-history-cold-parity-red.log` is running
before any OFF production fix. One control requires a fresh preparation cached
before OFF mutation to cold-restore the actual new state after ON preparation.
The other rejects capture of existing untrusted history until a live base has
been prepared. The separate same-gauge replacement control will also be updated
to the corrected contract after observed RED.

Inspection additionally confirms the existing OFF emitted-counter-overflow test
expects partial emission before the counter error. That pre-existing edge is
not a correctness proof for preflight-before-emission; audit/remediation of
that behavior remains part of the complete issue scope.

## OFF history invalidation correction

`off-history-cold-parity-red.log` records two expected failures after 1m14s
compilation/0.01s execution: stale fresh output fails cold restore with retained
charge inconsistency, and existing untrusted history wrongly permits capture.
Both fixtures drain/refund before reporting these failures.

OFF invalidation now checked-advances the tracker before changing trust. All
previous cut preparations become stale; automatic ON preparation prioritizes a
live Base when history is untrusted or migration-required, and synchronous
capture refuses untrusted history. A successful live Base re-establishes trust;
normal trusted dirty continuation uses Delta/rebase.

`off-history-rebuild-green.log` PASSed five affected trust registrations (1m23s
compilation, 0.01s execution). The same-charge fixture evicts/replaces rows with
the same retained gauges, rebuilds and cold-compares the replacement payload,
then proves a subsequent trusted mutation uses prepared Delta with no extra
Base. Existing stale-cut trust callers now handle fallible invalidation.

The changed process_data/progress methods exceeded the per-function CCN-8 gate
(their existing bodies already exceeded it). Lifecycle preparation, append
preparation, emission/commit and progress wait/notification were extracted with
the same order and ownership boundaries. All seven related new/changed handlers/helpers
are now <=6. `off-history-helper-refactor-green.log` PASSed the five affected
trust registrations (3m28s compilation, 0.02s execution). HistoryIdentity
binding and the remaining phase/issue requirements are still pending.

`off-history-projected-lifecycle.log` PASSed the existing projected runtime,
layout-3 continuation, V1/V2 migration, ownership and cancellation registration
on the extracted handler source (0.35s cached build, 0.13s execution).
`off-history-stale-install.log` PASSed both affected existing stale rebase and
pending/history installation controls (0.15s cached build, 0.03s execution).
`clippy-off-history.log` PASSed in 2m06s. No full matrix ran.

`complexity-off-history.json` freezes 187 changed/new Rust functions, maximum
CCN 8, with zero changed/new parser omissions. Five parser omissions are exact
unchanged HEAD functions in inventory.rs. Source digest uses sorted
file/sha256/LF framing:
`e56ef977e43c8edeb02ea3ae42abc14aef14417b1acea0afa360d0f6f2fb3984`.
The subsequent identity test fixture changes this source snapshot; preserve the
manifest as stage evidence, not final current-head provenance.

## Layout identity RED acquisition

`history-layout-identity-red.log` is running before identity implementation. The
fixture prepares an empty full-layout base, installs a compiled narrow layout,
then requires a new base despite identical row/charge gauges. After admitting
live narrow rows, replacing the layout with full retention must be rejected.
It drains/refunds before reporting both results. No schema/capability/wire
change or identity GREEN is claimed.

`history-layout-identity-red.log` records the expected combined failure after
2m28s compilation/0.02s execution: `(base_count=1, live_layout_refused=false)`
instead of `(2, true)`, after zero-credit drain.

The implementation caches a fixed 32-byte compile-time identity over the
independently framed layout/charge tags, full logical/physical schema digests
and selected indices. Tracker/Cut/WriterWork and writer history carry it;
strict restore binds the target compiled identity, legacy migration remains
unbound until its new base, and reset preserves the compiled identity. All
base commit routes bind it. Rebase/history/pending compare identities before
encoding/merge, with the actual Operator cache checked at final installation.
No wire, capability or graph fingerprint field changes. Fixed fields remain
under size_of funding; HistoryInput adds an explicit funded seed.

`history-layout-identity-first-green.log` PASSed (6m47s compilation, 0.23s),
including the new layout controls and existing runtime/new-format/V1/V2
projection/ownership/cancellation checks. This is a build/runtime result, not a
performance measurement. Bounded specialist source review found no new blocker
in identity installation, initial empty history, encoding or funding; final
phase approval remains pending.

`history-identity-writer-focused.log` PASSed all 28 directly affected prepared
writer/rebase/pending registrations (3m13s compile, 1.15s execution). The new
controls include a stale prepared cut identity, target-ID binding after restore
and reset, matching stale Tracker/Cut identities rejected against the actual
Operator cache, and mismatched pending/history refused before encoding.
Original cancellation, abandonment, owner retirement and per-admission refusal
controls remain active and passed with the new fixed fields.

The additional direct worker/history identity mismatch control is running under
one existing stale-preparation registration; it checks state/cut/history/credit
preservation after refusal and actual source release. Latest identity lint and
frozen complexity provenance are still pending. No phase 2.2 commit, new remote
operation, final CI or performance verdict is claimed.

## Identity, public inventory and recovery contract reconciliation

`history-identity-worker-refusal.log` PASSed (4m58s compilation, 0.05s
execution), preserving installed state/cut/history and resident credit after a
funded worker rejects a mismatched history identity, then awaiting actual source
release. `history-identity-projected-contracts.log` PASSed the existing selected
runtime and recovery registration (1.27s cached build, 0.19s execution).
`clippy-history-identity.log` PASSed in 3m31s. These supersede the running-worker
and pending-lint observations above, but no final complexity provenance or
phase-wide approval is implied.

The Python capability inventory now reports stream_join physical layouts
`(1, 2, 3)` while retaining semantic state_version 1. The two focused catalog
tests first failed with the expected old `(1, 2)` tuple in
`python-layout-inventory-red.log` (2.59s), then PASSed in
`python-layout-inventory-green.log` (0.53s). Scoped Ruff check and format check
also passed. This used a target-only Python source overlay with the compatible
baseline native artifact `552b6313ccabc59d5080ea30c97965075685fbb6164b8f0b6923fa75c460e924`;
`python-overlay-provenance.json` records that limitation. It verifies the
Python catalog, not new native Python recovery execution.

The phase 2.1 projection recovery helper still assumed projected and full
snapshots had identical bytes and that a full plan could restore projected
state. `prior-projection-recovery-contract-red.log` records that expected
failure (0.34s cached build, 1.02s execution). The reconciled test compares the
seven common logical metadata fields, full admission charges and subsequent
outputs while asserting the distinct layouts explicitly. It verifies full
plans refuse layout 3, authentic full V1/V2 migrate into projected plans, a
fresh narrow base is mandatory before capture, and that base cold-restores and
continues correctly. Original alias/filter/constant/multiple-consumer,
fingerprint/schema/config and caller-immutability controls remain.
`prior-projection-recovery-contract-green.log` PASSed (3m02s compilation,
0.23s execution); all owned jobs drained and Join pools returned to zero.

Normative runtime-envelope and streaming-guide prose now describes layout 3,
exact descriptors, V1 admission charges versus physical funding, migration and
history invalidation. A bounded specialist terminal call-path audit found no
descriptor bypass: runner terminal branch -> join_terminal::restore_terminal
-> restore_terminal_owned_join -> Join.restore_managed_snapshot ->
metadata::projected::validate_expected. Empty Join state follows this same
strict path before sink recovery and terminal completion commit. The existing
runtime-envelope claim that terminal recovery did not restore per-Join ingress
observations was stale: it publishes restored ended-ingress observations
without starting handlers, reopening sources or repeating final emission.
That prose is corrected. This is source-audit evidence, not an executed
whole-manifest negative fixture.

The additional malformed-layout-3 registration is running. It covers exact
root/descriptor fields, selected-index and schema-digest mismatch, carried
live/dead charge bounds and descriptor checks with no payloads in terminal
empty state, through both direct and managed restore into an already
checkpointed target. Final phase review, dictionary/sparse/worker lifecycle
controls and the issue-wide preflight/performance requirements remain pending.

## Adversarial layout validation and nested dictionary correction

`projected-snapshot-refusals.log` initially failed compilation because the
new state-projection test accessed writer-private fields. A test-only owned
observation of cut/history/trust/migration/prepared identities now preserves
their private implementation scope. No production visibility was widened.
`projected-snapshot-refusals-corrected.log` PASSed (5m13s compilation, 0.26s
execution). Both direct and managed restore reject missing/unknown root and
descriptor fields, wrong charge mode, missing side, duplicate/reordered/
broader/narrower/out-of-range columns and logical/physical digest mismatches.
It also rejects a live historical charge below its physical lower bound even
when the live summary is adjusted to match, and a dead historical charge over
the side limit even though on_end has removed all live rows. A genuinely empty
terminal base with no payloads must still supply a complete descriptor. Every
refusal preserves the installed target state, epoch, writer observations and
resident credit; a subsequent valid cold restore proves continued usability.
Final cleanup drains the job and returns both pools to zero.

Independent review then identified a concrete nested dictionary ownership
defect: multiple noncontiguous selected runs from the same parent let Arrow
concat share their identical dictionary values backing inside Struct/List.
The real process_data fixture selects rows 0 and 2 by dropping a null-key
middle row. `nested-dictionary-retention-red.log` records the expected
recursive buffer-overlap failure after zero-credit cleanup (1m14s compilation,
0.01s execution).

The retained-copy reconstruction now detects dictionary nodes recursively,
remaps their referenced keys in first-seen order and independently copies only
the corresponding values. Nested dictionaries and views recurse through the
same funded selected-copy path. All eight integer key types dispatch through
the generic implementation, and all-null dictionaries construct funded empty
values. Mapping/selection/key workspace and backing are admitted before
allocation, with cancellation checks during each row and recursive copy.
The focused GREEN run and bounded specialist funding/correctness review are
in progress. This is not yet final phase acceptance or a performance claim.

`nested-dictionary-retention-green.log` PASSed (1m21s compilation, 0.10s
execution), including both actual Struct/List noncontiguous admissions and all
prior selected-state/recovery controls. Additional runtime fixtures assert
dictionary values really compact to the two referenced entries, recursively,
and cover all-null selected dictionaries.

`nested-dictionary-retention-edge-controls.log` compiled in 1m26s but failed
before selected copy because the attempted Int8/UInt16 runtime fixture reaches
the pre-existing V1 logical-charge implementation, which supports Int32
dictionary arrays only. The real runtime edge fixture therefore uses supported
Int32 dictionaries at both levels; this phase does not broaden full-input V1
type support. The generic copier's eight key dispatches need a component test
independent of that admission contract. The corrected runtime run is active.

Bounded review found the new short validity-bitmap estimate did not explicitly
cover BooleanBufferBuilder's 64-byte aligned allocation. The estimate now
rounds validity bytes to 64-byte blocks before applying its peak multiplier;
empty values explicitly pay aligned empty offset buffers for their recursive
shape. Key values remain funded through the existing Vec-based geometry.
Neither change relies on unrelated control-credit slack.

The intermediate `complexity-nested-dictionary.json` found five changed/new
functions above CCN 8. The reconstruction dispatcher, dictionary workspace
budget, charge proof and two writer-test stale-cut dispatches were extracted
without changing behavior; the manifest remains intermediate until the
refactored source is measured and its focused checks pass.

`nested-dictionary-retention-edge-corrected.log` PASSed (1m31s compilation,
0.10s execution) on the explicit bitmap/empty-offset funding and extracted
helpers. `dictionary-copy-component-controls.log` then found test-only seed
path/qualification compile errors, corrected before the next build.
`dictionary-copy-component-corrected.log` PASSed (1m21s compilation, 0.14s
execution). The private component fixture exercises all eight integer key
dispatches with a nonzero slice/validity-bit offset, reordered and duplicate
keys, a null key and first-seen remapping. It checks exact key/value order,
source immutability, recursively independent buffers and the aligned bitmap
estimate. Early/middle/final cancellation and a genuinely exhausted pool all
refuse without leaking credit. These component controls intentionally do not
claim to verify generic worker admission/abandonment or native sparse repair.

Bounded specialist source review confirms the dictionary alias and explicit
funding findings are closed; it does not grant phase-wide approval. The new
emitted-counter preflight RED is now running under the replaced existing
registration. It requires OFF/full and ON/projected handling to refuse a
known counter overflow before output or sequence/state commit, while allowing
the intended writer invalidation. Collector/gather execution failures remain
a separate partial-emission lifecycle contract.

## Emitted-counter preflight correction

`emitted-counter-preflight-red.log` records the expected failure after 1m01s
compilation/0.01s execution: with emitted_match_rows already at u64::MAX,
the old handler advanced output sequence 4 -> 5 and emitted a message before
reporting counter overflow. The replacement fixture drains both jobs and
refunds the pool before asserting the failure. It covers OFF/full and
ON/projected handling once the correction passes.

`emit_prepared` now computes the checked prospective emitted total after the
existing materialized-range and output-sequence validation, before calling
any collector. A successful emission returns that total, which finish_data
assigns without a new fallible counter operation. Empty output preserves the
prior total. This maintains the existing budget/sequence error precedence and
does not claim collector/gather execution failures are atomic. The adjacent
retained charge recomputation in commit_prepared has already been proven by
the same retained rows/charges/gauges during state admission; narrowing does
not alter those historical charges.

The scoped serial runner is currently checking the corrected overflow,
selected-state/dictionary/recovery controls, both changed stale-cut helpers
and core lib/test Clippy. No CI, release build or performance measurement is
part of this run. Generic selected-copy worker lifecycle and actual native
sparse repair remain explicit remaining phase 2.2 controls.

`emitted-counter-preflight-green.log` is a compile failure (the borrowed
runtime prevents borrowing the whole operator for the new pure counter
helper), not an executed GREEN. The helper now receives the current count,
row count and operator name explicitly, preserving the same validation order.
`emitted-counter-preflight-corrected-green.log` PASSed (1m20s compilation,
0.01s execution) for OFF/full and ON/projected controls. Bounded specialist
source review found no remaining emitted-counter blocker and confirmed the
budget/sequence-before-counter order, empty-output preservation and the
limited retained-charge preflight argument above.

On that same refactored source, `dictionary-copy-refactored.log` PASSed the
selected-state/runtime/checkpoint/dictionary registration (0.12s cached build,
0.11s execution). `identity-stale-cut-refactored.log` and
`identity-pending-refactored.log` PASSed the two modified stale-cut helper
registrations (each 0.11s cached build, 0.02s/0.03s execution respectively).
The serial runner then entered `clippy-dictionary-preflight.log`; that check
remains pending at this observation.

`complexity-dictionary-preflight.json` freezes this intermediate source:
270 changed/new functions, maximum CCN 8, no changed/new parser omissions.
Five omissions in inventory.rs are exact unchanged HEAD excerpts. Sorted
file/sha256/LF source digest:
`8fb8c733bcc2cf8895bef1d7eb498cb831941f5a84e71908a7d62c03229c811f`.
Whitespace and generated-schema no-drift checks pass. This is current stage
provenance, not final phase provenance; subsequent worker/sparse control edits
will supersede it. No performance target, CI or whole-phase final approval is
inferred from these focused results.

`clippy-dictionary-preflight.log` found a production wildcard import and two
identical match arms (1m31.843s runner observation); both were corrected without
lint allowances or behavior changes. `clippy-dictionary-preflight-corrected.log`
PASSed core lib/test Clippy in 1m30s. The passing behavior tests above were not
repeated for these import/pattern-only changes. `cargo fmt --all --check`,
generated-schema no-drift and whitespace checks passed; the explicit whitespace
scan includes all 53 changed and untracked source/document/artifact files.

`complexity-dictionary-preflight-corrected.json` supersedes the preceding stage
manifest: 270 changed/new functions, maximum CCN 8, the same five exact unchanged
inventory parser omissions and no changed/new omissions. Source digest:
`8b0d52243ded7c229f352df7f6fbebedbcb82bd5a72846dff61d0f352fd09c9e`.
This is a verified local stage, with remaining real-worker/sparse controls,
phase final review and all eight original performance targets still pending.

## Real native sparse projection control

The new state_projection/tests/sparse.rs fixture runs 1,024 actual native
Shared rows with an unused UTF-8 field preceding key/time/value. A right
watermark of 1023 removes rows 0..1022 and leaves row ID 1023 at event time
1024. The original selected backing exceeds 4096 bytes and four times the live
payload, so this exercises real replacement eligibility. It requires a new
chunk/offset zero, three physical columns with exact key/time/value, unchanged
V1 charges/status versus the full plan, and pending Upsert rebinding. A matching
probe must emit exactly one row and agree with full-plan projection. It also
checks caller immutability, a new layout-3 managed restore, and an escaping old
value buffer keeping job drain pending until its release and pool-zero cleanup.

The initial `retained-sparse-progress-red.log` is a fixture compile failure,
not an expected production RED: watermark_micros is typed Option<EventTime>.
The observation now explicitly converts EventTime to integer microseconds.
A suspected publication omission from handler early returns was disproven by
bounded review of the actual source and HEAD: repair_sparse_chunks already
publishes ingress progress after both successful repairs. No product change
was made for that hypothesis; the new watermark assertion is a coverage control.

Read-only specialist review confirms narrowing remains through replacement:
owned_payload_columns receives chunk.columns, payload construction uses
selected schema bytes, and initialize_funding uses retained_schema. Installation
preserves row ID/time/key/charge, marks the new dense position, rebinds pending
payloads and removes the old live reference. Cancellation precedes installation;
optional-admission refusal keeps/requeues the original chunk. This fixture
deliberately covers Shared rather than claiming generic RestoredV2/dictionary
sparse support. `retained-sparse-projection-controls.log` and its scoped Clippy
follow-up are now running; real generic-copy worker lifecycle remains pending.

`retained-sparse-projection-controls.log` PASSed (1m20s compilation, 0.20s
execution) after the fixture-only type correction. It additionally asserts
one emitted match and exact physical key/value 1023 and timestamp 1024, avoiding
an empty-output oracle. All sparse replacement, pending rebind, full-charge
parity, managed restore and escaping-buffer/pool-zero controls completed.
`clippy-retained-sparse-controls.log` found one redundant closure (23.895s
runner observation); replacing it with EventTime::as_micros leaves behavior
unchanged. `clippy-retained-sparse-corrected.log` PASSed in 23.85s; no unchanged
passing runtime test was repeated for that style correction.

`complexity-retained-sparse.json` freezes the latest Rust source: 275 changed/new
functions, maximum CCN 8, zero changed/new parser omissions; five inventory.rs
omissions remain exact unchanged HEAD excerpts. Source digest:
`e159e5e34556588f016090951996744ac1ac6deb87bb5a81b7c6177fe8614e8a`.
Generated-schema no-drift and diff whitespace checks pass. All owned test/lint
commands in this stage are settled. Real generic-copy worker admission,
cancellation and abandonment controls remain the next phase requirement,
followed by final source/artifact reconciliation, specialist review and a local
commit. All eight original performance targets and later issue phases remain
active; no new performance attempt, CI or remote operation was performed.

The final stage formatter check PASSed, and the explicit whitespace/final-newline
scan PASSed all 54 changed or untracked files, including this stage's new test.

## Real generic selected-copy worker lifecycle controls

The existing selected-state registration now owns one isolated TestService
outside its Tokio executor and shuts that service down after the executor.
New worker fixtures use nullable Float64 selected payloads and an empty
opposite input, naturally forcing CopyWork with no SQL/output gather.
Test-only, thread-local instrumentation pauses that actual native worker
before copying or after successful result construction, before installation;
the non-test run retains its original copy call and ownership boundaries.

The controls cover input-stage cancellation and abandonment, result-stage
cancellation, actual Attempt/Generation credit refusal, and cancellation at
the Attempt refusal boundary. AdmissionProbe gains an optional test-only token
to trigger that race after its real insufficient-headroom event is recorded.
They check actor responsiveness, unchanged installed status/writer/counters,
no output/pending commit, immutable caller input, selected-input ownership and
absence of an omitted full-parent owner while blocked, positive funded credit
and pending job drain until native release, then pool-zero cleanup. Ordinary
refusal must preserve the operator for successful retry with the probe spent.

`selected-copy-worker-controls.log` is the first focused build/run of these
coverage fixtures; its result is pending. The new instrumentation is private
and cfg(test), not a public control API or a performance measurement. The prior
frozen complexity snapshot predates these changes; final phase review and
current-source provenance remain pending alongside all eight issue targets.

`selected-copy-worker-controls.log` compiled in 2m28s, then failed in 0.02s
because the fixture expected a wrapper phrase instead of the preserved native
DataFusion resource error. The three paused cancellation/abandonment controls
completed before that assertion. No product admission behavior was changed.
The error oracle now checks both Resources exhausted and the actual native
reservation consumer, alongside node ID and the real fee-minus-one event.
A fixed Float64 literal oracle supplements shallow Arrow equality before
stopping/retry and after successful retry, so selected-value mutation would be
detected independently of the caller's shared buffers.

`selected-copy-worker-corrected.log` PASSed (1m21s compilation, 0.18s execution),
including all six new worker controls and the existing selected-state suite.
Every blocked worker held selected inputs but no omitted parent after caller
release, kept funded credit/job drain alive until release, and then refunded
to zero. Both refusal stages recovered for successful retry; cancellation at
refusal returned Cancelled with no worker execution or state/writer/output
commit. Bounded specialist review found no new ownership, funding or test-hook
blocker. These are actual CopyWork controls, distinct from the earlier pure
copier cancellation/refusal component tests.

`clippy-selected-copy-worker.log` found one test-only &Option lint; the helper
now accepts Option<&Hook> with no behavior change. Its corrected scoped lint
run is active; unchanged passing runtime tests were not repeated. The issue's
explicit doc-writer checklist is being fulfilled by a bounded reconciliation
of runtime-envelope, streaming-guide and the qualifying Oct14 changelog.
Fresh read-only issue inspection still reports OPEN, last body update
2026-10-10T12:24:17Z; no checklist or acceptance scope was changed remotely.

The corrected scoped Clippy run, `clippy-selected-copy-worker-corrected.log`,
PASSed in 56.81 seconds. The final Rust provenance inventory
`complexity-phase22-review.json` covers 50 Rust files and 293 changed/new
functions, maximum CCN 8, with no changed/new parser omissions. Its digest is
`706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302`.
No unchanged passing behavior check was repeated for the test-only lint fix.

The assigned cf-doc-writer reconciled runtime-envelope, streaming-guide and
CHANGELOG, verified 84 local links/anchors, whitespace and final newlines,
unchanged tables, and no project-schema drift. These documentation edits do
not establish performance acceptance. Final whole-phase specialist review
remains pending; full CI, coverage and all eight issue targets remain unproven.

Final handoff checks PASS: cargo fmt --all --check, git diff --check,
generated project-v3 schema no-drift, and an explicit whitespace/final-newline
scan of all 58 changed or untracked files. The manifest exactly matches all
50 changed/new Rust files and every recorded content hash. No build or
formatter handle remains live. This local candidate is ready for final
specialist review; no remote CI exists for this unpushed increment.

## Final local specialist review

cf-reviewer approves the complete phase 2.2 implementation at
`e884adc362ac8f8d03c1364d0dfdb4e205407696`, with no remaining blocking or style
findings. The reviewer independently matched all 50 Rust source files between
the committed tree, worktree and provenance manifest, including the aggregate
`706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302`.
The [report](../reviews/issue406-join-state-projection.md) records the focused
evidence, old-native Python inventory limitation and unproven acceptance gates.
No new builds, tests, benchmarks, CI polling or remote mutations were required
for this read-only review. Phase 2.2 is locally reviewed and committed; remote
required checks, combined 90% coverage, release/native Python integration and
performance are still pending. This does not complete issue 406 or certify
merge readiness. The full objective and other phases remain active.
