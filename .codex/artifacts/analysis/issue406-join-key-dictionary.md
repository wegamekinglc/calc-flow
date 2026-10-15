# Issue 406: Join persistent key dictionary implementation

Status: local implementation and checks complete; final review and CI pending.
Isolated base: `798a97ee44b701b00bb89196a6483b01c62e16bf`.

The implementation follows the approved
[specification](../specs/issue406-join-key-dictionary.md) and
[critic](../critiques/issue406-join-key-dictionary.md). All FR1–15 and AC1–13
remain obligations. The prior batch and codec sources remain frozen and reviewed.

Join now retains one operator-local canonical identity per live key across both
sides. Column-bound typed input reuses known identities without rebuilding V1
frames; only new retained distinct keys construct canonical owners. Native
indexes consume shared leases and dense identities, and probe-only unknowns do
not grow persistent state. Funded provisional births, capacity replacements and
reference changes publish only after all output emissions succeed.

Canonical bytes, namespace/lease controls, directory capacity and legacy-row
bindings have separate actual-owner lifetimes. Genuine optional refusal keeps
complete SQL matching and full V1 fees; explicit rebuild restores complete native
coverage. Recovery rebuilds private identities after historical validation and
preserves the existing layouts and writer/history selection. New deferred V1
history validation rejects a bad fee on a subsequently tombstoned row while
preserving earlier structural/live/writer error priority.

## Initial focused RED

Evidence directory: repository `target/issue406-join-key-dictionary-20261015/`.
`persistent-keys-red.log` records two genuine failures through `process_data`:

- Known keys crossing batches/sides encoded four canonical owners against two.
- Equal retained values across both sides did not share a canonical Arc owner.

Both controls first passed five-row sorted output, physical IDs and full V1
row-fee assertions. Fee oracles were computed before the work-counter reset;
the counter covers complete production admission, matching and retention.
Compilation took 7 minutes 59 seconds; execution took 0.01 seconds. No production
source changed before these failures; all owned handles were settled afterward.

## Internal design boundary

The first implementation slice passes both initial controls in
`persistent-keys-first-green.log` (6 minutes 53 seconds compilation, 0.04 seconds
execution). It stages a private dictionary snapshot and independent canonical
owners, installing only after output completion. This is deliberately not final
acceptance: native side indexes still use their legacy content lookup, and the
typed binding, eviction, refusal and recovery work remains incomplete.

The next RED acquisition accidentally omitted the shared target environment.
It was interrupted with exit 130 before tests and has no RED evidentiary value
(`typed-dispatch-red.log`). No compiler children remained before the correctly
configured acquisition `typed-dispatch-red-shared-target.log` started.

That correctly configured run reports two genuine RED assertions (3 minutes
10 seconds compilation, 0.02 seconds execution): known-key type dispatch grows
from 54 to 106 when selected rows double with the same two records; Int8 uses
eight legacy frames rather than two distinct canonical owners. The original
two cross-batch controls still pass. All build handles settled before typed
codec changes.

`typed-dispatch-green.log` passes all four controls (4 minutes 17 seconds
compilation, 0.03 seconds execution), including independent exact V1 frames for
all 17 tags and all timestamp units/timezone variants. Known-key dispatch now
depends on record/key-column binding rather than selected row count.

`identity-retirement-red.log` then records four genuine failures: native content
hashing remains three; probe-only unknown input retains directory capacity two
and a namespace; twenty expired identities remain live; right-side expiry leaves
three references rather than one. The corresponding focused union is eight GREEN
in `identity-retirement-green-fixed-fixtures.log` (4 minutes 58 seconds compilation,
0.04 seconds execution). An earlier compile failure from two Option/Arc fixture
accesses is preserved in `identity-retirement-green.log` and is not behavior RED.
Fixture owner assertions now account for one dictionary lease plus one distinct
probe owner, independently of repeated row count. Old native-index standalone
byte-oracle constructors are test-only; production identified constructors own
actual leases and use address/pointer equality without canonical content hashes.

The next refusal/ownership RED first failed test module path resolution before
execution (`refusal-ownership-red.log`); the corrected acquisition is separate.
No production change is justified by that path error.

The bounded maintained harness validator and candidate-only preflight
registration have focused Python RED/GREEN evidence. Their tests validate exact
retained rows/fees/output, actual pool and SQL witnesses, cleanup and separate
source/binary identity recording. A reviewer-found boolean-as-one list witness
has its own minimal RED/GREEN pair. These unit checks do not yet establish an
actual production refusal diagnostic; that execution remains pending.

One crate-private identity component will own namespace, opaque address and
immutable entry leases. Join retains its V1 codec and time-qualified runs; ASOF
and its wire/rank semantics remain unchanged. A copied history address is never
trusted for current lookup without namespace, live-entry and canonical-owner
validation. History bytes hold their own credit without leasing a live slot.

Prepared births and retained-reference changes remain private until successful
output completion. Native indexes will map identities to their existing runs,
with no independent canonical-content dictionaries. Canonical allocations have
per-owner funding separate from directory/handle capacity and batch scratch.

The genuine optional-grant refusal route preserves batched SQL and existing
legacy canonical allocation sites; its exact final helper/counter and maintained
workload inventory are pending. No new unfunded map/lease owner is permitted.

The corrected refusal/ownership acquisition reports four genuine failures in
`refusal-ownership-red-fixed-path.log`: three real pool-refusal/error paths panic
when native append encounters unaddressed legacy keys, and one expired unique
canonical owner remains pinned by an unrelated surviving key. The focused union
of thirteen controls passes in `refusal-ownership-green.log` (4 minutes 23 seconds
compilation, 0.06 seconds execution). The already-passing checkpoint/history
control is regression evidence, not a newly manufactured RED.

`cooperation-mixed-rebuild-red.log` records three further genuine failures:
cancellation during an 8 MiB value still constructs a complete canonical frame;
a forced 256-key collision chain exceeds the 64-visit cancellation bound; and
third-batch SQL correctly matches a legacy key after refusal but explicit funded
re-enable does not coalesce it. The preceding thirteen controls remain GREEN.
The first cooperative implementation compile failed because a hashbrown iterator
was held across await (`cooperation-first-green.log`); that compile failure is not
behavior RED or GREEN. A funded entry-index collision chain replaces that cursor.

An intermediate ordinary preparation snapshot copies all retained identities.
This is explicitly unacceptable as the final ordinary path: cooperative checks
bound latency but do not remove its O(D_retained) work. A separate structural
control compares the same one-key batch with one versus 128 live identities,
counting actual directory-copy visits, batch owners and unrelated canonical Arc
clones. The final path must stage only hit leases and new births, reserving a
full-directory replacement for actual capacity growth or explicit rebuild.

`cooperation-and-directory-red.log` passes both cooperative cancellation
controls and thirteen prior controls, and reports the separate ordinary snapshot
RED (one retained identity still causes one actual snapshot visit). Compilation
took 4 minutes 41 seconds; execution took 0.13 seconds. Explicit mixed-state
rebuild remains a separately identified RED.

The incremental implementation's first acquisition was interrupted after its
structural assertions passed because the test retained its admitted bundle while
awaiting owner closure (`incremental-directory-green.log`). That log is not an
aggregate GREEN. The fixed test drops the actual admission owner before draining.
The subsequent three-control acquisition reports a genuine structural GREEN at
one and 128 retained identities, and two genuine ownership REDs: a failed
state-limit admission increases backing capacity from four to eight, and byte-only
history keeps retired identity-control credit alive
(`staged-backing-and-control-credit-red.log`, 4 minutes 26 seconds compilation,
0.01 seconds execution).

Pending now owns any capacity replacement until commit. Ordinary known-key
preparation holds only its distinct matched leases, canonical aliases and local
probe IDs. Global namespace/slot handles remain separate. Only actual growth
can copy/rehash directory backing, under independent old-plus-new credit and
capacity-replacement counters; no per-batch retained-state snapshot remains.
Canonical byte credit and entry-lease control credit are split before allocation;
byte-only history does not keep a namespace or entry-control reservation alive.
`staged-backing-and-control-credit-green.log` passes seventeen controls, including
these two ownership corrections, the independent ordinary-work control and both
cooperative cancellation controls (5 minutes 25 seconds compilation, 0.20 seconds
execution). Its eighteenth control remains the explicit mixed-state rebuild RED;
this is not a complete module acceptance claim.

The next native-only refusal acquisition had a sibling-private fixture helper
compile error and was interrupted after diagnosis (`native-only-refusal-red.log`).
Its owned compiler group was settled before the local fixture correction. The
corrected acquisition is separate; no implementation is justified by that error.


The corrected native-only acquisition passes both focused controls, including
final pool zero (`native-only-refusal-green-fixed-lifetime.log`, 1 minute
10 seconds compilation, 0.02 seconds execution). The earlier aggregate failure
held a test-owned canonical Arc through cleanup; its behavior assertions passed,
but that acquisition is not reported as a full GREEN.

`incomplete-references-retired-red.log` records four genuine failures (3 minutes
13 seconds compilation, 0.06 seconds execution): the three-row control attempts
two fresh grants instead of distinguishing a persistent incomplete route; known
reference preflight scans 128 unrelated retained entries for three incoming rows;
an ordinary operation after pressure release births an identity while its equal
legacy row remains unregistered; and twenty expired leased cold keys grow the
directory while an unrelated hot key remains live.

The author, critic and root approved a funded stable-row-ID sidecar for legacy
rows. It shares actual canonical identity leases without copying or rewriting the
legacy `Arc<Vec<StoredRow>>`. This avoids imposing mandatory funded row-vector
growth on a future SQL-acceptable append. Normal identified rows require no
sidecar growth. All native build/removal/owner refresh and retirement resolve the
same effective canonical owner. Sidecar removal releases its lease before slot
retirement; real worker indices hold their own leases. Byte-only history retains
its original V1 bytes and cannot resolve a stale copied address after reuse.

A real origin grant refusal may leave a persistent `needs_rebuild` state.
Subsequent operations use SQL and the explicitly inventoried legacy framing path,
without pretending another pool failure or silently scanning all retained keys.
Only an explicit, independently counted funded rebuild can publish complete
coverage. Its candidate replaces full actual reference counts and preserves
existing live canonical identities; all partial/cancelled/refused preparation
remains private. Ordinary known-reference preflight uses provisional entry deltas
and an intrusive touched list. Its structural counter includes actual cleanup
visits; three incoming rows permit at most six slot visits, never a directory
scan. Retired leased slots are revisited through a bounded retired-only queue,
rather than scanning unrelated live entries.

These changes are still under focused validation; no complete phase 1.2 approval
is claimed.


`sidecar-and-exhaustion-red.log` passes twenty-three focused controls, including
all four preceding incomplete/reference/reclamation REDs, explicit mixed rebuild
and the diagnostic's real pool/process_data/SQL/retention/cleanup path. Four new
reduced-domain exhaustion controls fail for the desired unexpected-success
assertions. Compilation took 4 minutes 19 seconds and execution 0.05 seconds;
the owned Cargo handle was fully closed before the checks were implemented.
The diagnostic cfg(test) execution is not yet the required non-cfg(test) dev
bench witness.

The next acquisition (`unlink-result-restore-red.log`) stops at six fixture
compile errors: five redundant `std::fmt` qualifications and one test cleanup
moving a job beside its borrowed context in a tuple. These are not behavioral
REDs. Production restore, collision unlink and the delivered-result gate remain
unchanged until their corrected controls actually fail.

Rebuild publication performs a constant number of field replacements. Dropping
the replaced index, directory and sidecars may synchronously visit their owned
contents; the whole publication tail is not claimed constant-time. This explicit
rebuild cost remains a final-review caveat; no throughput or latency claim follows
from the structural tests.

The corrected acquisition `unlink-result-restore-red-fixed-fixtures.log` records
twenty-nine passes and five genuine failures: collision-tail unlink visits 256
entries; a delivered Fill result installs before the intended result gate;
authentic V1 replacement retains the old dictionary's one reference instead of
the candidate's two; valid V2 recovery attempts no optional dictionary grant;
and a dead V1 base row with stored full fee 134 instead of 133 is accepted.
The last control first fails on direct recovery, so it does not establish a
managed-recovery RED. All four reduced-domain exhaustion controls and the real
local-AtomicU64 namespace arithmetic pass in this acquisition.

Backlinks make unlink independent of collision-chain length. Restore dictionary
construction follows validation in the existing V2 worker and in a separate
observed owned CPU work item for full V1. The full V1 path keeps its original
decoder, history, writer selection and optional SQL acceptance; it does not add
mandatory V2 Container/IPC grants. A prepaid fixed capture slot holds the complete
private replacement. Its stack guard clears that slot on error or future drop.
New historical time/full-fee diagnostics are deferred until the old structural,
live-row, metrics, limit and writer checks have passed. This deliberately fixes
the previously accepted dead historical V1 fee mismatch without changing wire
formats or the existing duplicate-before-tag error priority.

`restore-publication-first-green.log` stops at twenty-one import/type errors;
none is a behavioral RED. The corrected
`restore-publication-green-fixed-types.log` passes all thirty-five controls
(3 minutes 11 seconds compilation, 0.38 seconds execution), including direct
and managed dead-V1-fee rejection, all authentic layout controls, valid optional
refusal installing the new SQL state, subsequent refused rebuild preserving that
state, and actual Count/Fill input plus delivered-result ownership.

The worker proofs use an actual refusal followed by explicit legacy-sidecar
rebuild. The test retains byte-only PendingOp history and a Weak identity,
without adding a strong test Lease. Captured immutable native indices keep real
leases through worker input and delivered output. Drop/cancel preserves old
state; the existing release barrier prevents a healthy mutation until actual
worker cleanup. Old byte history cannot resolve a new namespace after reset and
re-retention, and its own funding survives independently until its actual owner
is released.

`restore-cancellation-semantics-red.log` contains three redundant fixture type
qualifications, not behavior evidence. Its corrected acquisition
`restore-cancellation-semantics-red-fixed-qualifiers.log` reports thirty-seven
passes and two genuine failures (1.05 seconds execution). Cancellation cannot
return while an actual full-V1 DerivedWork is held; after dropping the future,
all old-state/Weak/credit/healthy-retry/pool-zero assertions succeed before the
final timely-cancellation assertion fails. Separately, actual capture-grant
refusal publishes a replacement, retires old dictionary credit, and then returns
Cancelled from cleanup. The observation pool cancels on that real credit
retirement; a successful terminal publication is allowed, while an error must
preserve the old state. Composite sparse/sliced nullable input, nested projected
payloads, independent full V1 fees, long composite collisions, ordered matches
and partial collector failure all pass in the same acquisition.

The corresponding fixes select job cancellation/deadline while awaiting the
derived ticket, and move whole existing decisions into local owners before
cleanup and the last cancellation check. PreparedSides retains its workspace,
SchemaSuccess retains its descriptor credit, and metadata-only decisions contain
only inline scalars. Only then does the original synchronous install run. No
additional grant, unfunded candidate, map, Box or changed cleanup contract is
needed. All thirty-nine persistent-dictionary controls subsequently pass in the
serial directly affected union below, including both cancellation corrections.

The first broader affected acquisition attempted eighty-four tests with the
default parallel test runner and exhausted the bounded test service's process
resources (`restore-cancellation-and-affected-controls-green-fixed-stop.log`).
Its SIGABRT is infrastructure failure, not an aggregate GREEN. After settling
the process, the same binary ran serially: sixty-four passes and twenty failures
(`restore-cancellation-and-affected-controls-serial.log`). These exposed a real
restore-worker regression and obsolete credit oracles; the previously observed
thirty-nine persistent controls all passed.

The restore worker's input snapshot deliberately contains only segment data,
without the actor metadata JSON. Looking up event-time indices there incorrectly
changed a prepared worker decision into actor fallback. Workers now borrow the
two already compiled indices from SchemaWork before consuming schema validation;
no new name clone, allocation or layout field is needed. Existing Prepared,
copy-gate, worker, cancellation and error-priority assertions remain unchanged.

Empty full-V1 restores skip the new dictionary capture when their declared
retained counts are both zero; all original validation still runs. A healthy job
whose gather home was explicitly closed also retains its previous valid SQL
restore acceptance. Only that closed-capability case may decline new derived
work without an actual memory grant failure. A separate `closed_restore_routes`
counter records it; cancellation/deadline is checked and propagated, and
`dictionary_grant_refusals` is not incremented. The nonempty regression proves
installation of the new SQL candidate, literal full fees and carried history
owners. This narrow restore capability boundary is explicit final-review scope;
it must not be represented as a fresh pool refusal or an ordinary type fallback.

The next current-source serial union reports 113 passes and three failures among
116 controls (`persistent-and-consumer-controls-green-fixed-test-visibility.log`,
3.36 seconds execution). All persistent controls and the restore-worker fixes
pass. The three remaining assertions omitted new, genuinely surviving
dictionary ownership while the target operator remained alive. Their corrected
oracles independently sum actual identity funding, require exact live credit,
then drop the operator and retain final pool-zero checks. All three pass in
`historical-diagnostic-overlap-red.log`; the healthy closed-home control passes
there too.

Native allocation controls now compare actual allocation peaks against the
observed prepaid overlapping credit for the same production preparation. Exact
headroom minus one refuses and refunds; exact headroom plus one accepts. This
replaces the former per-batch arena ceiling with independent persistent-owner,
replacement-overlap and last-alias retirement assertions. The existing sparse
Int8 SQL control now uses a genuine native-index grant refusal, witnessed by the
pool, because Int8 is accepted natively by this increment. Its physical IDs,
output, full fees and recovery assertions are retained.

Deferring the first historical error creates a real dynamic-owner overlap with
an older priority error being formatted. A focused allocation control keeps two
actual checkpoint errors alive together and compares their peak with the former
diagnostic inventory. It records genuine REDs at DEL names of 340 characters
(8,960 bytes versus 6,973 funded) and 600 characters (17,920 versus 11,913);
the 1,024-character case is already covered and is not called RED. The shared
three-worker inventory now reserves the original sum of checkpoint/time/reason
construction peaks plus their maximum. The first-only historical latch owns at
most one retained diagnostic, bounded by one such construction peak; the
original sum continues to cover the old priority path. Arithmetic is checked,
and no row fee or wire field changes. The final current-source union below
supplies GREEN for all three names.

The first scoped core lib/test Clippy pass found legacy test-reference paths
still compiled in production, small ownership/literal lints, two long test
helpers, and an oversized async future. Reference paths are now test-only;
the helpers were split without changing assertions. The owned probe path borrows
the admitted container, scopes synchronous capture before await, and restores
declined or recovered admission in place. This avoids copying its bounded inline
owner into nested async inputs/results, without boxing, a new grant or a lint
exception. SQL and V2 loan failure extraction is synchronous, and their retry
paths recover the admitted owner in place before awaiting. The native recovery
path also returns only key metadata after returning admission to the caller.
Four test-only wrapper futures use Box::pin; production does not add a boxed
owner or mandatory grant. `clippy-final.log` passes core lib/tests with
benchmark-internals and warnings denied in 1 minute 46 seconds. The final
current-source affected union passes below.

## Compatibility allocation inventory

Only `join.rs::legacy_retained_key` retains the absent-native-key compatibility
allocation: `encode_join_key_columns_v1` produces its existing Vec, conversion
through `FramedKey::from` produces the existing canonical bytes, and Arc wrapping
owns them without separate workspace funding. This path is selected after a
real dictionary grant refusal, the resulting explicitly counted incomplete
state awaiting explicit rebuild, or a valid SQL-only restore whose new optional
derived work was unavailable. The latter includes the separately counted
healthy closed-home capability described above and requires final-review
confirmation. Its `legacy_key_encodings` counter covers every
retained row on that route; an equal known owner is recovered by bounded content
lookup. Healthy native-index, capture or submission refusal reuses already
prepared dictionary keys and does not select this allocation exception.

The maintained `join_dictionary_refusal_retention --check` component covers
left[a,b] then right[a] through real process_data commits and an actual pool
refusal, with exact full fees, SQL availability, result parity, distinct actual
refusal/needs-rebuild counters and final pool zero. Its cfg(test) execution is
GREEN above; the separately compiled non-cfg(test) dev-bench witness passes below.
Checkpoint historical decoders still construct original wire-key bytes
for validation; those existing recovery allocations are outside ordinary
admission and must precede coalescing of live identities.

## Focused acceptance mapping

- AC1–3: `persistent_dictionary_tests` covers cross-batch/cross-side aliases,
  distinct canonical construction, probe-only unknowns, column binding counts,
  and fixed known input with one versus 128 unrelated identities.
- AC4–5: exact independent V1 tag frames, composite sparse sliced nullable
  records, nested projected payloads, physical ID gaps, output order, full fees,
  and unchanged typed state/match-limit controls cover semantic parity.
- AC6: `persistent_dictionary_refusal_tests`, existing paid-scratch zero-headroom
  controls and the maintained refusal diagnostic distinguish real grant failure,
  persistent incomplete coverage, native/capture/submission decline and genuine
  non-budget failure. The non-cfg(test) diagnostic passes below.
- AC7–9: cooperation and worker modules cover long-value and collision-chain
  cancellation, actual Count/Fill input and delivered-result leases, abandoned
  work, partial collector failure, side-specific eviction, byte-only history,
  cold-slot reclamation with a live hot key and final real-owner credit release.
- AC10: four reduced-domain exhaustion controls and the local AtomicU64
  MAX-1/MAX algorithm control cover slot, reference, byte-length and namespace
  boundaries without huge allocations.
- AC11: authentic layout 1/2/3 and full/projected restore controls distinguish
  invalid/cancelled old-state preservation, valid refused new SQL installation,
  later refused rebuild preservation and eventual native continuation. Existing
  deleted V2 historical key/time/fee controls accompany the new dead V1 fee check.
- AC12: `operator/key_identity.rs` is the one crate-private namespace/address/
  lease implementation consumed by Join; NativeIndex has no content dictionary.
  No ASOF source or checkpoint format is changed.
- AC13: final scoped checks, source seal, documentation integration and mandatory
  specialist review are tracked separately below. Local tests do not establish
  full CI, cross-platform or the combined 90% coverage gate.

## Final local checks

`final-focused-green.log` passes 141 affected controls with zero failures:
compilation/linking takes 2 minutes 16 seconds and execution 3.65 seconds. This
serial union includes persistent dictionary, metadata/restore workers, native
lookup, batch admission, dictionary order, SQL scratch ownership/retry and two
malformed selected-history controls, plus existing V2 history and typed limits.
The historical overlap peaks are 8,960/17,920/17,920 bytes against independently
funded 13,276/22,896/38,584 bytes for names of 340/600/1,024 bytes.

Core lib/test Clippy with benchmark-internals and warnings denied passes in
`clippy-final.log`. Repository formatting check, generated project-v3 schema
no-drift and whitespace check pass. The source audit against the frozen base
includes tracked and newly added Rust files: 441 changed/new functions, maximum
CCN 8 and no changed-function parser omissions. Its Rust aggregate SHA256 is
`f6b333f9b77a44890a2b6225c4f60f60f0da2523a24aa7524780a95d2c383761`.
Three parser exclusions have function excerpts byte-identical to the base;
their exact hashes and every changed-file hash are in `complexity-final.json`.

Focused maintained Python validator/preflight checks and scoped Ruff previously
passed and their unchanged source was not rerun. Some earliest Python outcomes
exist only in captured tool output, not a retained raw log; the diagnostic
progress artifact distinguishes these from named log evidence.

The maintained three-row diagnostic passes through the dev bench binary with a
non-cfg(test) core library (`diagnostic-native-control-provenance.json`). It
observes one actual dictionary try_grow failure for 1,472 bytes, one persistent
needs-rebuild route, three legacy frames and one SQL table build. Two successful
SQL pool grows independently witness available SQL memory. Real process_data
commits retain two left rows and one right row, with full fees 114 each,
retained bytes 228/114 and the sole a/a match. Limit-failure metrics remain zero;
awaited owned cleanup returns pool credit to zero. The strict Python validator
passes on the actual output. Calling the same binary without --check fails with
the required diagnostic.

This is correctness evidence only. The library artifact has test=false,
opt_level=0/debuginfo=2; Cargo labels the bench artifact test=true with the same
dev optimization settings. Build time is 115 seconds. The feature set is only
benchmark-internals, and binary SHA256 is
`5c0080e7e06b61943b96d4921197855584e10b26a00c624e21fdeec5a808d39e`.
All 633 inventoried source inputs and the binary remain byte-identical before
and after build/execution. The broader input aggregate is
`f7075beee671beb2ebb907cb7d69965a947956912dd81c0ae71d150a05af3b4f`;
its domain differs from the changed-Rust-only complexity seal. No timing sample
or release-artifact claim is made. All build, formatter and runtime handles are
closed before source handoff.

## Outstanding evidence

Documentation is integrated in the delivery branch. Final specialist review and
required CI checks remain pending. No local result substitutes for those gates.
Phase 1.1 offset-derived fees, retained chunks, pairs, journals and all eight
original performance targets remain unfinished. The user's narrowed delivery
scope freezes their further development; this task only finishes the current
persistent dictionary and its required review/delivery work. No performance
acquisition, native package build or full matrix is part of this implementation.

## Final review B1: repeated live row IDs

Formal review found that the V1 live-identity check rejected only repeated
(event time, row ID) pairs. Two live rows on one side with the same row ID but
different valid times could reach the new sidecar's uniqueness assertion. The
authentic two-row checkpoint changes only its second row-ID header; both times,
payloads and full fees remain valid. Direct and projected restore panic, managed
restore reports TaskPanicked, and a dictionary-refused route incorrectly accepts
it in the initial acquisition. Its closed-home fixture self-wait was interrupted
and recorded separately; it is not a completed group result.

The fix reuses each existing 16-byte pair tree ordered by (row ID, event time).
The original per-row validation loop latches repeated IDs through a bounded
same-ID range. Inline per-side facts are reported only after original payload,
metrics, limit, writer and deferred historical checks, before optional dictionary
work. There is no second tree, additional full-state tail scan, changed row order
or new checkpoint format. Existing V2/V3 history already rejects reused row IDs.

The final changed control passes all five direct, managed, projected, configured
refusal and healthy closed-admission routes. It proves typed mismatch, unchanged
old state/writer/history/canonical owner, healthy continuation, no dictionary
grant attempt and final pool zero. Independent state and writer reservations
plus actual home/generation/attempt controls exactly account for the pool. The
projected writer's 8,873 bytes come from actual Base and segment reservation
owners, deduplicated by reservation pointer. Closing admission legitimately
retires one 16,384-byte generation reservation while the state remains 15,896
bytes. The test-only segment accessor and writer helper deliberately require
fresh direct segment owners rather than claiming a general restored-owner ledger.

Prior-payload-error priority and authentic-layout continuation controls passed
in the preceding focused acquisition. Their production restore bytes are exactly
unchanged in the final fix; only the failed main control's test-only accounting
was subsequently corrected. No passing unrelated controls were repeated.
`validation-review-fix.json` records these revisions, logs and source equality.
The original 141-control run and maintained non-cfg(test) diagnostic remain
evidence for ca3de78c; their binaries and source aggregates are not relabeled as
new builds. This bounded restore-only delta leaves the diagnostic/process_data
paths unchanged.

The final B1 changed control passes in `duplicate-id-root-green.log` (1m55s
build/startup, 0.02s test runtime). Final scoped Clippy passes in 1m30s;
formatting, generated-schema no-drift and whitespace checks pass. Final inventory
contains 58 changed Rust paths and 464 changed/new functions, maximum CCN 8 with
no changed parser omissions. Its exact sorted-path seal is
`1add6adfb5b2acb0a04a50e2fc4b2ae8f7d031f93201eb2ccc7120a856eaa8e6`.
All build and test handles are closed. Formal final approval and required GitHub
CI remain separate delivery gates.
