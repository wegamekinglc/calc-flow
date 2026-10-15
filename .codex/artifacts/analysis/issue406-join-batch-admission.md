# Issue 406: Join batch admission implementation

Status: locally verified incoming batch-control increment; final specialist
source review pending. No performance or issue acceptance verdict.
Base: `bb142320062995aeb318ad406c901288537fd766`. All eight targets remain active.

The isolated branch is `feature/issue406-join-batch-admission`. Its design follows
the clarified [specification](../specs/issue406-join-batch-admission.md) and
[critique](../critiques/issue406-join-batch-admission.md). Incoming ownership moves
to ordered record segments with columnar selection metadata. Retained rows,
matched pairs and checkpoint row operations remain separate unfinished phase 1
work. Offset-derived charge sums and vectorized full-fee calculation remain
explicitly unfinished phase 1.1 work; the persistent key dictionary remains
unfinished phase 1.2 work.

## Focused RED evidence

Evidence lives under repository `target/issue406-join-batch-admission-20261014/`.

- `dense-admission-red.log`: the initial focused test observed 130 owning
  incoming row records for two 65-row generic batches, against the required zero.
  Build and execution took 418.320 seconds; test execution took 0.01 seconds.
- `segment-ownership-red.log`: an independent assertion before retained-state
  construction observed 65 strong owners of each record parent, against one.
  Compilation took 2 minutes 20 seconds; the focused test took 0.00 seconds.
- `generic-sql-red.log` is a fixture error, not valid behavior evidence: Float64
  Join keys are outside the supported key contract. The corrected fixture uses
  supported Int8 keys, which require generic SQL equality.
- `generic-sql-corrected-red.log` and `generic-sql-schema-corrected-red.log`
  record further fixture mistakes: the right payload is Utf8, and null key/time
  cells require nullable declarations. They establish no behavior RED.
- `generic-sql-nullable-corrected-red.log`: genuine Int8 SQL equality executed
  once, nonempty matched pairs, durable IDs `[0, 4, 5]`, and drop metrics passed.
  The final ownership assertion observed three incoming owner records against
  zero. Compilation took 1 minute 13 seconds; execution took 0.01 seconds.
- `workspace-preflight-red.log`: a 65-row input with zero pool headroom and an
  overflowing next row ID reported the old row-ID error before the required
  metadata workspace refusal. Compilation took 3 minutes 44 seconds; the
  focused test took 0.01 seconds.

## Ownership and representation decisions

The temporary immutable table owner stays within serial admission. Borrowed
access returns a payload position without cloning an owner or creating a row
batch. Owned transfer detaches only selected record parents after checked credit;
refusal selects serial evaluation. Payload replacement releases the temporary
table once no borrowed segment remains. The durable ID derives from the physical
record ID base and selected original position; dense probe and copied-payload
positions remain separate.

Inline metadata covers at most 64 physical rows and four record segments. Larger
capacity is checked and funded before admission classification. Mandatory
workspace refusal therefore precedes semantic row errors. After this preflight,
row ID, null time, timestamp conversion, null key, lateness and metric failure
retain their existing physical-row ordering. The runtime's default pool policy
does not change.

## Legacy parent funding boundary

At the existing retained-state boundary, borrowed retained records acquire one
legacy `Arc<RecordBatch>` parent per record, preserving the old legacy ownership
contract. This does not create one batch view per retained row. The independently
funded detach before an owned native/SQL input is a separate operation; its
reservation covers only required record parents and moves with the batch owner.

The bounded advisory review compared this with `bb142320` and withdrew a proposed
new state-funding blocker. The old `admit_legacy_record` already allocated one
unreserved `Arc<RecordBatch>` and column vector per record. Moving that parent to
`prepare_retained_parents`, skipping existing owners and records without retained
rows, preserves that legacy contract. A `Rowed` parent contains no whole-table
owner. Detach workspace covers owned transfer, rather than promising new funding
for the entire lifetime of legacy state. Independently funded projected Shared
and restored payloads keep their existing lifetime funding. This caveat does not
excuse new unfunded metadata, duplicate parents or broader background retention.

## Incremental GREEN and subsequent regressions

- `batch-admission-first-green.log`: the initial two focused controls passed;
  compilation took 6 minutes 51 seconds, execution 0.01 seconds. This covered
  sparse Int8 SQL and independent dense parent/iteration assertions, together
  with four inline/spill and workspace-priority cases. It did not cover an
  eligible first copy followed by a borrowed record.
- `mixed-pending-red.log` is an intermediate refactor compile error, not valid
  RED: the extracted admission loop initially omitted its `SidePlan` parameter.
- `mixed-pending-corrected-red.log`: two independent controls reproduced the
  advisory's pending-record bugs. Lookup included uninitialized suffix segments
  (`record segment` panic); replacement discarded a table still needed by later
  records (`borrowed admission table` panic). Compilation took 5 minutes 24
  seconds; execution took 0.02 seconds. Hashes are in `mixed-pending-red.json`.
- `mixed-pending-green.log`: restricting lookup to initialized records and
  retaining the table while records remain pending passed both controls.
  Compilation took 2 minutes 6 seconds; execution took 0.00 seconds. The actual
  first-copy/second-borrow witness also checks physical IDs, order and full fees.
- `last-drop-and-nested-view-red.log`: the completed batch retained an omitted
  dropped suffix, and nested output costing created two one-row record views.
  Existing compact dictionary output and budget assertions passed before the
  view assertion failed. Compilation took 4 minutes 42 seconds; execution took
  0.01 seconds. Hashes are in the matching JSON file.
- `last-drop-and-nested-view-green.log`: releasing the table when the final
  record completes and measuring nested Arrow slices directly passed both
  controls. Compilation took 2 minutes 4 seconds; execution took 0.01 seconds.

- `ranges-packed-retention-red.log`: independent storage assertions observed
  dense sparse-index capacity 130 against zero, inline retention storage 64
  bytes against eight, and 130 physical indices against 65 for a single sparse
  record followed by a dense record. Compilation took 3 minutes 42 seconds;
  three tests failed in 0.01 seconds. Source hashes are in the matching JSON.
- `sparse-promotion-stop-red.log`: a real partial-prefix cancellation and
  abandonment control failed because the old identity-index path never entered
  sparse promotion. Compilation took 6 minutes 57 seconds; execution took
  0.01 seconds. This control observes genuine writes and cooperative yields;
  it does not inject an artificial yield.

- `ranges-packed-retention-green.log`: all four layout controls passed after
  compilation of 5 minutes 10 seconds, executing in 0.01 seconds. Dense inputs
  allocate and write zero sparse indices; inline/spill word capacities and
  credits match; sparse coordinates preserve physical IDs and source fees; real
  partial-prefix cancellation and abandonment roll back and refund on drop.

Dense segments now keep first position and length, with no
identity-index allocation. The first gap promotes only that segment's prefix in
bounded cooperative chunks, using prepaid worst-case workspace. A dropped or
cancelled promotion rolls back its uninstalled prefix. Retention uses packed
`u64` words. Mixed worker gates and post-restore gaps passed focused verification.

- `focused-controls-green.log`: 27 of 29 focused controls passed after a
  2 minute 9 second compile, executing in 1.52 seconds. Passing controls include
  actual mixed native count/fill worker gates, refusal, zero-headroom SQL,
  physical drop/retention masks, all timestamp units, counter atomicity, and
  state/match limits. Two new fixtures were wrong: Float64 was assumed to support
  native copy, and manual commits disabled the V1 checkpoint journal.
- `focused-controls-corrected-green.log`: corrected both fixtures without
  changing production behavior; both affected controls passed after a 1 minute
  49 second compile, executing in 0.23 seconds. This includes mixed selected-copy
  input/result gates, complete sliced Int8 SQL Arrow output and full-fee parity,
  V1 SQL restore, layout 1/2 managed restore, and layout 3 sparse restore followed
  by dropped physical ID gaps. Combined, all 29 selected controls passed.

The initial scoped Clippy run rejected large inline-vs-heap enum variants and
oversized nested futures. Structural refactoring keeps inline sparse positions
as `u8` within the guaranteed 64-physical-row bound, with `usize` spill indices;
fixed segment slots and optional prepaid spill retain the inline allocation
contract. Borrowed outer matching boundaries reduce repeated owner storage in
nested futures. No uncredited production future boxing or lint suppression was
introduced. Matching and SQL outer futures borrow the batch and restore returned
worker ownership in place. Large test-driver futures use `Box::pin`; these are
fixture allocations, outside the engine and its measured allocation controls.

## Final local verification

- `focused-final.log` and `focused-final.json`: all 38 selected tests passed,
  with 2,011 filtered out. Compilation took 4 minutes 18 seconds; execution took
  2.22 seconds; recorded subprocess elapsed time was 262.014580329 seconds.
  The ownership refactor
  directly affected the earlier passing controls, which justified this final
  rerun. Added controls cover metadata capacity one byte short and exact credit,
  V1 native framing and composite keys, unchanged legacy Int8/date codec
  behavior, serial/nonserial SQL retry, and empty/singleton output independence.
- `clippy-final.log`: `cargo clippy -p calc-flow --lib --tests -- -D warnings`
  passed in 1 minute 53 seconds, with no production future boxing or lint
  suppression. Workspace lint configuration is unchanged.
- `cargo fmt --all --check`, generated project schema no-drift, and whitespace
  checks passed. No project schema, fingerprint or checkpoint layout changed.
- `complexity-final.json`: all 283 changed/new functions across 32 changed Rust
  files have CCN at most eight; parser omissions and exceptions are empty.
  Sorted Rust source aggregate SHA-256:
  `dda80c61ca8560652d13399ea614ffd57cea068f259bf439be5a806e98f2b16b`.

All owned build and formatter handles are closed. Actual worker controls drain
their gather owners and verify zero final pool reservation. No native package
build, performance timing acquisition, full workspace suite or coverage matrix
was run. These local checks do not establish CI, coverage, timing/RSS acceptance
or merge readiness. Final specialist review and required green CI remain pending.

Build settings are recorded in `build-environment.json`: shared target directory
`target/issue406-join-hotpaths-build`, `CARGO_BUILD_JOBS=2`, pinned Rust 1.88.0,
default debug/test profiles, and no Rust flags, wrappers, incremental or profile
environment overrides. The test executable is
`debug/deps/calc_flow-b7e69e6a365d09b0` under that target directory.

## Reviewed delivery publication

The separate reviewed branch `feature/issue406-second-stage` remains frozen at
`bb142320`. The user explicitly authorized push, PR creation, and merge once
final review and all required CI pass. The attempted `git push --set-upstream
origin feature/issue406-second-stage` was rejected before process creation:
`approval required by policy, but AskForApproval is set to Never`.
No push, PR or new branch CI was created by that attempt. Exact publication
state is recorded in `target/issue406-integration-20261014/publication-state.json`.
PR #421 remains already merged. This tool restriction does not complete issue406.
