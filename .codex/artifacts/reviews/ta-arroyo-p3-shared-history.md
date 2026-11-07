## Branch Review: P3 same-input numeric Rows shared history

**Branch:** `feature/ta-arroyo-p3-shared-history` → `main`
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Scope:** uncommitted source; 15 runtime/test/fixture files and two upstream artifacts before this report.

There is no P3 PR, published candidate head, check-run result or GitHub review
to consult. This is an independent source-stage review, not merge approval.
The supplied worktree contains no withdrawn P1a/P1b or unmerged P2 runtime
changes. No additional review worktree was created or removed.

The reviewed freeze JSON is
`target/ta-arroyo-p3-debug-cargo/p3-source-freeze.json`; its actual SHA-256 is
`ed2e0fa7bc340a682216c791174de2ac799437442fbd8467f1436a9c783d31d1`.
Its `code_manifest_sha256` identifier is
`97cd2b97fd269ecea161ed91ce6bd78271624b36bdad2f2ee772a058e96fcc6d`.
All 11 listed file sizes and hashes were independently checked for the initial
review. The focused follow-up is sealed by
`target/ta-arroyo-p3-debug-cargo/p3-source-freeze-review-followup.json`, whose
actual SHA-256 is
`5907a767e4c6251f8e07096581b0ec074a4a7cdbf93b989189158d04d40e6731`.
Its code identifier is
`52807475a37fa96781260c578dc089f5a3abe22c8109bfb2caf02786259570bd`.
All 15 current file sizes/hashes match. Nine prior files, including the numeric
tape implementation, Mean/parallel arithmetic and pinned fixture, are
unchanged. The follow-up reads only the six changed/added files and their
direct call/ownership context; the prior core review is retained.

### Summary

P3 optionally replaces two same-column Float64 numeric Rows queues with one
credited tape while retaining each window's original accumulator and repair
sequence. It preserves the public API, logical history budget, kernel
fingerprint and checkpoint encoding. Quota refusal retains the existing
independent-queue representation.

No numerical, ownership or checkpoint implementation defect was found. Both
source-review findings are now closed: the reachable SQL sorted dual-AVG path
uses the shared owner/views within the original scope, and positive-credit
serial refusal/cancellation has direct integration coverage. Source-stage
review is cleared. Real performance and final-head CI are still pending.

### Build and Test Results

- Rust: **Passed, focused author evidence read**. No native test or build was
  repeated by this reviewer. `p3-final-cow-resource-tests.log` records all four
  final numeric/resource tests passing after a 2m45s build; execution took
  0.02s. They cover independent accumulator bits, borrowed restore rebase,
  actual COW allocations/credit and refusal fallback.
- `p3-final-credit-cold-generic-tests.log` records three passed tests in 0.40s:
  positive-credit restore/budget rejection, the cold allocator regression and
  the generic-versus-compiled property. `p3-legacy-compatibility.log` records
  the pinned old-writer comparison passing in 0.03s.
- The initial parallel batch was **48 passed, one failed**. Its unique failure
  was the cold process-global pool allocation inside an owned-merge probe.
  The final fixture explicitly initializes that infrastructure outside the
  probe; the identical zero-live-allocation assertions remain and the test
  passed in the final isolated selection. A new full 49-test run is not claimed.
- An earlier 16-test focused batch was **15 passed, one failed**; the restore
  test had installed its measuring pool after initialization. The corrected
  final test installs the pool first, requires positive initial credit, and
  passes in the three-test selection above. These old failures are resolved
  evidence, not silently reported as passing batches.
- Scoped lib/tests Clippy with `benchmark-internals` and `-D warnings` passed
  in 2m18s in `p3-scoped-clippy-final.log`. Formatting is reported passing;
  the reviewer independently checked whitespace and project-schema no-drift.
  Parent reports the actual complexity-ratchet script passes; no new remote
  Codacy result exists yet.
- Follow-up evidence: `p3-review-followup-red.log` records the actual SQL
  sorted storage RED, with the expected "still owns two queues" assertion.
  The independent serial-credit coverage test already passed and is not
  represented as a behavior RED. `p3-review-followup-green.log` records all
  five new tests passing in 0.04s after a 4m09s build.
- `p3-review-controls.log` records six directly affected controls passing in
  0.03s, including original sorted single/dictionary/empty continuation, the
  four numerical/COW/credit/rebase tests and pinned old-writer comparison.
  `p3-review-sql-route-control.log` records the separate COUNT fingerprint
  control passing. `p3-review-scoped-clippy.log` records warnings-denied scoped
  lib/tests Clippy success in 3m20s. Parent reports actual ratchet and format
  checks pass; the reviewer did not rerun them.
- Python: **Not run for this review**. Python source is unchanged; the real
  dual-SMA declaration/lowering chain was traced statically below. Native
  Python execution and the focused finance suites are not claimed passed.
- Performance: **Not run**. No P3 acquisition protocol, sealed release pair or
  measured speedup is available. Full workspace/service coverage and
  cross-platform results remain required CI responsibilities.

### Blocking Issues

No unresolved source-stage blocker remains. The two original findings and
their focused resolution are recorded below; the SQL sorted admission belongs
to the original batch scope and was implemented rather than excluded.

- **Merge gates remain pending**: no final committed-head required CI/Codacy/coverage result
  or preregistered end-to-end improvement exists. A smaller tape, fewer copied
  bytes or allocation-only success cannot satisfy the user's merge condition.
  Retain the maintained sample/statistics/provenance/resource contract and
  total 600-second acquisition budget when that separate task is authorized.

### Style Issues

No new style or dependency exception was found. Necessary mutation is confined
to private entity candidates. The two `Arc::get_mut(...).expect(...)` sites
express the private-owner invariant established before the row loop; expected
quota refusal follows the owned fallback and does not panic. No unsafe block,
lint suppression, workspace lint change or public compatibility shim was added.

The first specification wording confused the manifest's embedded code identity
with the JSON file's own digest. Parent corrected that documentation during
review; the runtime/fixture seals did not change.

### Numerical and Execution Review

`RollingKernelPlan::compile_with_order` requires the actual input field to be
Float64. `MeanKernel::compile` first proves the exclusive one/two-output
Float64 Mean/Mean-Difference family; `SharedRowsShape::compile` then requires
exactly two Numeric Rows groups on the same column with different widths.
Float32 casts, a single group, same-frame groups, other columns, mixed
statistics, Duration, exact integers, Pair, extrema, EWMA/cumulative and Decay
do not acquire this sharing shape. Logical descriptors and their order remain
unchanged.

For a private tape after append, `len - (W + 1)` identifies that window's
outgoing original-row position. None/filtered NaN entries still occupy tape
positions. Each group calls the unchanged `add_float64`, then
`remove_float64`, then its repair before the next group. Repair receives
`min(W, tape.len())` and an oldest-first borrowed suffix. Trimming waits until
all groups finish, so the largest window's outgoing sample remains available.
There is no new lifetime ordinal or earlier counter-overflow condition.

The new repair code preserves sticky NaN-as-value behavior, per-window
stable-v2 cadence and the original non-finite/M2 trigger. Stable-v1 refold and
stable-v2 fold use the same value order and existing arithmetic functions.
No sum/count replacement, removed M2, FMA or changed tolerance was introduced.
The independent-state tests cover both profiles, null/NaN gaps, NaN-as-value,
signed zeros, infinities, alternating extremes, M2 overflow and near-overflow
finite values across 300 transitions and 2/4, 5/20 and 64/80 windows.

All routes that create shared owners have corresponding readers:

- MeanRows makes each touched entity private before its row loop and uses a
  batch-selected const dispatch. A quota-refused entity executes the original
  owned-group loop.
- Generic fill prepares eligible touched entities before updating, including
  output-cleared replay; the shared update reads the same source column once.
- Parallel preparation contains only touched entities, reserves one private
  tape per candidate, and retains the prior Rows5/20 eligibility. Lane entry
  reuses exclusive owners. Cancellation hooks, row/group/error tags and
  deterministic error arbitration remain at their original sites. A stop
  after append discards the private candidate.
- Forced stable-v2 restore rebase reads each group's suffix without copying or
  mutating its shared tape. The plan's private fact survives `outputs.clear()`;
  the exclusive two-finite-group admission means replay does not reindex it.

### Ownership and Budget Review

`RowsTape` reserves the bounded max-width-plus-slide capacity before payload
allocation and reconciles to actual capacity. Each owner includes tape/Arc
header and a conservative registration charge in its private 64 MiB
process-local Greedy pool. It does not grow during the row loop. Warm COW
reserves another owner before copying the tape once; the old owner remains
credited while resident/candidate/snapshot aliases hold it.

`MemoryReservation::new_empty` shares only the registration/consumer. Its new
size starts at zero, and each reservation independently frees its own bytes;
no old reservation is split or transferred. This was checked against the local
DataFusion 55 implementation. The resource test observes the same consumer,
two actual COW allocations, two simultaneous credits, an allocation-free first
slide, unchanged old rows, and release only after each last owner drops.

Refusal materializes each proper Rows suffix with private time placeholders
and leaves every accumulator/count untouched. Only Rows readers can see those
placeholders, and durable encoding still uses original retained input history.
An unshared warm entity is not refolded or globally rebuilt to retry admission.
The optional pool does not account for legacy fallback queues or process RSS;
it is not advertised as doing so.

Logical charge still uses retained history, buffered records and unchanged
`WindowState`/`WindowAccumulator` layouts. The new optional typed-owner header
does not enter `StateBudget`. Historical nonsharing queue-estimate formulas
remain; shared estimates add the owner's actual reserved bytes. The fixed
entity header increases, and generic fill adds a per-row storage-dispatch
branch. The Mean single-window const dispatch avoids per-row tape work. These
costs must remain visible in the controls; zero global overhead is not proved.

Prepared output owns candidate states until the existing emit/commit boundary.
Error/cancellation unwinds those private values and leaves installed tape
owners intact. The new positive-credit serial integration test directly
observes that combined resource effect, as detailed below.

### Python and SQL Route Boundaries

The actual warm tool calls
`benchmarks/rolling_indicator_comparison.py::_native_program`, which declares
Float64 `price` and `ts.mean(...Rows5) - ts.mean(...Rows20)` in one feature.
`lower/segments.py::_fused_difference_outputs` recognizes the two rolling
leaves; `lower/planners.py::_plan_rolling_stage` hides their intermediate columns
and emits a single `difference` declaration. Rust `compile_aggregate` keeps
two groups because the widths differ, and the typed output is a Mean
Difference. This statically reaches P3 without changing the Python graph,
adding endpoint schemas or running native code.

SQL sorted execution is another admitted boundary. DataFusion 55 groups windows by
partition/order keys, not by frame. Pure AVG Rows5 and Rows20 over the same
column/order can therefore reach `DataFusionRollingKernel::compile` as two
groups/two Mean outputs. This reachable original-scope gap was confirmed by
the focused RED and is now fixed. `sorted::fill_sorted_ranges` prepares the
shared owner once per nonempty partition range before any row transition,
then enters the existing shared group updater with unchanged group ordering,
per-window repair, final trim and readouts. Empty batches retain their last
state without mutation; completed partitions drop their current owner
reference before admitting the next tape. A held alias retains its own credit.

The actual finite SQL test selects two AVG frames into one rewritten sorted
node, feeds multiple physical RecordBatches and two partitions, and compares
identity plus finite-integer mean bits against rewrite-disabled DataFusion.
It proves that the real query planner/physical stream reaches the adapter,
not merely a manually assembled plan. The original Rolling batch/stream
routes remain admitted as reviewed above. Count-compatible SQL continues
through the separate `SqlRollingKernel` and does not gain a shared tape.
The Python symbolic and SQL routes are distinct acceptance paths; their future
performance evidence must identify which one was actually measured.

### Focused Follow-up: Sorted Ownership and Serial Credit

The production DataFusion boundary now moves `RollingStream.state` into the
crate-private consuming transition inside `CalcFlowRollingExec::execute`'s
`try_unfold`. This state belongs to one physical execution stream; it is not
the managed operator's durable/recoverable state. An error terminates that
physical stream and drops its moved state/reservation; no public retry or
restore promise requires preserving a consumed private value.

A continuing unique partition keeps its tape allocation instead of forcing a
deep copy each batch. An explicit clone has independently cloned group
summaries and an Arc alias of the tape; preparation therefore obtains a
separate credited owner or materializes private owned suffixes on refusal.
It cannot mutate the held alias. The borrowed reference/test entry is
`cfg(test)` and clones before calling the consuming path. It retains the
transactional reference semantics used by existing tests; no public caller's
borrowed state is passed directly into the mutation loop.

The existing DataFusion `reservation.try_resize(transition.metrics.state_bytes)`
remains before output publication and state continuation. Shared entity metrics
include empty group-queue headers and exactly one owner reservation, plus the
prior key estimate; there is no per-readout tape charge or skipped reservation.
The zero-byte DataFusion pool still rejects the real query and returns to zero;
this test does not claim that the refusal necessarily occurs at this operator
rather than an upstream allocator. The source call itself is unchanged.
COUNT's separate update continues to borrow its state normally.

New sorted tests compare NaN/partition and initial private-quota refusal against
the original independent-queue reference. They observe stable owner identity
on unique continuation, separate credit with an unchanged held alias, quota
refusal materialization with exact outputs, and release at each last drop.
The shared numerical/repair implementation itself is unchanged from the prior
review.

`p3_serial_dual_rows_refusal_and_cancel_release_candidate_credit_before_replay`
installs a measuring pool before creating the actual Rows5/20 operator, warms
it, requires positive resident credit and holds a real old typed-state alias.
At the first output `emit`, ordinary collector refusal and an explicit
`CalcFlowError::Cancelled` are exercised separately. Encoded state, typed bits,
tape address, logical charge, watermark and output sequence remain identical;
candidate credit returns exactly to the resident amount. Explicit direct-
operator replay succeeds with the exact next means, both old/new credits stay
held while the alias lives, and alias/target drop returns credit to zero.
This is a direct callback/result-boundary test; it does not promise managed
job retry after cancellation or claim to exercise dropping a suspended emit
future. Its coverage closes the specific requested credit/refusal gap.

### Checkpoint and Documentation Consistency

The codec and logical hash input are unchanged; the new plan fact and tape
are private caches. The pinned fixture hash is
`b14aae9b7b9d65e1e01d0d144e481bfad7a7feafaaf3111991f4364493888825`.
Its two-profile scenario has two entities, 164 rows in two RecordBatches,
four pending rows at epoch 1, and two 72-row/two-output continuations per
profile. Candidate tests compare complete old writer bytes/metadata/checksums,
logical charge and each uninterrupted/restored route bit-for-bit. Since the
old runtime itself restores those exact bytes, equal candidate writer bytes
retain that old-reader acceptance for the fixed fixture.

The zero-context generator patch is test-only, SHA-256
`50588772b47dbbb70e557e459a63adadc720f0ee888b9138938daa353f0939ec`.
Provenance retains its original context-form digest and old-source/binary
identity. This presentation change leaves the fixture unchanged. The final
positive-credit restore test checks refusal leaves the existing typed cache,
logical charge and credit intact, wrong profile fails closed, and final drop
returns the measuring pool to zero. Existing unsupported schema/configuration
and layout contracts were not bypassed.

Specification, critique and fixture-document links, whitespace and final
newlines pass structural checks. `git diff --check` and the project-v3
schema no-drift check pass. The updated implementation record accurately
includes the SQL boundary above. No public API/documented numerical contract
or guidance definition change is needed. Native Python tests and performance
remain unrun.

### Verdict

**Comment Only — source-stage follow-up cleared; not merge-approved.** Both
original-scope findings are addressed at code identity `52807475…`, with no
additional source correctness blocker found. The prior numerical, COW and
legacy review remains applicable. P3 still has no PR or measured improvement;
performance, final committed-head required CI/Codacy and final specialist review
remain merge gates. This review edited only its own artifact and performed no
build, product test, timing acquisition, remote review submission or merge action.
