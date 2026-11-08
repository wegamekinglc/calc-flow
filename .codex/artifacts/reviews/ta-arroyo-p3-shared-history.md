## Branch Review: P3 same-input numeric Rows shared history

**Branch:** `feature/ta-arroyo-p3-shared-history` → `main`
**Original base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Current integration base:** `7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`
**Current reviewed head:** `ab41426feacf854a7cce25246ea7a54483b22623`
**Scope:** 15 runtime/test/fixture source files, two upstream artifacts, and a later four-file Python helper follow-up.

The initial source review preceded publication. The helper follow-up uses
published source head `bb4950f949dc8c93cca08403501bb86d1d38c06b`, tree
`5878498042869d20b542f0fae52eb3b1ae484597`, with four uncommitted helper files.
No new remote PR/check-run state was queried. This is an independent source
and helper review, not merge or full acquisition approval.
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
- Python product suites: **Not run for this review**. Product source is unchanged; the real
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

**Comment Only — source and helper follow-ups cleared; not merge-approved.** Both
original-scope findings are addressed at code identity `52807475…`, with no
additional source correctness blocker found. The prior numerical, COW and
legacy review remains applicable. The four helper updates below have no new
blocking workload, timing-order, identity or bitwise-comparison defect found.
No measured improvement is established;
performance, final committed-head required CI/Codacy and final specialist review
remain merge gates. This review edited only its own artifact and performed no
build, product test, timing acquisition, remote review submission or merge action.

### Minimal Python Harness Follow-up

This follow-up covers only `benchmarks/warm_stream.py`,
`benchmarks/test_warm_stream_p3.py`, `scripts/profile_warm_stream.py` and
`scripts/test_profile_warm_stream_p3.py`. It does not reopen the accepted
runtime review, change raw evidence/specification, or approve the future
measurement controller. The handoff JSON at
`target/ta-arroyo-p3-debug-cargo/p3-harness-handoff.json` has SHA-256
`7b742d48b3910ffe6803cad11455bf7e48f5540345536fb15c763379477f65c1`.
All four current helper size/hash seals, all nine saved log hashes and the
protocol-spec identity match that receipt. All 15 runtime/test/fixture seals
still match code identity `52807475…`.

The reviewed helper hashes are:

- `benchmarks/warm_stream.py`:
  `11e21194aeae96191f82a6051a8ba3ece2bde3b7cf8ff3adf79c3341d043e659`.
- `benchmarks/test_warm_stream_p3.py`:
  `d06f93f20bae48a0a7db7aeeeadcef8422cedf636c612f7562b13e0c4d5025ab`.
- `scripts/profile_warm_stream.py`:
  `9d1408f21977ca560d684b917ba1e8b0c30d806796cea0bf739d7bbd7efa33ff`.
- `scripts/test_profile_warm_stream_p3.py`:
  `1cdcf9471001f965b3796992cd8da6f2404240f4850ab34c1caddc1392d726dc`.

The four positive scalar CLI flags enter both compare/diagnose configuration
and the expanded start receipt. Omitting them retains the existing defaults
and matrix construction. Explicit active-entity cases use the existing product
selection, so the registered T1/C1/T2 commands add no implicit 1,024,000-row
history case. Nothing broadens the three-case protocol or sample count.

For T2, shared `_symbols` uses four-digit labels through `S9999`, preserving
numeric/lexical order; smaller old fixtures retain three digits. History
600,000 remains divisible by 10,000 entities. The segment rule admits a divisor
of one complete tick without accepting partial history or a 3,000-row segment.
`_warm_group` enqueues ten original 1,000-row DATA events, then only the final
watermark closing that tick. It drains and validates all 10,000 output rows
across returned chunks before the next tick. Thus this fixed case has exactly
600 DATA events, 60 strictly advancing complete-tick watermarks and at most one
pending warm tick. No empty output or overrun chunk is accepted.

Measured sparse append generation is unchanged: every appended row retains its
own new timestamp from `positions=sequence`, and symbol selection wraps over
the first 1,000 active entities. It does not wait for or update the other 9,000
entities, combine ten measured batches, or switch appends to history-tick time.
Existing `_entity_context` and sparse-oracle formulas are retained. Input
identity/sequence checks run on every output, and warm/terminal metrics require
observed zero late rows.

`_Float64Digest` reads the original Arrow value buffer at each chunk's logical
offset as integer bits; it never converts a valid value through Python float.
Per-column logical validity bytes and only valid 64-bit payloads feed separate
streaming hashes, canonicalized to little endian. Undefined null payload does
not participate. Schema fields/order/nullability/metadata, column positions,
total rows and valid counts are included in the versioned framed digest.
Chunk boundaries, backing offsets and metadata-map order do not affect it;
signed zero, validity and valid NaN payload changes do.

Warm output is hashed incrementally in validated row order before start returns.
Each append returns its own exact digest after the maintained timed interval.
The existing `isclose` check remains an independent direct-window reference
oracle; it does not replace cross-version bitwise equality. The controller
requires equal warm digests before sampling and equal digests for both arms
at every corresponding AB/BA pair index, even when both tolerance oracles pass.
Start-row/config guards ensure those paired results represent the same
advancing input state. Raw start/sample receipts remain in the result.

Preflight checks the effective logical streaming plan, stable-v1 profile,
20/60 or single-60 frames, minimum periods 1, lateness/value policy, and fixed
DataFusion settings. Expanded configuration, logical project SHA, Program and
plan fingerprints must match across arms. It does not compare explain text,
private storage layout, tape quota implementation or physical allocation
strategy. The previously reviewed P3 private sharing fact stays outside those
logical identities, so a valid private physical optimization is not rejected.

Saved verification evidence was read without repetition:

- `p3-harness-focused-final.log`: **51 passed, two subtests passed**.
- `p3-harness-existing-helper-final.log`: **14 passed, 36 native-runner tests
  deselected**; this is helper coverage, not a native streaming run.
- Ruff check/format, C901 limit 8, the actual complexity ratchet, whitespace
  and schema checks are reported green; the ratchet log was read. The review
  independently checked all four helper text files and the current diff for
  whitespace/newlines and confirmed schema no-drift.
- The installed-native smoke only lowered/compiled one declaration and printed
  logical identities. It started no runner or input processing and used an
  unsealed native installation. It is not release compatibility, timing or a
  final baseline/candidate comparison.

**Helper-source disposition: cleared; no additional must-fix found.** Formal
execution remains blocked on the separately recorded acquisition seams:
merge P2 and rebase P3 onto that exact latest main, incorporating its real
Worker.close/cancellation fixes rather than copying unmerged changes; seal
compatible baseline/candidate release builds and all helpers/fixtures; provide
one inclusive 600-second controller across both commands with quiet/resource
guards, owned process/RSS observations, cleanup and final seal/postscan checks.
The standalone CLI currently supplies none of that global deadline ownership.
Warm oracle/digest costs and all validation gaps must count in that budget.
Keep exactly the registered three cases and 30 advancing AB/BA pairs in fresh
process pairs per case; do not relabel them as 30 fresh-process pairs. No native
acquisition, extra case, performance retry or full-suite run occurred here.


### Main 7ac Integration Delta Review

**Disposition: source/helper integration cleared; Comment Only.** This focused
follow-up reviews the resolved main integration and two Python seams, retaining
all earlier numerical, owner/credit, SQL sorted and helper-source conclusions.
It is not final performance or merge approval. No native build, test repetition,
acquisition, remote action or CI query was performed; only this artifact changed.

At review entry the local worktree was clean at
`ab41426feacf854a7cce25246ea7a54483b22623`, tree
`42ddfe53a9e96745f998fcbaf6eb7c999156c665`, based on merged main
`7ac64990739a5d7dd8d6acecbc5e6ce2b65e5d75`. The original bb495 remote
head and older 8d source-review identities above remain historical. The latest
freeze, `target/ta-arroyo-p3-debug-cargo/p3-source-freeze-main7ac.json`, has
actual SHA-256
`b24dddfa8d5a345d00477bfbb1ba6d4db2ce2c3c274e0fbed17e1a089b545511`
and runtime code identity
`701d423975e7f20a6dcfcdb2359e3fd634681431b3bd45f17d6221c91dfc93d6`.
All 15 runtime/test/fixture entries, four reported helper entries, all 23
changed-path size/hash entries and all 13 saved log hashes independently match.
Those seals precede this review-only documentation edit.

The 25 preservation facts were independently reproduced from git blobs and
actual source: twenty complete files; the unchanged P2 `Worker`, `_warm_program`
and `expected_decay`; the prior P3 MeanRows production prefix; and the exact
first 532 specification lines containing the requirements and full frozen
performance protocol. The preservation receipt SHA-256 is
`72751e2ef988e15caeb80efb78d8e2dbfb1e67ea4df588e5922f45393fed6fe4`.
No assertion was weakened or test scope silently substituted by the merge.

The conflicts retain both `decay_pool` and `shared_rows`, initialize them in
each relevant plan/test constructor and leave existing main changes intact.
`SharedRowsShape` still requires exactly two Numeric Rows groups on the same
Float64 column with different widths; `MeanKernel` additionally requires only
Mean/Mean-Difference readouts. A Decay Scan group, including mixed decay/mean,
cannot enter sharing. Generic preparation with no sharing fact leaves P2's
already-private admitted decay owner/reference state intact. The original
history-aware ordered/ordinary stream entry, fallback input preparation,
`Arc::get_mut` private-cache update, work accounting and readout remain present.
There is no decay owner folded into a Rows tape or bypass of its quota.

P2 decay/cumulative algorithms, ordered-stream integration, codec, counters,
legacy fixture and cancellation/recovery tests are byte-identical to main.
P3 tape allocation/leases, parallel ownership, sorted SQL continuation,
per-window repair arithmetic, restore rebase and its old-writer fixture are
byte-identical to the previously cleared P3. MeanRows production code is
unchanged; its constructor merely includes P2's `decay_pool: None`. Existing
logical-budget/checkpoint identity rules, typed memory estimates and original
DataFusion reservation boundary remain applicable. The previously disclosed
fixed header/dispatch cost is still not claimed globally free.

Python plan preparation now calls the byte-identical P2 `_warm_program`.
Mean/dual declarations therefore remain the same, while decay and mixed
outputs are no longer sent to the mean-only declaration helper. Preflight
requires the exact existing Decay output and, for mixed mode, its Mean output,
frame, minimum period, profile and runtime policy; project/Program/plan hashes
and expanded requested/effective configurations still match across both arms.
It does not reject private physical allocation strategies.

`_validate_pair_digest` still requires exact warm and every corresponding
AB/BA sample digest for `rolling_mean` and `dual_sma_spread`, which cover all
three frozen P3 cases. The indicator is part of the checked case configuration;
there is no new flag allowing a P3 mean workload to relax bitwise comparison.
Both decay indicators retain P2's independent D13 numerical oracle; all their
digests remain recorded and validated, but valid recurrence rounding need not
match cross-version raw bits. The unchanged direct decay reference/tolerance
and mixed Mean check continue to run. Existing default CLI/matrix behavior and
P2 owned close/cancellation/exit-receipt handling are retained.

Saved evidence was inspected without rerunning it:

- Eight shared-history resource/legacy/real-SQL tests pass in 0.05s after a
  3m20s integrated build. Seventeen further exact P2/P3 seam selections from
  the same binary pass in 0.90s, covering typed/mixed/buffered recovery,
  refusal/cancellation, budget/work counts and Mean/generic/parallel/sorted bits.
- The small existing decay/mixed stream test has a real `KeyError:
  rolling_decay` RED, then passes after restoring `_warm_program` and its
  preflight. It checks both indicators with an existing installed runtime;
  it is not a final integrated release-build/performance witness.
- The controller's two decay acceptance cases have a real digest-gate RED.
  The focused correction selection passes nine tests; the final combined
  Python selection passes 62 tests plus two subtests. Twenty P2 helper
  controls and the existing small stream test also pass: 83 tests plus two
  subtests in these final disjoint selections.
- Scoped lib/tests warnings-denied Clippy passes in 2m27s. The actual
  complexity-ratchet log says it holds. Format, Ruff/C901, staged whitespace
  and schema checks are recorded passing. The reviewer independently checks
  project-schema no-drift against main and review whitespace; full product,
  coverage and cross-platform results are not inferred from these checks.

No new integration must-fix remains. The reviewed source may be published for
exact-head CI/Codacy and separately sealed candidate release construction.
The three-case/30-advancing-pair/600-second protocol is unchanged. Formal P3
performance is still zero; compatible baseline/candidate release provenance,
the inclusive deadline/quiet/RSS/ownership/full-seal outer adapter, measured
improvement, green required final-head checks and final specialist review
remain outstanding. This clearance authorizes no timing claim or merge.


### TARGET Three-case Controller Review

**Request Changes — one quiet-guard source gap remains before acquisition.**
This follow-up reviews only the target wrapper and pure-mock evidence; prior
runtime/helper/main-integration clearance remains applicable. No native test,
build, timing acquisition, protocol change, remote action or D1 edit occurred.

The measured clone is clean at
`8b5c13109b3d2cfc10802e91a5bacc3fd9aa3624`, tree
`4e56b71ff6d9d13ddcf6b838174132acc4ae8dbd`. The frozen wrapper
`target/ta-arroyo-p3-performance/collect-p3.py` is 20,449 bytes/609 lines,
SHA-256 `e8b311c82c769fb04d8f5ad0e5507691afd772eb5ea912939810ebf3d9986806`.
Its 150-line mock module has SHA-256
`0b15e6322148a8b99440dec1bc94b0054affedcd431215aae9b3711d7fdded9f`.
`controller-review.json` has SHA-256
`beee07520f09f16b9f0ed63e50a030632aae1dcfa91ec70ff4bcb5e2291367d2`;
both embedded source texts and declared hashes exactly match the actual files.
Both source files were read in full, with directly used shared helper context.

#### Blocking Source Finding

- **`collect-p3.py` — `quiet_guard`: supported direct-script invocations can
  evade the guard.** The recognized argv values are exact module names such as
  `scripts.profile_warm_stream`; the roots sent to `generic.competing_process`
  are only the two Cargo output trees. A competing supported command such as
  `python /<source>/scripts/profile_warm_stream.py compare ...`, or the relative
  `scripts/profile_warm_stream.py` form, matches neither rule. This runtime
  explicitly supports direct script execution, and its Python executable/site
  lies outside Cargo roots. Add a target-local recognizer covering module and
  direct-script/worker forms associated with the actual candidate/baseline
  source or worker/site identities, while retaining own-PID/owned-descendant
  exclusion. Use those real identities rather than relabeling receipt Cargo
  paths or rejecting every unrelated Python process. Add the smallest mock
  proving a direct warm competitor is rejected and owned workers are excluded.
  No shared framework, new case or native probe is requested.

#### Verified Matrix, Oracle and Ownership Boundaries

The full preregistered performance section was cross-checked. The two commands
select only the three exact configurations: T1 dual20/60 and C1 Mean60 at
history1000/segment1000/entities1/active1/append1024, then T2 dual20/60 at
history600000/segment1000/entities10000/active1000/append1000. Normal GC,
30 advancing AB/BA pairs, seed20260905, CPUs0/2, default Tokio4 and DF1/8192
remain fixed. There is no scheduler override or extra implicit matrix point.
`validate_complete` rejects missing/duplicate/substituted configurations;
source commands cannot add a fourth workload. Previously cleared T2 tick,
watermark, timestamp, label and digest implementation is unchanged.

Each actual start must acknowledge the exact expanded configuration, versioned
warm digest and logical project/Program/plan identities. The actual preflight
requires stable_v1, correct windows/minimum period and DF settings. Warm outputs
and every matching 30-pair sample retain exact Float64 digests, valid start-row
advancement and independent oracle acceptance. All three P3 indicators remain
in the strict Mean/dual branch; no D13 exception or new relaxation flag is used.
The two-sided sample/terminal shapes, normal completed native state and
maintained paired statistics are checked again from the written reports.

Exactly six distinct worker PIDs must have normal OS-exit-zero/reaped receipts
and a positive observed worker RSS sample count/peak. Missing sampling is fatal,
not zero. The thin 20 ms descendant sampler identifies the actual module
`_worker` argv, records create-time identity and sampled worker-tree peaks;
whole-command-tree peak is separately recorded. Last sampled process CPU is
correctly labelled incomplete rather than a complete lifetime total. Root's
whole-tree owner retains remaining-owner detection and termination after
collector failure; the existing Worker finally/close receipts remain unchanged.
A primary command error is preserved when cleanup also fails. No retry or
partial three-case acceptance is present.

#### Verified Deadline and Seals

`BOOT_STARTED` precedes path/identity setup and helper/psutil loading; parent
protocol and owner monitor use it across both commands. The child consumes the
parent's absolute acquire/total deadlines. Normal process return is checked
against that same acquisition clock, closing the generic final-poll race.
Cancellation gives existing Worker cleanup its reserved interval, then retained
generic settlement and the outer absolute owner enforce the hard bound.

Source/provenance/helper preparation, installation, fixture/history/oracle
work, both compare commands, reporting/bootstrap, terminal cleanup and final
seals are inside one 600-second phase, with acquisition at 530 and 70 seconds
reserved for cleanup. Successful postseals and empty owned tree are required;
final receipt writing is followed by a deadline check that invalidates late
success. The wrapper does not rely only on a self-reported success boolean.
No source/helper import, fixture or second command receives a fresh budget.

Seals reuse the maintained build/wheel/native/source revalidators and include
both source inventories with vendor/configuration files, locks/build manifests,
full common-helper union, four reported helpers, controller, ownership helper,
actual thread environment, taskset and static baseline-equivalence evidence.
The baseline proof SHA-256 is
`3968f4ad8b7ba09b8e0c6059e83b52bd17a69887159101fed8e239e55f1358e1`;
its final P2 approval reference matches SHA-256
`9b6ea9fe57287a2f5f869c26c56635d61a4938e41fbb34af9c7a68edb301e990`.
It honestly retains baseline cc6/build/native identity and invokes the reviewed
release-equivalent metadata/cfg(test)-only integration to main7ac; it claims
neither full directory equality nor an artifact newly built from main7ac.
No P2 runtime or final approval was re-reviewed in this target task.

The candidate build receipt does not yet exist at this handoff. No final core,
binding, wheel/native hash or completed candidate build is inferred. Actual
compatible release receipts, compiler/artifact freshness and root's quiet
machine confirmation remain pre-acquisition conditions after the source fix.
Planned acquisition and acquisition-outer paths are absent; formal samples and
native probes remain zero.

#### Evidence and Disposition

The saved metadata RED has two real failures (duplicate fixed case and changed
sparse physical segment); its GREEN control passes three tests. The final
pure-mock selection records 11 passing tests in 0.149s and was not rerun.
The provided `controller-ruff.log` ends with the original 52 lint findings and
predates the final formatted source. It must remain historical; retain a
successful lint/format result for the final frozen helper rather than citing
that old log as green. This is an evidence correction, not a native test/build
request or a finding that all old lint errors remain in the current source.

Only the quiet-guard source correction, its narrow mock and final helper
lint/format evidence are needed. Then freeze the updated controller/mock/review
identities and perform the actual release/quiet/resource admission. No new
case, framework, sample count or timing retry is requested. This review edits
only its artifact and passes whitespace; source-stage clearance does not
establish speedup, final required CI/coverage success or permission to merge.


### Real Candidate Release Build and TARGET Guard Closure

**Comment Only — release evidence and corrected controller are cleared for
one preregistered acquisition, subject to actual quiet/resource admission.**
The e8b quiet-guard Request Changes above is closed by the target-only delta
below. Prior source/helper/main review remains unchanged. This is not a claim
of performance improvement, final CI/coverage success or merge approval.

#### Independently Verified Terminal Build

The actual owner session 7225/owner PID23604 terminates with exit 0; builder
PID23734 reports success. Owner wall is 887.4934346489608 seconds, maintained
builder wall 886.7632291639457 seconds, and Cargo wall 883.2192111240001
seconds. The separate cache copy took 3.211518742 seconds before that owner
build interval. These are build/preparation costs outside acquisition, not
part of an already-consumed or reset 600-second measurement phase.

The candidate receipt has SHA-256
`7642c75d2cdb991e3e6314ab777822e3f72710df841e13d6d59d6b546c222a6c`.
Its source is the clean immutable 8b5c1310 clone/tree 4e56 already identified
above. The original source aggregate was independently recomputed from
1,106 entries and equals
`0657b633a58f6707019c0e53b39a6f12cd170dcffda4b98948a3771bec3ae8af`;
Cargo.lock also matches its receipt. No generated source `_native*.so` exists.

The actual Cargo stdout at `build-audit/cargo-audit/23828.stdout` has SHA-256
`4b42fa0b200d7980527e2ab29cdce6f9be3da9b6f218ce8eefa76e1361b0413d`.
It contains exactly one core `calc_flow` and one binding `calc_flow_python`
compiler artifact, both `fresh=false`, followed by one successful
`build-finished`. Each package/manifest/src path names the actual immutable
candidate checkout. Both are optimization level 3, test=false, no debug
assertions or debug info. Core features are empty; binding features are
`connector-file`/`default`. No diagnostic test feature or reused stale core/
binding artifact is substituted for actual release compilation.

The original Cargo journal has SHA-256
`3fb7e3eae8df808f6e8f45c096edf96e00f8d5a29d1b0a11bcecfd7155851614`.
Its command uses the real Cargo binary with release/locked/abi3-py313 flags,
the exact binding manifest and Maturin's strip-symbols flag. The inspected
launcher forwards original stdout bytes and argv without replacing Cargo or
inventing compiler events. Captured environment has CARGO_TARGET_DIR isolated,
CARGO_BUILD_JOBS2/incremental0 and no Rust/profile/toolchain override. Maturin
finds the managed CPython3.13 interpreter; the maintained builder supplies that
interpreter to PyO3. Build command/profile, rustc, Python/interpreter and lock
match the retained baseline after ignoring only the output directory.

The 38,642,361-byte wheel independently hashes to
`24b8bee2141a8159fc8eac7422d80fb1bf822c91b557673b02b48df9b1bd0ec3`.
Its only native member, `calc_flow/_native.abi3.so`, is 108,396,664 bytes and
hashes to
`6b599135f830eb3b88f7bb7c4ac9162f2ca2d8e2902d6eeec05cad987c835bfc`.
Those native bytes exactly equal the actual compiled binding library. The
outer-build copied compiler records exactly match the original Cargo events;
the final audit SHA-256 is
`90cedb63302bb5c6d705377f3b39b66c2077574f742b768282512521dca7291c`.
This closes the formerly pending candidate receipt/freshness/hash evidence.
The baseline remains its actual reviewed cc6 wheel/native, under the existing
release-equivalence proof; no artifact is renamed main7ac or candidate8b.

#### Focused Quiet-guard Correction

The corrected wrapper SHA-256 is
`cda80106c96e2020ccca50c02d0f27a4138d5c4f73045c5bf77017912d6bfc74`;
mock SHA-256 is
`dfe062d559fb7db7d3a58248da918648d9306c5c831e2f9f61fa4df027b47464`;
review JSON SHA-256 is
`e315095a334a2e0fa34faf7a139ba6e1ef05e19d42f76b295a36ca57b02b20fc`.
Both embedded full source texts/hash declarations match. The archived e8b
wrapper matches its original seal, and the diff is limited to quiet-guard
identity/root recognition and removing historical resource-PID exclusions.
Matrix, sample/oracle/digest gates, budgets, cancellation and receipt handling
are unchanged.

`measurement_roots` obtains the actual baseline source/wheel directory,
immutable candidate source, actual target/site and baseline Cargo roots, and
managed venv context. Known module and direct absolute/relative script forms
are recognized. A private `_worker` using the relevant source, temporary site,
PYTHONPATH/cwd/interpreter or VIRTUAL_ENV context is also recognized. Genuine
own current compare root/descendants remain excluded. Historical sampled PID
keys are no longer broadly exempted, so an old resource entry cannot hide an
unrelated current process.

The saved quiet RED has three real assertion failures for absolute direct
worker, relative script compare and worker site/PYTHONPATH forms; its owned
control remains passing. The focused quiet GREEN passes five tests. The final
combined pure-mock selection passes sixteen tests in 0.094s, including three
source/venv context subtests. No native workload or timing probe was launched.
`controller-ruff-final.log` now explicitly records `All checks passed!` and
`2 files already formatted`; the old 52-error log is correctly labelled
historical. No passed tests/lint were rerun by this reviewer.

No target-source, release-provenance or supplied-evidence must-fix remains.
Planned acquisition/outer paths remain absent. Root may now confirm the actual
quiet/resources and execute the original two commands exactly once: three
fixed workloads, thirty advancing paired appends each, six distinct fresh
workers, normal GC, exact P3 bitwise results and the one 600-second phase with
530-second acquisition/70-second cleanup reserve. Preserve partial evidence
and stop on failure/noise/drift; there is no extra probe, case, retry or budget
reset. All hashes/source/worker identities and owned cleanup must still be
observed again by the acquisition's real before/after checks. Performance and
final required CI/final specialist review remain merge gates. Only this review
artifact changed; whitespace passes. No source/controller edit, build, native
execution, acquisition or remote action was performed here.
