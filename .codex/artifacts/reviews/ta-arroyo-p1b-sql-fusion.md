## PR #564 Final Withdrawal Review: grouped SUM stream SQL fusion

**Author:** wegamekinglc | **Branch:** `feature/ta-arroyo-p1b-sql-fusion` → `main` | **Files:** 19 at the measured head | **State:** closed, unmerged

Reviewed head: `6d22a508eff3b2c18fbc43554c54296f179455f3`, tree
`f510d94802f4258f5bf1e4e0f085c9589b895475`, against main
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
The initial full review covered `2a8bdc797428ed50c58c256f90974f349fab6c6e`;
the source follow-up covered its two helper extractions. This final review
validates the frozen measurement archive and withdrawal decision. Product,
harness and schema files in the feature worktree still match the measured
head; only documentation/evidence/review archival changes remain pending.
No separate review worktree was created or removed; the caller-owned worktree
and all other agents' changes remain intact.

### Summary

P1b admits an immutable, row-local Expression followed by a conservative grouped
built-in SUM SQL candidate into the existing ready group. Unknown endpoint
schemas retain their original callback validation, and the actual SQL ingress
alias is preserved. Original operator callbacks, collectors, SQL state and
checkpoint-loan ownership are reused; withdrawn P1a computed-tail admission is
not included.

P1b fails its preregistered performance acceptance rule. The 1,000-row case is
a confirmed regression; the 64,000-row case is inconclusive. Neither case
establishes an improvement. All 30 canonical-head CI checks succeeded, but
the performance gate requires withdrawal. PR #564 is closed without merging.
No source correctness blocker or evidence-validity failure was found.

### Build and Test Results

- Rust: **Passed, focused author verification**. Directly read the saved logs:
  `target/ta-arroyo-p1b-focused-green.log` records 42 passed and one existing
  ignored release allocation diagnostic; `target/ta-arroyo-p1b-final-green.log`
  records all 18 final P1b tests passing; the final lib/tests Clippy log records
  successful completion in 2m56s with warnings denied.
- Rust follow-up: **Passed, focused author verification**. Read
  `target/ta-arroyo-p1b-codacy-test.log`: all five directly affected tests pass
  individually, with no failures. They cover grouped-SUM shape rejection, real
  None-schema/non-`input` alias admission, the actual Python export, baseline
  ASOF exact-projection grouping and rejection of unsupported proofs/topology.
  `target/ta-arroyo-p1b-codacy-clippy.log` records successful completion in
  3m03s. Formatting and diff checks are reported passing for this head.
- Rust formatting and whitespace: author reports passing final checks.
  Reviewed Markdown has no trailing whitespace, missing final newline or
  misaligned pipe table. The review's project-schema diff check is clean.
- Python: **Passed, focused author verification**. Four adapter tests and Ruff
  are reported passing. Their implementation and assertions were reviewed;
  no Python product implementation changes are present.
- Initial failures: Codacy check `128494715476` on the initial head was
  `action_required` for `candidate` (complexity 10) and
  `projection_head_schema` (complexity 9), against limit 8. Its actual API
  annotations were read during the initial review. Both source fixes are
  reviewed below; local Clippy does not establish Codacy service success.
- Final CI: root's canonical-head audit reports **30/30 successful checks** at
  `6d22a508`, recorded in the frozen evidence audit. Authoritative Codacy check
  `128499454739` succeeded. The earlier duplicate `128499309886` remaining
  in progress did not represent the canonical gate; it is superseded by this
  correction. No extra remote poll, watch or wait was performed for this
  final handoff.
- Regressions: the measured small-batch wall-time regression is confirmed by
  both maintained paired intervals. The larger case remains inconclusive.
- No product test, build or performance acquisition was rerun for this review.
  Existing evidence was read and validated with pure Python checks. Local
  full-workspace/service/cross-platform verification was not repeated.

### Blocking Issues

- **Performance acceptance — confirmed regression and no improved target**:
  the 1,000-row case is 17.37% slower by overall P50, and both rounds' paired
  interval lower bounds exceed +5%. The second case receives the actual
  `inconclusive` verdict. This fails both the required improved-target condition
  and the no-confirmed-regression condition. Green CI, correct output and
  reduced driver count do not satisfy this performance requirement. Withdraw
  this candidate; do not merge or extend this acquisition with tuning/retries.

### Style Issues

No additional style issue was found in the follow-up delta. The actual initial
Codacy failures superseded the preliminary static estimate; Option
short-circuit paths also contribute to its result. Neither rule suppression
nor lint-limit changes were introduced.

### Follow-up Source Review

`candidate` now delegates its final projection loop to `projected_sums` only
after all original table, grouping and canonical-query guards pass. The loop
still visits every projection in the same order, short-circuits on the same
`None`, and returns the same accumulated `has_sum`. Passing the existing key
slice directly replaces only the prior reference to the owned vector.

`projection_head_schema` now delegates its Expression arm to
`exact_expression_output_schema`. The helper retains the original input/output
lookups, optional-schema rejection order, exact-column-projection predicate
and cloned output schema verbatim. Rolling/ASOF arms and admission remain
unchanged. The combined production delta is 17 insertions and six deletions;
it adds no planning, I/O, row consumption or state changes.

### Test Coverage

The evidence covers real example-30 None-schema ports, the unchanged Python
export, actual non-`input` SQL alias routing, lazy initialization, every full
snapshot, empty/filtered input, multiple RecordBatches and newly created groups.
It also covers selected UDF/volatile/nonlocal rejection, wrong endpoint facts,
fan-out, actual propagated late-output markers, both edge budgets, failure
owner/accepted-output counts, full-queue cancellation and continuation, and
control FIFO before data.

Native layout 3 and retained fallback layout 4 are captured while unfused,
restored into the group and continued. Capture refusal/panic, paid capture-loan
cancellation, restore refusal and retained-entry cancellation assert ownership
and cleanup. Existing quoted-uppercase alias errors and Int64 CASE wrapping
are preserved as same-error/value comparisons, not advertised as repaired
semantics. Unsupported removed SQL layouts 1/2 are not introduced.

The checked-in `holdings_project.json` hash is independently confirmed as
`ad764ad8c07437414df54a165a86e2521eda4a6a0e9778ab3db87cb82bfc4671`.
It is semantically identical to the saved actual Python export at
`target/ta-arroyo-p1b-performance/python-stock-positions-project.json`.
The native test asserts only the middle CASE/filter→SQL pair groups, reducing
four logical nodes to three physical drivers, and validates its output.

The five focused follow-up checks retain source-equivalent evidence for the
two complexity refactors. No additional broad test matrix is requested.

### Final Performance Evidence

The adapter retains the maintained SUM builder, source/lock/vendor/native seals,
resource guard, subprocess ownership, sample validator, per-snapshot arithmetic
oracle and two-round paired interval classifier. The new Python fixture keeps
the original data content and explicitly controls seed/business IPC boundaries.

The two selected cases have 266,000 total input rows each:

- batch 1,000: 266 snapshots and 2,615,000 emitted snapshot rows;
- batch 64,000: five snapshots and 50,000 emitted snapshot rows.

These counts are internally consistent; 88 formal/warmup invocations together
produce 11,924 snapshot checks. The common Rust harness requires compact state
and validates all snapshot rows, schemas, duplicate-key absence and hashes
against its independent arithmetic map. Its tuple serialization/schema digest
matches the adapter for the fixed ASCII labels and Utf8/Utf8/Int64 nullable
output. Full descriptor/resource/status validation and the template-boundary
regression guard avoid the missing-dimension failure from P1a.

The preregistered 80 formal observations and eight warmups ran once. All use
the same sealed fixture and output volumes, CPU affinity 0/2/4/6, four Tokio
workers, fixed DataFusion p4 and disabled checkpoints. The source-required
compact budget passed on every native job. All native lifecycle, clocks,
affinity, thread-pool, identity and snapshot descriptor checks pass.

| Batch rows | Baseline P50 | Candidate P50 | Overall change | Round 1 interval  | Round 2 interval   | Maintained verdict |
|------------|--------------|---------------|----------------|-------------------|--------------------|--------------------|
| 1,000      | 143.341 ms   | 168.241 ms    | +17.37%        | [+8.36%, +20.20%] | [+15.53%, +23.52%] | regression         |
| 64,000     | 37.320 ms    | 40.617 ms     | +8.83%         | [+3.41%, +12.91%] | [+0.99%, +13.65%]  | inconclusive       |

Intervals have the maintained discrete paired-median coverage of 97.8515625%.
The larger case establishes neither a confirmed regression nor improvement or
equivalence. Lower average candidate CPU concurrency is consistent with lost
stage overlap, but no profile or causal isolation was collected. The decision
rests on the observed gate, not that explanatory inference. It applies to the
matched local WSL2 environment; unobservable host power/load limits broader
claims.

Collector-inclusive elapsed time is 71.625883 seconds. The outer controller's
first observed terminal bound is 92.286844 seconds from preflight start,
including 0.584837 seconds of preflight and observation tail. This bound is
not an additional workload timing. Both fit the 600-second budget including
setup, fixtures, warmups, arithmetic validation, startup and cleanup. Build
times of 929.630 and 1289.909 seconds are recorded separately.

All 88 distinct native PIDs have completed/natural-end status, zero remaining
tasks/errors/overflow/epochs, and `owned_process_reaped=true` receipts. The
frozen final ownership audit records `remaining_owned_processes=[]`; parent
reports controller 98791 exited and was absent from the final scan. No live
ownership probe or new acquisition was performed during this review. Audit
records one attempt, zero performance retries and no profiles.

### Archive Verification

The [performance report](../../../benchmarks/runtime-fusion/p1b-20261108.md)
links the [evidence manifest](../../../benchmarks/runtime-fusion/p1b-20261108/evidence-manifest.json)
and [complete evidence](../../../benchmarks/runtime-fusion/p1b-20261108/complete-evidence.json.gz).
The gzip archive is exactly 1,417,468 bytes, SHA-256
`365e6d55dafa002f7be1e29f70f446132621c66731eafbf49e976049c9bfd5a9`.

The final read-only audit independently performed these checks:

- Decompressed and parsed the complete JSON bundle; verified the published
  companion/archive file sizes and SHA-256 seals, four embedded JSON companions
  against their standalone files, and local report evidence links. The final
  formatting audit below extends these checks to all seven published file
  seals and all eight report evidence links.
- Verified all eight archived helper source texts against their saved hashes
  and current feature-worktree files. Ran the unchanged maintained
  `compare_builds` on the archived receipts.
- Rehashed both actual native binaries and all 1,012 baseline/1,018 candidate
  sealed source files. Matched both packet receipts to the plan, revision and
  successful build attempt. Original compiler messages show actual core and
  example artifacts with `fresh=false` on both sides.
- Revalidated every saved native row with the unchanged maintained
  `validate_sample`, including resource/lifecycle/identity and all 11,924
  snapshot schema/count/hash descriptors. The native rows attest full
  independent arithmetic validation; native execution was not repeated.
- Compared every one of 40 formal and four warmup cross-version oracle pairs,
  checked 88 distinct PIDs and reaping receipts, and confirmed 80 formal CSV
  rows. Matched each saved timing to its case/arm/round/pair observation.
- Recomputed both complete statistics objects with the unchanged maintained
  `comparison` function; every saved median, interval and verdict matches
  exactly. Checked the recorded budget, single-attempt and settled-ownership
  audit against the final summary.

### Documentation Consistency

The main plan, specification, critique, protocol and documentation index are
consistent with an internal scheduling change, preserved public API and
unchanged output/checkpoint protocol. The specification names baseline
Rolling/ASOF exact-projection chains, excluding the withdrawn P1a scope. The
final performance report and plan now record the failed gate and withdrawal;
neither presents the larger case as a confirmed regression or claims
equivalence. The performance report's tables and local evidence links pass
structural checks. Earlier reviewed Markdown formatting remains applicable;
the final review artifact has no trailing whitespace or missing final newline.
Product/harness/schema files have no diff from the measured head.

The baseline receipt was read independently from
`target/ta-arroyo-p1b-performance/baseline-release/build.json`: revision
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`, product and example actually
recompiled, elapsed 929.630 seconds, binary SHA-256
`6ae78aa5c98cbed23f9089314dd10374cdddc8d3e5931339c0f5e9fb7de45f1e`.
These values match the frozen report. The completed candidate receipt has
revision `6d22a508eff3b2c18fbc43554c54296f179455f3` and binary SHA-256
`4ceb9509fa4bd3040ff49a8d9cfa1183eb3c84741e26e65f741d9e1167a10143`.
The published evidence archives both full receipts and original compiler logs.

The review includes the prior read-only peer's full production-diff and generic
collector/lifecycle context checks, then independently reads the new proof,
changed admission, all new test modules, adapter/tests and evidence logs. No
remote review or merge action was performed.

### Final Archive Formatting Follow-up

The readable CSV's 81 CRLF line endings caused the staged whitespace check to
fail. Earlier unstaged `git diff --check` did not inspect that newly added file.
Root normalized only that companion to LF and retained the original bytes in
[samples-original.csv.gz](../../../benchmarks/runtime-fusion/p1b-20261108/samples-original.csv.gz).
This follow-up independently confirmed that decompression reproduces the
previously staged original CSV exactly, and that replacing its 81 CRLFs with
LF reproduces the new readable CSV exactly. CSV parsing yields the identical
header and all 80 data rows, with every string value unchanged.

The readable CSV is now 12,602 bytes, SHA-256
`ed1eeb0b65ebb74c1433b32ba5fe61f2a1017beada5d776385885356ebb23b08`.
The original-byte gzip is 2,544 bytes, SHA-256
`8a5f858da6101f815d9e9dab9ababb25e798176f2d2aa48bcfce67cc43b1ace2`;
its 12,683 decompressed bytes retain SHA-256
`c34977375b33c6c26b9b83a10e57314d3a5a3d56a13e943162bd0f3aacd93cc2`.
All seven current manifest file seals and the compressed/decompressed backup
metadata verify, and all eight local performance-report evidence links resolve.
The complete 1,417,468-byte evidence bundle retains its original
`365e6d55dafa002f7be1e29f70f446132621c66731eafbf49e976049c9bfd5a9`
hash. The final performance report discloses this format-only transformation.

No native validation, source/helper review or statistics computation was
repeated for this follow-up. The review artifact passes its own unstaged
whitespace and link checks. Root must restage the final archive and run
`git diff --cached --check` on that final index; the pre-restage index is not
claimed clean here. The withdrawal verdict below is unchanged.

### Verdict

**Request Changes — do not merge; withdraw P1b.** The evidence is valid and
canonical CI is green, but the candidate fails the fixed performance gate.
PR #564 is already closed unmerged, which is the correct disposition. No new
implementation revision, tuning, case expansion or timing retry is requested
for this work item. Preserve this result in the immutable archive; parent owns
archive publication and the closed-PR link. This review performed no remote
mutation or merge action.
