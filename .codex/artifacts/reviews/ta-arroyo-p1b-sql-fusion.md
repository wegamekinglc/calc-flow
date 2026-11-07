## PR #564 Review: fuse expression stages with grouped SUM stream SQL

**Author:** wegamekinglc | **Branch:** `feature/ta-arroyo-p1b-sql-fusion` → `main` | **Files:** 18 before this report

Reviewed head: `2a8bdc797428ed50c58c256f90974f349fab6c6e`, tree
`747b4f3d38f9544b61950b5db03d085606eea8bb`, against main
`8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
The supplied isolated worktree became this published head during review.
No separate review worktree was created or removed; the caller-owned worktree
and all other agents' changes remain intact.

### Summary

P1b admits an immutable, row-local Expression followed by a conservative grouped
built-in SUM SQL candidate into the existing ready group. Unknown endpoint
schemas retain their original callback validation, and the actual SQL ingress
alias is preserved. Original operator callbacks, collectors, SQL state and
checkpoint-loan ownership are reused; withdrawn P1a computed-tail admission is
not included.

No additional source correctness blocker was found. Actual Codacy has two
blocking complexity findings, and performance acceptance and remaining CI are
not yet established.

### Build and Test Results

- Rust: **Passed, focused author verification**. Directly read the saved logs:
  `target/ta-arroyo-p1b-focused-green.log` records 42 passed and one existing
  ignored release allocation diagnostic; `target/ta-arroyo-p1b-final-green.log`
  records all 18 final P1b tests passing; the final lib/tests Clippy log records
  successful completion in 2m56s with warnings denied.
- Rust formatting and whitespace: author reports passing final checks.
  Reviewed Markdown has no trailing whitespace, missing final newline or
  misaligned pipe table. The review's project-schema diff check is clean.
- Python: **Passed, focused author verification**. Four adapter tests and Ruff
  are reported passing. Their implementation and assertions were reviewed;
  no Python product implementation changes are present.
- New failures: Codacy check `128494715476` is `action_required`, with the two
  actual findings below. The API annotations were read directly in the single
  review-handoff snapshot.
- Regressions: not performance measured. Fused/unfused correctness tests pass;
  this is not a throughput or latency equivalence claim.
- No test, build or performance acquisition was rerun for this review. Full
  workspace, service coverage and cross-platform results are not claimed.

### Blocking Issues

Must be resolved before merge:

- **`crates/calc-flow/src/operator/sql/fusion.rs` — `candidate`**: Codacy reports
  cyclomatic complexity 10 against limit 8. Extract syntax/SELECT parsing or a
  similarly narrow proof stage into a helper while preserving all existing
  rejection cases. Do not suppress the rule or merely rename the function.
- **`crates/calc-flow/src/runtime/streaming/runner/operator_fusion.rs` —
  `projection_head_schema`**: Codacy reports complexity 9 against limit 8.
  Extract the Expression-specific optional-schema/projection proof, preserving
  the original exact-projection semantics and early rejection behavior.
- **Performance and CI gates**: neither candidate acquisition nor its real
  `improved` verdict is available at this head. Candidate sealing, complete
  preregistered evidence, required CI/Codacy and a final updated-head review are
  still necessary. A smaller driver count is not performance evidence.

### Style Issues

The actual Codacy failures supersede the preliminary static complexity
estimate; Option short-circuit paths also contribute to its result.
Local Clippy success does not waive these findings.

Two documentation corrections are needed before the final handoff:

- **`benchmarks/runtime-fusion/p1b-20261108.md` — Preparation checks**: its
  opening status now correctly records the started baseline build, but its
  final paragraph still says no product build has run. Update that remaining
  sentence to distinguish the running baseline from the unbuilt candidate and
  unstarted measurement.
- **P1b specification — Focused acceptance**: replace the remaining phrase
  “Existing P1a chains remain admitted” with the baseline exact-projection
  chains. The implementation correctly excludes withdrawn P1a computed-tail
  admission; the checklist should state the same scope.

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

Final source-equivalent evidence must be retained after the two complexity
refactors; run only the directly affected checks. No additional broad test
matrix is requested.

### Performance Evidence Contract

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

Both target cases must retain their identical fixture, full output volume,
affinity, four Tokio workers, fixed p4 and disabled checkpoints across builds.
Build time is separate; all fixture/setup/oracle/warmup/acquisition/cleanup work
shares 600 seconds. No timing, full fixture generation, changed case, extra
sample or favorable retry was executed during this review. Actual native
compact activation, machine topology, final source/binary/helper hashes and
clean lifecycle witnesses remain acquisition-time requirements.

### Documentation Consistency

The main plan, specification, critique, protocol and documentation index are
consistent with an internal scheduling change, preserved public API and
unchanged output/checkpoint protocol, subject to the two wording corrections.
All five reviewed Markdown files pass structural table-width, whitespace and
final-newline checks. Project-v3 schema has no diff.

The review includes the prior read-only peer's full production-diff and generic
collector/lifecycle context checks, then independently reads the new proof,
changed admission, all new test modules, adapter/tests and evidence logs. No
remote review or merge action was performed.

### Verdict

**Comment Only — not merge-approved.** No additional functional or measurement
contract blocker was found, but both actual Codacy findings must be fixed and
real performance plus required CI must pass. Re-review the final source and
evidence before any merge.
