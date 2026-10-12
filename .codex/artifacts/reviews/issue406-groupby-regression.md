## PR #412 Review: group_by diagnosis and scoped repair

**Branch:** `fix/issue406-groupby-regression` → `main`
**Base:** `e538099228b54882fe3b9fcc72d802e9998221ed`
**Reviewed:** 2026-10-12, diagnostic implementation and completed-run evidence,
followed by the window scratch-map and test-harness repair on
`86d0a3bdad39deb0fa5b736d95f0802cf8efb2a5` plus the reviewed worktree diff.
The repair touches two Rust files and its specification, analysis and review.
The subsequent review covers only test comments on
`6978f3bf773d76f77d60a4d181e2ad88596438d1` and the narrow static-analysis exception
documented below.
PR #412 remains draft according to the implementer's handoff. No new PR check
snapshot or prior GitHub reviews were requested for this incremental review.
The reviewer read the changed files and existing verification evidence without
running tests, native builds or benchmarks.

### Summary

This adds a manual diagnostic for the original October 12 wheels: A/A, B/B and
A/B comparisons followed by one address profile per native module. It retains
the original workload, paired classification and common harness, with a fixed
600-second collection budget and cancellation at 550 seconds.

The original diagnostic is complete and its results remain inconclusive for A/B.
The subsequent user-authorized repair uses the existing randomized AHash for
four temporary window lookup maps and makes the sensitive-log test reuse its
current executable. The repeated 10M streaming `group_by` regression remains
unresolved pending the separately authorized complete benchmark and required CI.

### Build and Test Results

- Rust: **Passed, focused local verification only**. The reviewer read
  `window-before.log` and `window-after.log`: both selections pass 35 tests with
  one pre-existing ignored soak, in 0.28 and 0.31 seconds. The implementer
  identified the after-run executable as the rebuilt all-feature
  `calc_flow-00fd9a3140990999`; later source changes affect only the child harness.
- Same-executable child regression: **RED then GREEN**. The RED log proves the
  old nested Cargo command launched `calc_flow-242dcf7e84a09783` instead of its
  parent's all-feature executable; the assertion failed after 181.80 seconds.
  `privacy-same-binary-green2.log` records one passing test in 0.01 seconds after
  a separate 71-second build. All original privacy assertions remain present.
- Rust Clippy: **Passed**. `repair-clippy.log` records completion in 1m35s for
  `cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings`.
  The implementer reports `cargo fmt --all --check` passed; its inspected
  `repair-format-final.log` is empty. No new dependency or feature is required.
- Comment-only follow-up: **Reviewed without rerunning checks**. The retained
  `semgrep-current-exe-before.json` and `semgrep-current-exe-after.json` both
  report Semgrep 1.180.0 and no scan errors. The original rule reports one
  informational `current-exe` finding before the annotations and zero afterward.
  This proves the supplied rule's local suppression, not a new hosted Codacy
  result. The reviewer inspected the rule, both result files and the exact diff.
- Python diagnostic: **Passed** in focused verification. The reviewer read
  `focused.log`: nine checks passed in 0.044 seconds, and `imports.log`: four
  import/dependency checks passed in 0.279 seconds.
- The implementer's combined selection recorded 52 checks: 49 passed and three
  subprocess import checks timed out in the local PID sandbox. These timeouts
  were not counted as passes. The four import checks subsequently passed outside
  PID isolation; this environmental explanation and the combined result are
  recorded in `validation-notes.md`.
- Ruff check and format: **Passed** according to the inspected `ruff.log` and
  `format.log`; the format check reports ten files already formatted.
- Workflow validation: both YAML files parsed according to the implementation
  evidence. The focused tests execute the actual shell wrapper with harmless
  fixtures, verifying exit 42, both output streams and permission restoration.
  Actionlint was unavailable and was not run.
- Studio backend and frontend: **Not touched**; suites not run.
- New unresolved local failures: **None identified** in the reviewed repair.
- Required CI: **Unresolved**. The retained Windows log for `86d0a3bd` has
  2,029 passed, four failed and seven ignored Rust tests. The failures concern
  writer-v2 recovery, queued launch cancellation, SQL queue draining and
  checkpoint-soak cancellation. Reusing the current executable removes nested
  native compilation and a demonstrated feature mismatch; it does not prove
  that either caused all four timeouts. A new required Windows result must
  establish that they pass in the original parallel configuration.
- Existing regression: **Unresolved**. The original-wheel recovery again reports
  +10.097% for 10M streaming `group_by`; this review does not turn it green.
- Completed diagnostic: **Collection succeeded; A/B inconclusive**. The reviewer
  read the retained artifacts from run `38209692076`, artifact `11728218142`.
  No tests, native builds, benchmarks, workflow validation or CI queries were
  started for this incremental evidence review. The existing
  `workflow-yaml.log` records both workflow files parsed and every step having
  `run` or `uses`; the implementer also reported a passing whitespace check.

The new Rust validation files are under the main checkout's
`target/issue406-morning-benchmark-failure/`. The reviewer read existing logs
and source; no Rust execution, benchmark or CI query was started by the reviewer.
The implementer reports no generated-contract or `Cargo.lock` drift and a
passing final `git diff --check` for this repair.

Earlier diagnostic validation files are under this worktree's
`target/issue406-local-validation/`. The focused implementation checks were:

```bash
python -m unittest scripts.test_issue406_groupby_diagnostic
python -m unittest scripts.test_groupby_diagnostic_imports
ruff check scripts/groupby_diagnostic scripts/test_issue406_groupby_diagnostic.py scripts/test_benchmark_workflow.py
ruff format --check scripts/groupby_diagnostic scripts/test_issue406_groupby_diagnostic.py scripts/test_benchmark_workflow.py
```

The reviewer independently checked:

```bash
git diff --check -- scripts .github/workflows .codex/artifacts
git diff --exit-code -- schemas/project-v3.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts Cargo.lock
```

Changed diagnostic Markdown alignment, trailing whitespace and final newlines
passed the earlier review. Full local product suites, Rust coverage and Studio
coverage were not run; no coverage gate or final CI result is established here.

### Blocking Issues

No source findings remain in the reviewed implementation. Required CI failures
and the unresolved performance acceptance gate still block merge; they do not
block committing this repair and running its one authorized complete suite.
The following diagnostic findings were addressed:

- `AuditWorker.start` now owns subprocess creation through cancellation and
  settles a process created during that cancellation before propagating it.
- `AuditWorker.close` shields an owned cleanup task, then awaits settlement before
  propagating cancellation or recording the final process exit. The focused
  termination test verifies exit `-15`, the exit journal and closed log handle.
- The workflow mode assertion includes the new manual mode. The new test module
  is registered once in the existing step with installed benchmark dependencies,
  avoiding a `psutil` dependency in the standard-library preparation step.
- Artifact exclusion now removes installed package files while retaining wheel
  installation command journals and failure logs.
- `validate_trace` now requires a nonzero hexadecimal address in a native DSO
  frame, rejecting path-only warnings and zero addresses as well as lost,
  throttled and non-native traces.

The later `current-exe` finding is an accepted, localized false positive in
`sensitive_lifecycle_panic_payloads_never_reach_process_stderr`. Both calls are
inside `#[cfg(test)]`: the parent reenters its ordinary test binary at the same
privilege, and the child checks executable identity as a regression assertion.
Neither call makes an authorization decision or crosses a privilege boundary.
The parent call now explains this and both call lines carry `// nosemgrep`.
There is no rule modification, file exclusion or suppression of product code.
The generic annotations apply to those two lines only; this approval does not
extend to a privileged runner or future authorization use of the path.

### Style Issues

None remain in the reviewed files. The specification's table alignment was
corrected. The implementation reuses existing subprocess, measurement and
classification helpers rather than adding a general measurement framework.
The incremental analysis tables also align in the supplied formatting evidence.

### Test Coverage

The repair retains the existing window tests for null, full-width integer,
composite and string keys; colliding slot-cache entries; hopping windows;
stable checkpoint encoding and restored continuation; floating-point state;
budget refusal; and output ordering. The four maps supply lookup and insertion,
while first occurrence assigns IDs and the existing ordered installation supplies
observable order. No assertion merely mirroring the chosen hasher is needed.

The sensitive-log regression adds an exact executable identity assertion while
preserving commit, abort, recovery, status and panic redaction checks. Its child
is launched inside an awaited blocking task, so captured process I/O does not
block the Tokio executor. The task owns the child until output collection ends;
this test adds no timeout, detached process or public cancellation interface.

The focused tests cover fixed products/workload/counts, alternate-product
rejection, wrong-case refusal before dispatch, trace loss and address validation,
bad-host preflight, denied profiling permissions, process-creation cancellation,
termination cancellation and shell failure evidence/permission restoration.
Existing diagnostic tests retain alternating pair order, input-hash comparison,
environment validation, failure evidence and the old DAL-301 contract.

The completed diagnostic exercised the GitHub host, installed perf tool,
permissions and native address capture. This establishes collection on that host;
it does not establish comparable CPU hardware for the earlier failed runs or
prove that the observed sampling addresses explain their regression.

### Documentation Consistency

The repair specification explicitly supersedes the completed diagnostic's
no-full-suite restriction under the user's new request. It permits one complete
`benchmark-suite.yml` dispatch: main
`e538099228b54882fe3b9fcc72d802e9998221ed` against the immutable reviewed repair
commit, with ordinary locked release wheels and all 22 maintained shards.
The 409 engine and 26 warm cases retain two rounds of ten pairs and existing
oracles and regression rules; other families retain their maintained semantics.
The 260-minute total budget includes builds, queues, setup, transfer and cleanup.
Exhaustion requires cancellation and retained evidence, rather than an automatic
retry or threshold change. The broader eight-target performance goal stays paused.

The production diff does not change public APIs, input immutability, canonical
keys, state accounting, checkpoint versions, row update order or ordered
serialization. The test diff does not relax runtime deadlines or Windows
concurrency. The analysis correctly distinguishes the demonstrated executable
mismatch from an unproven explanation for the four CI timeouts and records the
test-harness timings separately from product performance. No normative API or
architecture documentation change is needed for these internal substitutions.

The implementation follows the fixed five-case specification. Both source refs,
wheel/native seals and the original common harness hash are pinned. The harness
hash includes the dependency lock and measuring adapters; workers also verify
their loaded native module and paired input Arrow hashes. Python 3.13.16, actual
four-CPU GitHub-hosted Linux, full affinity and the existing 32-thread settings
are preserved. Builds are not part of this collection.

Collection accounting includes installation, fixture creation, warmup, sampling
and owned cleanup. Setup and artifact transfer remain explicitly separate. A
regression classification can accompany a successfully completed diagnostic;
profiles are separate from paired samples and cannot change their verdict.

The profile claims are appropriately limited: exact stripped modules, 49Hz
`cpu-clock:u`, strict frequency and DWARF/16KiB unwinding; raw data, decoded/raw
events, maps and commands are retained. There is no claim of resolved Rust
functions, negligible sampling overhead, a direct lag metric or proven causality.
The temporary perf permission change is confined to the manual GitHub job and
restored by its shell exit handler. Existing DAL-301 normal behavior and its
profiling permission policy remain intact.

The updated investigation matches the retained recovery evidence: all 51 cases
are present, the original products and harness match, and `group_by` again shows
+10.097% with both interval lower bounds above +5%. All its measured oracles pass.
The 197 host resource snapshots have zero swap usage and at least 8,543MiB
available; the document correctly avoids treating these as continuous worker
peak measurements. Checkpointing remains enabled with a 24-hour interval, with
EOF and the terminal checkpoint outside the paired timer.

The completed five-case run records 179.518 seconds, no observed build overlap
and the original wheel/native seals and common harness hash. All 120 paired
observations pass their oracles, all twelve paired worker completions succeeded,
all fourteen input hash mappings match, and all fourteen exit records are zero.
All 692 host observations have zero swap usage and at least 14,273,576,960 bytes
available. These remain sampled observations rather than continuous guarantees.

A/A is -0.169% with `no-confirmed-regression`; B/B is +0.941% and A/B is -0.493%,
both `inconclusive`. A/B's first interval extends to +9.737%, and B/B's first
interval to +7.252%. Collection success does not establish equivalence or
supersede either earlier +11.663% / +10.097% regression verdict.

The runner reports AMD EPYC 9V45, family 26/model 2, two cores/four threads and
32MiB L3. Earlier artifacts lack CPU model/cache records, so attributing the
different timing level to a CPU-model change would remain speculative. The new
`lscpu --json` step runs before measurement and writes `host-cpu.json` beneath
the existing uploaded result tree. The shard runner preserves that file, and
the aggregate consumes `results.json` rather than arbitrary JSON metadata.

The raw profile streams contain 1,228/1,414 CPU samples in the two 30-second,
49Hz windows, with 44/52 successful oracle-checked workloads. Recorded trace
validation reports no lost or throttle events. Job logs confirm the temporary
profiling permission changed from 4 to 1 and returned to 4. The six documented
native ranges, self counts and percentages match `profile-address-summary.json`;
the leading range has 1,142/1,306 inclusive samples. Its most sampled instruction
is at offset `+0x21e5` in both products, with 67/70 self samples.

The retained normalized disassemblies each contain 3,746 identical offset and
instruction pairs for A, B and the existing ordinary local symbolized artifact.
External call targets and RIP-relative relocations are normalized away. This
supports an explicitly inferred mapping to the local
`WindowAggregateOperator::process_data` future, not resolved original symbols,
identical executable layouts or proven causality. Short prefixes do not resolve
other function names. Hash/lookup work is a lead for a future reproduction;
this profile includes work outside the paired timer and does not identify the
cause of the earlier regression.
The analysis now qualifies the lookup-region interpretation at its point of use
and explicitly states that these observations cannot exclude other causes.

The implementer reports the one authorized complete suite was dispatched as
`38228629578`, comparing the pinned main baseline with candidate
`6978f3bf773d76f77d60a4d181e2ad88596438d1`. No result is asserted here. The only
subsequent source-file diff adds the explanatory comment and two suppression
annotations inside the test module; all executable expressions are unchanged.
The product implementation, dependencies, features and benchmark harness remain
equivalent to that candidate. Continue using this run's exact candidate identity
and artifact seals; do not relabel its products as the later comment commit.
These comments require no additional benchmark dispatch. The reported running
Windows job also has no final result in this review.

### Verdict

**Approve** for the reviewed implementation, narrow test-only static-analysis
exception and continued use of the one authorized complete-suite run's original
candidate products. No source findings remain. No additional measurement is
required for these comments.
This is not a merge approval: required CI, including the four Windows failures,
and the complete benchmark acceptance result are still unresolved. The original
regression is not marked fixed, and no speedup is established by these local
tests or the earlier non-reproducing profile. Keep PR #412 draft until the full
report has no missing cases, correctness/infrastructure errors or confirmed
regressions, required CI is green, and the final specialist review is complete.
