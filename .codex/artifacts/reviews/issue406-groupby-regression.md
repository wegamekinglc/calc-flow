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
The previous increment reviewed Windows soak isolation on
`c449ad38cd9e412cc730bbedae37fee15af29e2e` plus the worktree changes to
`.github/workflows/ci-windows.yml` and `scripts/test_run_rust_tests.py`.
The current evidence review is for source head
`db7945baaac133a6a3449cd8244bb5fab1df5e36`, using retained CI logs and the
final documentation diff, all 22 benchmark fragments and the consolidated
809-result report. Measured product identity remains
`6978f3bf773d76f77d60a4d181e2ad88596438d1`.
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
current executable. The one authorized complete benchmark now passes with
809 unique results, no errors and no confirmed regressions. Its LOW 10M streaming
`group_by` comparison improves from 744.856435ms to 686.566208ms (-7.8257%).
This establishes the current main-versus-repair result; the original-wheel
regressions remain recorded and their historical cause remains unproven.
Source-head CI is successful. Final documentation-head checks still precede merge.
The Windows follow-up moves the complete checkpoint-soak module into the
existing serial stage while keeping the remaining Rust suite parallel.

### Build and Test Results

- Complete benchmark: **Passed**. Retained run `38228629578` is successful,
  with all 22 shards and the aggregate complete. It spans 09:14:49–11:12:09 UTC,
  117m20s including queues and builds, within the fixed 260-minute budget.
  All 809 consolidated entries exactly match their original fragment entries;
  no missing or duplicate case, correctness error or confirmed regression was
  found. The reviewer inspected artifacts without starting another measurement.
- Source-head CI: **Passed** in the supplied snapshot. All 34 check runs are
  completed/successful at `db7945ba`; all four external contexts are successful
  at that same full SHA. No failed, cancelled, pending or inconclusive CI checks
  appear in `source-head-check-runs.json` or `source-head-statuses.json`.
  These include Linux, Windows, macOS, Codacy, packaging, Python compatibility,
  Studio, dependency-policy and Coveralls gates. The reviewer made no CI query.
- Windows Rust: **Passed**. `windows-isolation-rust.log` records ordinary core
  1,969 passed / four ignored / 69 deferred in 51.42s; the serial module records
  66 passed / three ignored / 1,973 filtered in 48.40s. Name-by-name comparison
  against the earlier successful run gives the same 2,042 distinct core tests,
  with no missing or extra cases and no overlap between the two new stages.
  All 2,035 non-ignored cases execute. The three serial ignored cases are the
  existing opt-in private benchmark and two twenty-minute soaks.
- Linux Rust: **Passed**. `linux-isolation-core.log` records 2,041 passed,
  seven ignored and no filtered core tests in 177.58s. The three isolated PyO3
  executions each pass 132 tests, in 3.62s, 3.21s and 3.20s. The same source-head
  core job, including Clippy and rustdoc, is successful.
- Combined Rust coverage: **Passed**. `linux-isolation-coverage.log` reports
  176,363 lines / 12,415 missed / **92.96%**, with the unchanged
  `--fail-under-lines 90` gate. The retained log includes successful opt-in
  Kafka, PostgreSQL/CDC, MySQL and ClickHouse tests in this combined run;
  connector source remains in the reported coverage table.
- Earlier focused Rust verification: **Passed**. The reviewer read
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
- Python and Studio product sources: **Not touched by this repair**. Their
  source-head CI test/build checks pass in the retained snapshot. Detailed
  Studio coverage percentages were not independently re-extracted here.
- New unresolved local failures: **None identified** in the reviewed repair.
- Historical CI failures, superseded by the successful source-head run above:
  the retained Windows log for `86d0a3bd` has
  2,029 passed, four failed and seven ignored Rust tests. The failures concern
  writer-v2 recovery, queued launch cancellation, SQL queue draining and
  checkpoint-soak cancellation. Reusing the current executable removes nested
  native compilation and a demonstrated feature mismatch; it does not prove
  that either caused all four timeouts. The later retained `6978f3bf` log has
  2,033 passed, seven ignored and two smoke tests deferred; both smoke tests
  subsequently pass. The main lib execution takes 90.54s and the smoke stage
  71.13s. This is evidence for that run, not an absence of Windows instability.
- The retained `c449ad38` log from run `38231970645`, job `114753441793`, has
  2,029 passed, four failed, seven ignored and two deferred tests in 156.27s.
  These failures are `m5_fault_manifest_rename_restart`,
  `m5_fault_manifest_rename_panic`, `m5_fault_retention_restart` and
  `m5_fault_source_cut_restart`. The original executable-identity fix is already
  present. The subsequent `db7945ba` isolation run above passes all four cases.
- Windows workflow regression: **RED then GREEN**. Existing
  `windows-isolation-red.log` records the missing complete-module skip assertion
  failing once; `windows-isolation-green.log` records all four configuration
  tests passing in 0.002s. These test the workflow contract, not Windows runtime
  behavior. No Rust tests, measurements or CI queries were launched for this
  incremental review.
- The existing `windows-harness-tests.log` records 20 tests in 3.113s with
  `OK (skipped=1)`: 19 passed and one pre-existing platform skip. The implementer
  reports passing focused Ruff lint/format, PyYAML workflow parsing and
  whitespace checks. These were not rerun by the reviewer.
- Historical regression: **Preserved**. The original-wheel runs retain their
  +11.663% and +10.097% verdicts. The fresh main-versus-repair 10M comparison is
  `improved`; its result does not prove what caused those earlier regressions.
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
coverage were not rerun by the reviewer. The current CI and Rust coverage claims
come from the retained source-head evidence above, not from local execution.

### Blocking Issues

No source or evidence findings remain in the reviewed implementation and final
documentation diff. Source-head CI and the complete benchmark are successful.
The final documentation commit must retain green required checks before merge;
the source-head snapshot cannot be relabeled as that later commit's result.
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
- The final comparison initially mislabeled six 1,024-row throughput cases as
  OFF. Source inspection showed enabled checkpointing at a 24-hour interval.
  The specification, analysis, PR body and derived comparison tables now say
  LOW and explicitly retain required terminal completion outside the timer.
  No measurement or immutable raw artifact was altered by this correction.

The later `current-exe` finding is an accepted, localized false positive in
`sensitive_lifecycle_panic_payloads_never_reach_process_stderr`. Both calls are
inside `#[cfg(test)]`: the parent reenters its ordinary test binary at the same
privilege, and the child checks executable identity as a regression assertion.
Neither call makes an authorization decision or crosses a privilege boundary.
The parent call now explains this and both call lines carry `// nosemgrep`.
There is no rule modification, file exclusion or suppression of product code.
The generic annotations apply to those two lines only; this approval does not
extend to a privileged runner or future authorization use of the path.

The Windows isolation has no outstanding source findings. The same complete
`runtime::streaming::soak::` prefix is used by `--lib-skip` and the serial Cargo
selection. The harness applies its skip only to the calc-flow target group;
connector and PyO3 execution stays unchanged. The soak module is part of the
core lib test, including its macro-generated fault cases and nested modules.
No matching integration/example test is lost. Both stages retain all features,
and a nonzero serial result still fails the required Windows job and gate.
The dedicated stage is skipped if an earlier stage fails, as before; such a run
is failed and cannot establish full coverage or merge readiness.

### Style Issues

None remain in the reviewed files. The specification's table alignment was
corrected. The implementation reuses existing subprocess, measurement and
classification helpers rather than adding a general measurement framework.
The incremental analysis tables also align in the supplied formatting evidence.

### Test Coverage

The Windows change retains every soak test and its assertions. The successful
retained Windows log lists 69 module cases: 66 passed and three pre-existing
ignored cases, including the two separately run generation smokes. The later
failed parallel run lists 67 module cases: 60 passed, four failed and three
ignored; its two deferred smokes were not reached. A successful new serial stage
should account for the same 66 executable cases and three ignored cases.
There is no `--ignored`, new ignored attribute, deadline relaxation or blanket
serialization of unrelated core tests. Async work inside each test retains its
existing runtime behavior. The module's five-second checkpoint deadline and the
50-minute parallel, 10-minute serial and 60-minute job budgets stay unchanged.
The implementer additionally reports a read-only listing of the existing Linux
test executable: 2,048 total, split into 1,978 ordinary and 70 soak tests, with
no overlap or omission. Its platform-specific count does not replace the
69-case Windows inventory from the retained logs.

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
serialization. The child-harness diff does not relax runtime deadlines. The
later CI change isolates only the soak module from other test cases. The
analysis correctly distinguishes the demonstrated executable
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
These comments require no additional benchmark dispatch.

The subsequent Windows failures do not change that product comparison. Relative
to `6978f3bf`, source and workflow inspection finds only the reviewed Rust test
comments plus the Windows workflow and its Python configuration tests; product
code, dependencies and benchmark harness remain unchanged. Results from
`38228629578` remain applicable to that exact measured product and must retain
its original candidate SHA and seals. This CI scheduling change requires its
own required Windows result, which is now successful, not another product
measurement.

The failed `ManifestRename/Restart` case reports `SourcesCut`, 5.5044197 seconds
and zero fault triggers, so it expired before the intended injection. The
`ManifestRename/Panic` case did reach the injected panic but failed its restart
with sink/checkpoint errors; retention and source-cut restart cases also report
unexpected sink/checkpoint failures. These observations justify reducing
cross-test contention as a concrete CI experiment, but do not prove one shared
root cause. Neither the earlier passing run nor the absence of nested Cargo
establishes their historical root cause. The new CI result validates the
unchanged oracles and full test selection on its runner without claiming that
future Windows instability is impossible.

The final specification and analysis document this Windows scheduling follow-up,
the preserved deadlines, the exact selector and the now successful required CI
result. `harness_sha256()` reads the maintained benchmark sources, adapters and
lockfile; its inspected input list excludes both changed CI/configuration-test
files. The earlier partial 100k report remains an explicitly interim observation.
The retained fragment has 65 cases and no errors, with the pinned source refs and
common harness hash. Its streaming `group_by` result matches the documented
7.4830205ms / 6.770371ms medians, -9.5236% change and two negative intervals.
At that interim stage, the 10M workload and consolidated result were still
pending. The final evidence below now supplies both.

### Completed benchmark evidence

The final review completes the existing run `38228629578`; it does not start a
new measurement or repeat the already reviewed 640-case subset. The two newly
available fragments contribute 51 10M engine and 118 Rust cases. The reviewer
read `complete-run.json`, `complete-validation.json`, the consolidated report
and those original fragments, and reconciled every consolidated entry against
the complete fragment inventory. All 809 entries match exactly: 409 engine,
26 warm, 248 Python, 118 Rust, five Studio, two frontend and one lifecycle.
Every case reports `status=ok` and established correctness, all execution-error
lists are empty, and no duplicate or missing case was found. The complete
report artifact is `11742858650`.

All 22 fragments identify baseline `e538099228b54882fe3b9fcc72d802e9998221ed`
and measured candidate `6978f3bf773d76f77d60a4d181e2ad88596438d1`, ordinary clean
release builds, matching dependency-lock identity and consistent wheel/native
seals. The common harness hash is
`e30672098c018ef1a650492403bf5ae4813d0182e2b4921cb9233b4d26b8b2e6`.
The source changes through `db7945ba` remain test comments, CI scheduling/tests
and documentation; they do not change that product or measurement harness.

The previously reviewed engine/warm evidence contains 768 round records,
11,520 successful sample oracles and 1,152 completed worker records.
The new 10M fragment adds 102 round records, 1,460 successful sample oracles
and 146 completed worker records. Loaded native hashes match
the release manifests throughout. Each case retains two rounds of ten samples
per applicable arm, identical round environments and exact agreement between
summary timings and original sample timings. Recorded environments use Python
3.13.16, four available CPUs and 32 Tokio workers; this is not the paused
eight-target protocol's matched one-CPU/32-CPU acceptance configuration.

Read-only reconciliation of the 192 earlier historical paired rows and 22 new
10M paired rows reproduces their medians, order-statistic intervals and
maintained verdicts. All 1,040 previously reviewed warm
cursor observations follow their declared retained-history positions. The
640 paced checkpoint/recovery observations retain matching workload dimensions,
nonterminal epoch evidence, a partial pre-checkpoint row count, verified
recovery and at least the declared 100ms interval. The reviewer did not invoke
the product, a benchmark runner, a test suite or a CI query to inspect these files.

The Rust fragment contains 102 timed cases and 16 allocation metrics. All
16,280 original timings are finite and positive, and the case arrays agree
with all 408 retained arm/round block records. Aggregate medians, round-minimum
changes and maintained informational labels agree with the raw observations.
Both arms have matching machine, dependency, workload and compiled-dependency
identities; their recorded fingerprints and all per-target stamps reconcile.
All 12 benchmark targets have retained successful build completion in both
arms, and the common measured Join harness agrees across arms. The 160 retained
allocation repetitions complete their requested dispatches on the expected
thread and pass their output assertions; all 16 reported allocation metrics
agree with their underlying repetitions and are unchanged between arms.
Rust suite-block timings and allocation counters remain distinct from paired
engine acceptance; neither is silently promoted into the paused native goals.

The complete verdicts are 221 external references, 150 inconclusive,
62 no-confirmed-regression, two improved, 355 informational, three
informational slowdowns and 16 validated metrics. No confirmed regression
appears in this complete run. Inconclusive results do not establish equivalence. The three
informational slowdowns remain visible: small JAX elementwise plan execution,
standard JAX mean kernel execution, and the standard NumPy transpose/reshape
diagnostic plan. Their maintained suite-block labels use round minima;
aggregate medians do not override those labels or turn them into confirmed
paired regressions.

LOW 10M streaming `group_by` records 744.856435ms / 686.566208ms (-7.8257%).
Its paired intervals are [-8.7330%, -5.3521%] and [-10.2687%, -6.9408%], so the
unchanged rule classifies it as `improved`. The 100k result is also `improved`;
the 1M -6.2569% difference retains `no-confirmed-regression`, since its interval
highs are -4.7499% and -4.9600%. The 10M host reports AMD EPYC 9V74 and four
logical CPUs. Earlier failures lack comparable CPU metadata; the new result
does not establish their cause or erase their earlier regression verdicts.

All 18 final comparison rows and all 12 retained interim comparison rows
reconcile with the original case IDs, medians, percentage changes and labels.
The corrected descriptions consistently call throughput cases LOW and paced
100ms acknowledgment/recovery cases ON. The final specification, analysis and
derived tables have aligned pipe columns. Original raw results are preserved.

### Paused eight-target scope and checkpoint modes

The broader optimization goal remains paused. This repair comparison does not
supply new acceptance observations for the following eight target rows.
Historical references below are the user's issue references, not fresh main
measurements. No target is declared met by this complete maintained suite.

| ID | Workload                                   | Historical reference | Single-CPU target             | Parallel target | This review    |
|----|--------------------------------------------|----------------------|-------------------------------|-----------------|----------------|
| 1  | Sealed-dimension StreamJoin, 1M quotes     | 701 ms               | ≤60 ms                        | ≤15 ms          | Not reassessed |
| 2  | Sealed-dimension StreamJoin, 10M quotes    | 8,061 ms             | ≤600 ms                       | ≤150 ms         | Not reassessed |
| 3  | Retained StreamJoin, 1M→4M resident quotes | 5.7×                 | ≤4.4×                         | —               | Not reassessed |
| 4  | Join capture / restore per row             | 1.8 / 2.3 µs         | ≤0.1 / 0.2 µs                 | —               | Not reassessed |
| 5  | Join handler latency during compaction     | 40× steady           | ≤2× steady                    | —               | Not reassessed |
| 6  | StreamASOF, 1M rows per side               | 391 ms               | ≤100 ms                       | ≤30 ms          | Not reassessed |
| 7  | ASOF harness overhead, 1,024-row batches   | 167 ms over operator | ≤2× measured projection floor | —               | Not reassessed |
| 8  | Join operator cost, 1,024-row batch        | 0.2–0.9 ms           | ≤0.05 ms                      | —               | Not reassessed |

OFF means checkpointing is actually disabled, including terminal checkpoint
work. This complete run does not establish new OFF acceptance measurements.
Historical streaming `group_by`, ordinary/small-batch streaming and warm cases
use enabled checkpointing with a 24-hour interval: LOW, with terminal completion
still required outside their processing timer. They are not OFF observations.
The separate paced checkpoint/recovery cases use ON at 100ms and include their
declared durable recovery work; their clocks must not be substituted for LOW.

For the paused target protocol, rows 1, 2, 3, 6 and 7 retain separate OFF/LOW/ON
acceptance. Rows 4 and 5 are checkpoint-enabled component measurements rather
than invented periodic-frequency arms; row 8 distinguishes direct OFF/enabled
contexts. Row 7 requires the newly measured projection floor at its prescribed
size; the historical roughly 70ms reference is twice the historical floor,
not a current measured floor. No missing mode or eight-target row is silently
dropped, and no further eight-target measurement is authorized by this review.

### Verdict

**Approve** for the reviewed implementation, narrow test-only static-analysis
exception, Windows soak isolation and final complete-suite evidence. No source,
documentation or evidence finding remains. The 22-shard, 809-result acceptance
is complete with no missing cases, correctness/infrastructure errors or confirmed
regressions. The measured candidate retains its original SHA and seals.

The final specialist review is complete. Commit the reviewed documentation and
require green checks on that final head before the authorized merge. No new
product measurement is needed for these evidence-only changes. Preserve the
150 inconclusive cases, three informational slowdowns and unresolved historical
causality; do not claim equivalence or completion of the paused eight-target goal.
