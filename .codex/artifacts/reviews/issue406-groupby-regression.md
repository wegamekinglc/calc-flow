## Branch Review: Original-wheel 10M group_by diagnosis

**Branch:** `fix/issue406-groupby-regression` → `main`
**Base:** `e538099228b54882fe3b9fcc72d802e9998221ed`
**Reviewed:** 2026-10-12, uncommitted diagnostic, workflow, test and artifact changes.
No PR checks or prior GitHub reviews were consulted for this branch review.
The reviewer read the changed files and existing verification evidence without
running tests, native builds or benchmarks.

### Summary

This adds a manual diagnostic for the original October 12 wheels: A/A, B/B and
A/B comparisons followed by one address profile per native module. It retains
the original workload, paired classification and common harness, with a fixed
600-second collection budget and cancellation at 550 seconds.

The reviewed scope supports one diagnostic dispatch. It does not change
production behavior or the benchmark gate and does not resolve the repeated
10M streaming `group_by` regression.

### Build and Test Results

- Rust: **Not touched**; no native build or Rust suite run for this change.
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
- New unresolved failures: **None identified** in this diagnostic scope.
- Existing regression: **Unresolved**. The original-wheel recovery again reports
  +10.097% for 10M streaming `group_by`; this review does not turn it green.

Validation files are under this worktree's
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

Changed Markdown alignment, trailing whitespace and final newlines also pass.
Product regression suites, Rust coverage and Studio coverage were not run; no
coverage gate or final CI result is established here.

### Blocking Issues

None remain in the reviewed diagnostic. The following findings were addressed:

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

### Style Issues

None remain in the reviewed files. The specification's table alignment was
corrected. The implementation reuses existing subprocess, measurement and
classification helpers rather than adding a general measurement framework.

### Test Coverage

The focused tests cover fixed products/workload/counts, alternate-product
rejection, wrong-case refusal before dispatch, trace loss and address validation,
bad-host preflight, denied profiling permissions, process-creation cancellation,
termination cancellation and shell failure evidence/permission restoration.
Existing diagnostic tests retain alternating pair order, input-hash comparison,
environment validation, failure evidence and the old DAL-301 contract.

The actual GitHub host, installed perf tool, permissions and native address
capture have not yet been exercised by this diagnostic. Its preflight and trace
checks must fail closed if the host cannot supply the specified evidence.

### Documentation Consistency

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

### Verdict

**Approve** for one dispatch of the reviewed fixed diagnostic after committing
the reviewed files. Preserve the 600-second budget, all five fixed cases and any
partial/failure evidence; do not retry or expand the workload to obtain a passing
result. Evaluate the resulting artifacts before proposing a production fix or
claiming the original regression resolved. This is not a merge-ready or final-CI
claim and does not resume the paused broader performance goal.
