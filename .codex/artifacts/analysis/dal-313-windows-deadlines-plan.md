This branch prepares one normal Windows diagnostic execution for DAL-313. It is
observation-only evidence, independent of PR #355 gates and coverage measurement.
Do not push or execute it until the coordinator has obtained an independent
review of the exact control commit and authorized the execution.

## Fixed inputs and execution boundary

Code checkout: `beb0bebccad4fc42cac39e6cb055d60018a55860`. Input comparison:
main `d5906260f2518ba4544a7d717fced83bd49bd26f`. The crates, Python, original
scripts/workflow and build manifests agree; the entire frontend tree is not
claimed equal to main. Original Windows run `36751889485`, attempt **1**, Python
job `110012398542`, Rust job `110012398359` checked out merge
`bb1824fed15d7f3fa74a063c33f678968f547b40`, whose tree equals the final UI head.
Original logs are upstream evidence, separate from future diagnostic artifacts.

The dedicated push workflow checks out that source at the original workspace
root, and the diagnostic commit in `.dal313-control`. Only eight diagnostic files
may differ from the code baseline in the control checkout. No original production
or Python test file changes. A hash-guarded, temporary Rust fixture overlay is
restored byte-for-byte in `finally`; both original and observed copies are saved.
All seven original Rust cases and their complete assertion expressions remain.

The runner requests the original `windows-latest` label and fails before tests
unless it reports `ImageOS=win25-vs2026`, `ImageVersion=20260925.250.1`, native
x64 Windows/NTFS, CPython **3.13.15**, Rust **1.88.0** MSVC, and original attempt 1
on the dedicated push ref. Existing action SHAs and cache action are retained.
The original Python build mechanism is **`uv sync --extra dev`**, with uv
**0.12.21**, source `pyproject.toml` and default native features/profile. This
does not install the previous local `maturin --profile dev` wheel. The installed
package's `direct_url.json`, pyd location/hash and all 27 installed distribution
versions are recorded and checked against the original installation log.

There is no historical wheel artifact or binary hash, committed `uv.lock`, build
isolation lock, or immutable hosted image selection. Installed-version equality
and identical build source do not prove historical wheel byte equality. Dependency
or image drift stops this run; do not lower the guard to get a green diagnostic.
The original hosted runner binary was 2.337.0; it cannot be pinned by this workflow.
The initial runner log must be retained to compare it. Cache state, Defender load,
host scheduling, Azure region, exact original CPU affinity, worker ordering, and
all native thread stacks cannot be reconstructed from the original job logs.

The [official hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
describes rolling labels and hardware. The [image generation helper](https://github.com/actions/runner-images/blob/main/helpers/GenerateResourcesAndImage.ps1)
maps the VS2026 variant to `win25-vs2026`. A label cannot select the historical
`20260925.250.1` snapshot. If that image is no longer served, this proposal has
an explicit environment blocker: an owner must secure a matching archived VM or
authorize a separately labelled, non-equivalent environment study. This branch
has no self-hosted runner dispatch or fabricated image support.

## Preserve original pressure and behavior

Future Python command retains the complete original collection and auto-worker
policy: `uv run pytest -q -n auto --dist load -p no:benchmark python/tests`,
with only `-p dal313_pytest_observer` added. `PYTEST_XDIST_AUTO_NUM_WORKERS=2`,
`JAX_PLATFORMS=cpu`, debug=0 and normal plugin autoload remain. No serial subset,
`-s`, stress reduction, fault injection, deadline increase, or assertion changes.
The plugin activates only for the three previously failing node IDs in its
`TARGETS`; another xdist worker and other cases continue under the original load.

Future Rust command is the original
`python scripts/run_rust_tests.py --python-stress-runs 1 --lib-skip checkpoint_restart_soak_smoke`.
No `RUST_TEST_THREADS`, build-jobs cap, or target filter is imposed. Its serial
embedded-Python lib phase remains serial as the original harness specifies. The
overlay retains the full `late_output_file` binary, records all seven cases'
actual overlap and available parallelism, and observes only the two 30-second
target combinations. An earlier core failure may prevent reaching them; missing
events are missing evidence, not a reason to rerun or bypass that failure.

These broad commands are preparation for a specifically authorized CI diagnosis
to retain the original process topology. They are **not run locally** in this
preparation phase. Other full CI surfaces, soak, benchmarks and coverage are
outside this diagnostic workflow; its result cannot satisfy those gates.

## Bounded observation and artifacts

- Python: selective in-memory event buffer (2048 events per case), weak coroutine
  references, unchanged callback coroutine objects, creation stack only for
  `_native_close`; no extra callback coroutine, tracemalloc, sys profiler, retries,
  shield, or altered owner cancellation. Public `status()` and the existing
  private completed-callback profiler record checkpoint/owner state and callback
  timing. At **0.8s** of each original 1s wait, one event-loop snapshot records
  tasks, await chains, fixture events, live coroutine states and Python thread
  stacks. A **0.9s** faulthandler timer can sample even if the loop is blocked.
  Both timers are cancelled on completion. Error snapshots retain the actual
  exception. Native terminal return follows `wait_idle`, but the native pending
  counter, lease owner, wait-idle entry and native scheduler stacks have no
  existing read surface. They are marked unavailable; Python live coroutines
  are not substituted for native lease counts. Completed profile misses early
  callbacks before job registration and cannot alone identify never-awaited ones.
- Example child: retains original corrupted input, `-O`, argv, cwd, 60s timeout,
  subprocess communication and exceptions. A small diagnostic prefix logs child
  PID/start and `runpy.run_path` entry/return/error; a one-shot child stack at
  **45s** and parent stack at **50s** remain separate from captured stdout/stderr.
  `TimeoutExpired` partial output is saved and the same exception rethrown. This
  is an instrumented command, not byte-identical original argv. No continuous
  import/compute profiler; exact compute entry is unavailable unless a stack
  sample proves it. Empty stacks after a fast finish are explained by cancelled
  timers, not by missing failure evidence.
- Rust: buffer cap **8192** with truncation count, combination/phase/callback
  result events, observed wait point, full returned outcome/status, default
  concurrency and actual overlap. One sample per target case at **24s** records
  the currently awaited stage, checkpoint status, and sampling-thread backtrace;
  future Drop at the original outer deadline records the last wait/status. This
  uses original public job APIs. It is not a stack of every Tokio worker or a
  file-operation ownership/NTFS probe. No generated two-case replacement binary.
- Wrapper: preserves child exit status and separate flushed stdout/stderr logs,
  invocation/start/end records; Rust overlay manifest includes all assertion
  expressions, deadline values and both hashes. Input/precondition failures have
  their own JSON and exit 2. `always()` uploads separate per-surface artifacts
  named with run ID and attempt; no concatenating evidence from different runs.

Normal-path overhead consists of scoped wrappers, memory events, original private
completed-callback profiling and two log-copy threads. Rare deadline snapshots
serialize stacks/status and add I/O/backtrace work; Rust awaited-operation boxing
and timers affect scheduling. Python snapshot elapsed ns is recorded; no baseline
overhead percentage has been measured or asserted. Guard timing tests are not
performance evidence. Full failures preserve normal pytest/harness exit codes;
observation/compile errors must be classified separately. Buffer truncation,
timer delay, missing case entry or native gaps cannot be interpreted as success.
Pre-test pyd hashing and build/cache setup also touch the filesystem and may warm
the import cache. Rust samples require the test runtime to poll; an executor
blocked in synchronous work can miss them. Callback creation stacks are not
tracemalloc allocation stacks or proof of per-coroutine await entry. These gaps
remain explicit even when a diagnostic test passes.
Hard host shutdown can prevent `finally` and artifact upload; retain GitHub's
full original-attempt job logs and report that loss rather than claiming cleanup.

## Review, trigger, permissions and collection

After independent review and coordinator authorization, a repository writer with
workflow-write capability (SSH write access, or PAT contents-write plus workflow
scope / corresponding fine-grained workflow permission) verifies main/UI refs,
the reviewed local commit, Actions policy, branch rules and that the dedicated
remote ref is absent. Commands below are **planned, not executed**:

```sh
git ls-remote origin refs/heads/main refs/heads/fix/DAL-313-studio-editing-identity refs/heads/diagnostic/DAL-313-windows-deadlines
git push origin <REVIEWED_CONTROL_SHA>:refs/heads/diagnostic/DAL-313-windows-deadlines
```

This normal branch-creation push loads the workflow from that reviewed commit;
no default-branch `workflow_dispatch` requirement, empty commit, force push, ref
deletion, or rerun attempt 2. If the ref already exists, stop and hand back for a
decision; do not recycle it automatically. Runtime token is contents-read only;
artifact upload uses the standard Actions runtime artifact service. No coverage
or release token, deployment, production credential, or elevated action required.

Take at most one nonblocking snapshot after the authorized push. A later
explicitly routed collection uses the new run ID and verifies attempt 1, control
SHA, source SHA, runner startup, guard success, entry into each requested case,
deadline/exception, and observed artifact health. Planned read-only collection:

```sh
gh run view <NEW_RUN_ID> --repo wegamekinglc/calc-flow --json headSha,event,status,conclusion,attempt,jobs
gh api repos/wegamekinglc/calc-flow/actions/runs/<NEW_RUN_ID>/attempts/1/logs > diagnostic-attempt1-logs.zip
gh run download <NEW_RUN_ID> --repo wegamekinglc/calc-flow --dir diagnostic-evidence
```

Retain original `36751889485/attempts/1` logs independently. A local or future
diagnostic GREEN does not reproduce the original timeout, prove flakiness, restore
coverage provenance, or authorize merging PR #355. Native lease/idle uncertainty
may require an explicitly reviewed native observation design by the appropriate
specialist; this preparation invents no public API and makes no product repair.
