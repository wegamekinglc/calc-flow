# ASOF retirement CI repair

The PR365 Windows failure is `repeated_wide_candidate_is_finalized_in_bounded_chunks`:
`on_end` reports `AsofWorkspaceLimitExceeded` with the original 1 MiB workspace
ceiling, 100 left rows, and one 16 KiB right candidate. The paired Linux check
passed. Neither the budget nor the output assertions are changed.

The payload-pool commit replaces the live pool synchronously, then sends the old
pool and its workspace reservation to another blocking task. Previously the
accepted prefix returned before that task dropped its owners and reservation.
The next output chunk could compete with the previous generation's credit.
Prepared right-column copies have the same deferred destruction boundary.

## Focused RED

Base commit: `e4df13e89a6cb7cbdd6b34227bbb41a324f7b87a`.
The new fixture admits 64 separate one-row payloads and commits a 50-row prefix,
which requires a payload-pool compaction. Its collector occupies the runtime's
single blocking worker after accepting output. The test polls the prefix commit
while destruction is deterministically queued behind that worker.

Command:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  committed_prefix_waits_for_payload_retirement_before_reusing_workspace -- --nocapture
```

Actual result before production changes: one test ran, zero passed, one failed,
with `accepted prefix returned while retired payloads still held workspace`.
The corrected fixture completed in 0.01 seconds after a 49.25-second build.
An earlier fixture run had an async select ordering error; it is not counted as
the valid RED evidence.

## Managed cleanup RED

After the operator wait was implemented, a second controlled test dropped the
operator while a real payload and its 4 KiB reservation remained queued behind
the single blocking worker. Existing `close_and_drain` returned while the
payload's weak reference remained live and the pool still reported 4096 bytes.

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  managed_job_drain_owns_retirement_after_operator_drop -- --nocapture
```

Actual result before the private job guard: one test ran and failed with
`managed cleanup returned before funded retired payloads were destroyed`.
The test took 0.00 seconds after a 56.21-second build.

## Repair

A private operator retirement owner tracks tickets until a blocking destructor
has dropped both retained owners and their reservation. Mutable async handlers,
checkpoint preparation, and output chunks await outstanding retirement before
reusing workspace. Dropping a waiting future does not erase the count; reset
and restore preserve it. Completion releases credit and never installs state.

Each production ticket also owns a private guard registered with the real job's
gather owner before detached work is launched. Managed drain waits for these
guards independently of native pool state. Closing admission prevents new
registrations; dropping a drain future does not discard existing registrations.
The home stores only a count, avoiding an ownership cycle with the guards.

Both payload pools and right-column copies use owned retirement bundles whose
ticket is the last field. Input owners and workspace therefore drop before the
ticket even when an unpolled future or queued closure is destroyed. The custom
waker checks pre-cancelled and never-polled copy futures at the actual managed
refund notification. This is additional structural safety coverage, not a
separate observed RED claim.

The accepted prefix still installs state synchronously without allocations.
Its handler then waits for retirement; the older allocation test's immediate
`Ready` oracle is updated to `Pending`, retaining the zero-allocation and
immediate installed-state assertions.

## Verification state

The current job guard passed all four right-copy tests, all four retirement
tests, and all five prefix tests. A subsequent structural refactor made the
right-copy destruction closure own the complete retirement bundle.

The ASOF module then ran 294 tests: 293 passed, with only the older immediate
`Ready` oracle failing. Its zero-allocation assertion passed. The corrected
oracle passed as an isolated test on the same production tree; the other 293
unchanged tests were not repeated. This distinguishes those two executions
from claiming a subsequent complete 294-test run.

Current production checks:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::asof:: -- --test-threads=1
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::asof::tests::accepted_prefix_installs_pool_compaction_without_allocating \
  -- --exact --nocapture
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  runtime::streaming::gather_work::tests:: -- --test-threads=1
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --test stream_asof_join_state \
  repeated_wide_candidate_is_finalized_in_bounded_chunks -- --exact --nocapture
```

The corrected allocation test passed in 0.05 seconds. Gather lifecycle tests
passed 22/22 in 5.85 seconds, including dropped observers, cancellation,
deadline, worker panic, and native drain. The original 1 MiB integration passed
1/1 in 0.04 seconds after a 94-second build, retaining its 100 left rows,
16 KiB right payload, complete output checks, and resource ceiling. These are
Linux checks; they do not independently prove a Windows CI result.

`cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`,
`cargo fmt --all --check`, and `git diff --check` passed. Project JSON Schema,
Studio OpenAPI, and generated frontend API contracts have no diff. The fix is
local at this handoff; its own remote CI checks have not yet run.

The isolated shutdown fixture includes the test-only patch from root commit
`4954d5f3`, adapted to pass no job ticket. It deliberately proves that an
unregistered standalone retirement can remain alive with 4096 bytes after
gather drain and refund only at complete runtime shutdown. The separate managed
fixture uses a real job ticket and proves refund before managed drain returns.

## PR369 diagnosis boundary

The launch-cancel failure is the preceding `capture()` helper's five-second
readiness/settled wait returning `Elapsed`, before the queued-stop oracle runs.
That helper first waits for source tail and sink writes, then triggers a
checkpoint; the log does not identify which part exceeded the deadline.

The Compaction/Io restart fails at epoch 2 in SinksPrecommitted with two sink
checkpoint errors and RecoveryRequired. Public errors redact their source, so
the log does not establish a filesystem or command-wait cause. The cancel-window
soak takes a generic job-failed branch without logging its status or source;
that alone does not establish a checkpoint failure. The old run has no retained
underlying-source diagnostic files and its smoke diagnostic was skipped.

No timeout, fault, recovery, or terminal-metric assertions have been weakened.
The parent task subsequently observed the Windows Rust checks of quality heads
`7a66e029` and `d9e1e203` passing before this repair was integrated. That is not
evidence that this local fix passed Windows. No speculative PR369 source fix or
timeout change was added.
