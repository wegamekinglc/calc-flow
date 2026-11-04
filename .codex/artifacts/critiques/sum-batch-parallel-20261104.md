# Single-Batch Parallel Integer SUM — Preimplementation Review

**Branch:** `feature/sum-batch-parallel` → `main`

**Base:** `c9afa829ff025280ab346667582c62dc26a34785`

**Specification:** `c53dbe99042e591333b9697f8b8c7257d684d4d1`

**Scope:** concurrency specification and existing ownership primitives; no PR
or completed implementation is under final review here.

## Summary

The proposed route can parallelize local integer SUM work within one original
Batch while retaining one coordinator for ordered global groups, historical
state, output acceptance, and atomic commit. The existing job-owned gather
service provides the necessary bounded execution and abandonment ownership;
a second executor or cross-Batch concurrency is unnecessary.

The review permits focused RED/GREEN implementation with the concrete
restrictions below. It does not establish a performance gain or approve an
implementation, CI result, or merge.

## Build and Test Results

- Rust: **Not run**. This is a read-only specification/source review.
- Python: **Not run**. Measurement tooling has a separate forthcoming review.
- Native builds, measurements, remote updates, and CI polling: **Not run**.
- New observed test failures: none; no tests were executed by this reviewer.

Source inspection used `cat`, `sed`, and `rg` on incremental SQL, grouped SUM,
predicate and normalization code, compact transaction dispatch, and the gather
parallel, worker, process, cleanup, and lifecycle test modules. `git status`
and `git log` identified the specification checkpoint. These inspections are
not evidence that a future implementation passes the affected tests.

## Blocking Issues

No unresolved issue blocks focused TDD under the selected design. The
following findings were resolved as explicit design restrictions; their
implementation and regression evidence remain required before code approval.

### Optional parallel memory must not defeat serial admission

`GatherScope::submit_parallel_work` can encounter a budget refusal after
installing an attempt: `fund_work` adds execution/home credit and
`ProcessService::prepare_pool` adds generation credit. A failed cold-generation
admission can leave funded, reusable home control. Catching any `Budget`
error and immediately running serially therefore does not by itself preserve
the memory available to a serial-admissible input.

The selected first implementation checks for the exact concrete
`UnboundedMemoryPool` before creating parallel structures or submitting work.
Every bounded or custom scratch pool follows the original serial path,
whether its gather owner is cold or warm. This is an internal eligibility
restriction, not a public memory-policy change. A bounded one-byte-short
control must prove serial success, zero worker entry, and unchanged gather
funding; no generic gather modification is authorized by this review.

Unbounded admission still requires real checked reservations and lifetime
accounting. It does not justify uncharged allocations or a claimed RSS bound.

### Eligibility must inspect the bound work actually executed

Only concrete built-in, non-distinct, unordered Int64/UInt64 SUM lists with
nonempty grouping are eligible. Every actual key and SUM argument must be a
bound input `Column`. Mixed aggregates and unsupported shapes remain serial.

`normalize_groups` can retain eager `input_checks` after inlining input
projections. An apparently simple final aggregate does not prove those checks
are infallible. Require an empty `input_checks` list. Check the complete bound
physical WHERE/FILTER tree as well as the existing logical allowlist: reject
implicit `CastExpr`, `TryCastExpr`, CASE, division, callbacks, UDFs, and
unrecognized nodes. The existing WHERE allowlist admits casts and cannot serve
as the sole parallel certificate.

### Preserve serial ordinary-error precedence without serializing workers

Keep the original per-record 8,192-row chunk boundaries. Within each lane,
execute each chunk's reserve/selection/input-check/key/argument/filter stages
in their legacy order before advancing to its next chunk. Do not evaluate all
chunks' eager checks before their corresponding key/argument stages.

Contiguous lanes may execute concurrently. The accepted infallible bound
expression subset removes input-dependent expression failures from the
parallel route; checked optional-capacity/proof failures decline before
launch. The coordinator processes lanes and local first-appearance keys in
original ordinal order and alone applies global group/state/output limits.
Do not surface a later optional proof error ahead of an earlier serial
ordinary error. Unsupported failing expressions remain serial and retain the
existing first-chunk/stage error. Unexpected worker internal errors or panics
use the gather service's deterministic ordinal, non-cancellation-first, and
secondary-diagnostic rules, without serial retry after work starts.

## Accepted Internal Design and Code-Review Gates

### Admission and deterministic reduction

- Require at least four original chunks and at least 32,768 original rows.
  Tiny/empty and one-CPU inputs remain serial.
- Select `U = min(effective gather parallelism, floor(chunk_count / 2), 8)`;
  require `U >= 2`. Lane `i` owns original chunk ordinals from
  `floor(C*i/U)` to `floor(C*(i+1)/U)`, exclusively at the upper endpoint.
  This gives each launched unit at least two complete chunks, bounds queued
  results to eight, and avoids row-boundary changes.
- Local maps assign only local ranks. The coordinator visits contiguous lanes
  and each lane's keys in first-appearance order, assigns global slots, and
  incorporates historical state exactly once. Reversed completion must not
  alter group order or captures.
- WHERE-rejected rows introduce no group. A WHERE-selected row introduces its
  group even when every aggregate FILTER rejects it or all SUM inputs are
  null. Worker-local duplicate counts must never substitute for global
  distinct-group limits.

### Checked ownership and preallocation

The proposed conservative bound uses lane row count `n`, key count `K`, SUM
count `A`, sliced variable-key bytes `V`, and
`C = next_power_of_two(max(n, 4))`. Its encoded-key envelope is
`E <= 2*V + n*(40*K)`, with reservations for three encoded-data copies,
capacity-rounded local maps/ranks/first-row descriptors, existing native SUM
state charges with validity and merge copies, bounded 8,192-row expression
workspace, result metadata, coordinator rank maps, and gather execution,
home, and generation controls. All arithmetic and conversions must be
checked before the covered allocation. This is a rough design envelope;
the exact implementation constants, Arrow encoding bound, and peak overlap
still require source-level verification before the candidate build.

All worker-local keys, maps, partials, results, and their control structures
are engine scratch. Existing coordinator candidates, global partials, dirty
state, captures, and container charges retain their current classifications.
Local duplicate groups do not consume independent global compact-state rows.

The work package must retain the original input backing reservation passed
by `process_compact` until its last retained input buffer is released.
Charging only RecordBatch/Arc handles does not cover input buffers that
outlive a dropped caller future. A direct update without that backing owner
must either reserve its retained input buffers or decline parallel execution.
One shared credited owner may fund package/results if its final release is
after every covered fragment is destroyed. Destructuring `WorkOutput` must
not drop its reservation before coordinator consumption; transfer or retain
credit across every detached result and merge overlap.

### Cancellation, drop, and reusable gather ownership

Directly awaiting `WorkTicket::finish` is sufficient to await actual unit
settlement: it waits for `record.settled`, takes the attempt, and transfers
result credit. Workers must call full `GatherStop::check` at bounded chunk
and key-processing intervals; `check_abandoned` alone does not observe job
cancellation or deadlines. Do not add a cancellation-select branch that
returns before running siblings settle.

Dropping an unfinished ticket/scope calls `abandon`; the existing job owner
keeps running work, inputs, outcomes, and credit until destruction. This can
handle interrupted futures without a new SQL cleanup field when all
ownership remains inside that lifecycle. If results escape settlement into
separately retained asynchronous owners, use the existing observed-output/
retirement mechanism or another explicit proof of equivalent ownership.
Do not close the reusable job gather owner after each Batch. Job close drains
the owner and must release all credit.

No candidate or worker can modify committed groups, index, ledger, dirty
state, sequence, or checkpoint captures before the existing accepted-output
commit boundary. Barrier/EOF/next-input/capture cannot overtake that boundary.

## Focused Test Coverage Required

The specification's acceptance list covers the affected risks without a
Cartesian matrix. Prioritize a real two-unit overlap RED with finite,
stop-aware gates and a maximum-live-unit assertion; serial admission must
never wait on a barrier requiring two workers. The signed/unsigned prefix
oracle, null/FILTER/group-order, fallback eligibility, bounded-pool funding,
result/input lifetime, refusal/cancellation/panic rollback, and managed
Barrier/restore controls should exercise the actual SQL route.

Record genuine base failures for new behavior before implementation. Reuse
unchanged existing controls where appropriate and do not rerun broad product
suites merely to review this specification. The initial focused checks do
not replace required coverage or cross-platform CI.

## Performance and Documentation Consistency

The fixed five-case set is appropriate: dense streams at one/four/eight CPUs,
sparse stream at one CPU, and hot-key stream at eight CPUs. Freeze hot-key
remapping and an independent arithmetic manifest before timing; preserve the
10,000-group retained tail and full per-prefix snapshot checks. Reuse an
af400 artifact only with its real product identity and compatible complete
seals; a reused binary is not a fresh c9afa829 build.

Retain one 600-second budget, two rounds of ten alternating pairs and twenty
total warmups, with builds reported separately. Profile capture/decoding,
fixture preparation, preflight, oracles, analysis, publication, and cleanup
are inside that budget. New worker/coordinator frames need explicit source-
matched attribution and per-thread evidence; the old update-only classifier
must not hide parallel work in an unexplained Other category.

Full acceptance requires all five cases completed, at least one dense
four/eight-CPU case classified `improved`, and no `regression` on any selected
case under the maintained paired rule. Inconclusive results establish neither
equivalence nor speedup. CPU occupancy alone is insufficient. A failed target
must be reported without expanding cases or retrying for a favorable result.
Document the exact unbounded-pool restriction, serial threshold, unchanged
single-Batch/full-snapshot model, and unmeasured paths in the eventual report.

## Verdict

**Approve for focused, constrained TDD.** The specification plus the explicit
internal restrictions above has no remaining preimplementation blocker.
Exact production code, ownership/peak proof, focused RED/GREEN evidence,
measurement tooling/protocol, public candidate provenance, and final results
require subsequent independent reviews. No performance, CI, coverage,
Codacy, or merge approval is granted here; all required remote checks must
finish green before the already-authorized merge.
