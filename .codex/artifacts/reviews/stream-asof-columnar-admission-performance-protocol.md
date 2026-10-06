# A4 performance protocol review

## Verdict and scope

**Approve the bounded protocol below.** This is independent static protocol
review, not approval of measurements or the final performance result. No Cargo,
native import, benchmark, native hash scan, or release build was performed.
Only this new review document is owned by the reviewer.

The original suggestion is corrected in three respects, agreed by the parent:

- The public method is `StreamOperator::process_data`, not `on_data`.
- A large left callback includes the existing owned CPU/gather dispatch,
  constructor, admission install, and retirement waits. These cannot be
  excluded while timing the complete public callback.
- A left-only, zero-output callback measures admission as a diagnostic. It
  cannot establish FR16's 1M ASOF target with declared output work preserved.
  The unchanged eleven-case end-to-end fixture remains separate evidence.

The [source review](stream-asof-columnar-admission.md), [approved critique](../critiques/stream-asof-columnar-admission.md),
and [FR16](../specs/stream-join-asof-acceleration.md) remain authoritative.
The source allocation observation is a different result from throughput, RSS,
or target attainment. The 100 ms target is currently **unverified**.

## Exact comparison inputs

Baseline production is approved A3 `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`,
exported at documentation head `764843e634ae1a1da7a5b010095c3017349a25f4`.
The existing A3 wheel's release metadata records:

- Native SHA-256:
  `e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889`.
- Wheel SHA-256:
  `ab38ee224ee306e1ce8e6524419a6f45cd34170612a80d9e7dc149f327b07e69`.
- Locked dependencies:
  `84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840`.

These values were read from existing metadata, not rehashed during this review.
Future workers must attest the actual loaded native against that seal.

Candidate production is tested A4
`e3e3e11f7982ae7144d2b760b4b77bdddc20c060`, including approved A3.
Documentation head `fd73c1a91bb9d1b017c9de471a99789f867c3830`, tree
`54faf67a20f67ac250690a8f88c8204bef64e877`, preserves that production source.
PR #373's supplied publication tree is the same. Candidate native and Rust
probe fingerprints are not yet known and must be recorded after construction.

Static Git comparison confirms unchanged dependency/build manifests, Python
source, benchmark source, and maintained pairing helpers between this base
and candidate. No J1.6, J2, or other optimization may enter either comparator.

Build the new public Rust probe against **both exact source exports** with the
same probe bytes, Rust 1.88.0, release profile, lockfile, empty `RUSTFLAGS`, and
matching Cargo feature/configuration inputs. `calc-flow` itself declares no
optional feature set. Both sides require their own verified core artifact and
complete immutable link closure, including owned native dependencies and
matching sysroot. The pre-A3 `eccb2697` core is not an A3 baseline merely because
its dependencies match. An A3 Python wheel is not a substitute for the A3 Rust
core closure needed by the new executable.

Reuse the sealed A3 wheel for the Python comparison. Build only the A4 candidate
wheel with the same release, locked, `connector-file` default and
`pyo3/abi3-py313` settings, Rust 1.88.0, Python 3.13.9 and empty `RUSTFLAGS`.
Record source/export, build commands, features, compiler, lockfile, linked
artifacts, wheel/native hashes, and common probe/harness hashes independently.
A mutable shared Cargo cache accelerates builds; it is not sealed evidence.
Build and hash preparation require a later orchestration grant outside quiet
measurement, with all artifacts under `target/`.

## Public Rust admission diagnostic

Use `StreamAsofJoinOperator::new`, `AsofJoinSide::new`,
`StreamAsofJoinSpec::new`, `AsofStateLimits::new`, `Batch::table`,
`StreamJobContext::new`, `StreamOperatorContext::new`,
`EdgeCollector::new`, and the public `StreamOperator::process_data` callback.
Read committed counters with `StreamAsofJoinOperator::status` after timing.
No private admission call or test-only forced-legacy switch belongs in this
release comparison.

The primary shape has one million left rows, 64 canonical short UTF-8 keys,
non-null UTC microsecond time, one native `UInt64` sequence, an exact eighth
valued float, a 32–95-byte UTF-8 payload, and a nullable tag. Values and ordering
match the six-column A3 input construction. Independent owned Arrow records
contain at most 64,000 rows, including the final 40,000-row tail. Times increase
strictly by global row ordinal; key dictionary IDs do not prove order. The
right schema is declared identically but receives no rows. Use identical fixed
state limits on both sides, sufficient row capacity for the declared retained
input and bounded oracle continuation, and the original 1 GiB byte limit.
Any admission refusal is preserved; do not relax limits to manufacture a result.

Construct inputs, operator, context, collector, and Tokio runtime outside the
timer. Move prebuilt `Batch` envelopes through the callback without rebuilding
columns. Start a monotonic timer immediately before the first left callback and
stop after the last callback returns successfully. This one contiguous region
includes validation, identity encoding/duplicate proof, payload preparation,
funding, native dispatch, chunk construction, install/journal work, and both
retirement boundaries. It excludes input construction, checkpoint preparation,
restore, watermark/end output, oracle validation, and process cleanup.

ASOF `process_data` invokes `admit_batch` and does not finalize output. Assert
that no output was emitted inside this region and every left row remains
pending. The measured scope is complete left ingress callbacks into retained
state, not the private constructor, a pure kernel, or full ASOF completion.

Use fresh measurement processes for this new Rust scope so no earlier job or
unobservable native retirement overlaps the next sample. First native worker
registration/startup remains included when needed; do not subtract it. Run
warm-ups in separate fresh processes and record that this is a cold-job callback
scope. A future resident-worker scope needs its own protocol, lifecycle proof,
and case identity rather than silently reusing these observations.

Record actual caller/native thread identities or topology, OS task observations,
Tokio runtime configuration, CPU affinity and thread environment. A current-thread
Tokio runtime does not imply the entire operation uses one OS thread: large
admission runs on the existing native executor. Do not override worker limits,
configure a new executor, or replace the existing funded dispatch path.

Add explicitly labeled diagnostic threshold cases at 0/1/256/257/1,024/4,096/
4,097/64,000 rows with the same ordered scalar shape. Zero rows is correctness
coverage only. The existing inline proof also depends on identity-byte size and
strict ordering, not row count alone. A wide-key or reversal control must retain
its real fallback and funding decision. These added cases are new coverage;
they do not replace any original end-to-end case or prove all supported types
have the same performance.

`set_output_projection` is crate-private. This direct public probe therefore
keeps its full declared output schema for the post-timer oracle. Do not pretend
to measure the projected native constructor by trimming input schemas. Exact
public `Program` projections remain covered by the unchanged compiled runner
matrix below.

## Post-timer correctness and lifecycle evidence

Every accepted callback sample must perform complete validation outside its
primary timer, and preserve the raw evidence before reporting a timing result.

1. Check accepted/pending/state rows, right-empty counters, late/duplicate/error
   counters and logical state charge. Prepare the checkpoint asynchronously,
   then capture at the same epoch. Archive inline metadata and every complete
   state segment. Compare baseline/candidate metadata, bytes and hashes using
   matching schema, operator name, limits, epoch, and batch cuts.
2. Restore into a fresh compatible operator. Compare A3 versus A4 within both
   live and restored trajectories; do not assume a restored journal has the
   same byte topology as its live predecessor unless the baseline also does.
   Apply a real prefix watermark with explicit dual-ingress progress through
   `StreamOperatorContext::with_ingress_progress` and `on_watermark`. Include a
   512-row cut and a cut inside a retained record. Compare counters and
   subsequent complete snapshots across versions.
3. Compare emitted Arrow output against an independent all-column oracle in
   canonical order without sorting/deduplication. Right-empty output has the
   original left values and typed NULLs in every right field. Append a bounded
   nonoverlapping continuation within the unchanged limits, capture again, then
   flush with `on_end`. Prove the initial prefix and continuation are neither
   lost nor repeated and logical pending/state rows finish at zero.
4. Separate cancellation/refusal preflight cases from performance observations.
   A pre-cancelled context must reject without admission. A public cancellation
   or dropped future after its first observed `Pending` can record its actual
   outcome, but cannot identify a private worker/install phase without a gate.
   Do not classify that schedule as a deterministic during-worker or
   before-install proof. Preserve status, snapshots, errors and terminal outcome;
   do not retry a failed live callback merely to obtain a successful timing.
5. Public operator fields do not expose the pool, and
   `StreamJobContext::gather_owner` is crate-private. Logical state zero, RSS
   changes, context drop, runtime/process exit and public job status are not
   proof of exact reservation refund. Keep real pool/ownership assertions from
   the approved private source tests as **source evidence only**, including
   abandoned columnar preparation and eligible entry/final-commit retirement
   cancellation. The public managed runner verifies awaited lifecycle cleanup
   separately. Add no public pool introspection or API for this measurement.

Post-timer restore/capture/oracle memory is part of full worker resource cost,
even though its CPU time is outside the callback timer. Save callback duration,
oracle/lifecycle duration, total worker wall time and resource observations as
distinct fields. The source allocation counters cover actual caller and worker
construction; they are not release throughput measurements or a public cleanup
balance counter.

## Unchanged end-to-end comparison

Reuse the sealed A3 `worker.py`, `paired.py`, fixture, maintained helpers,
dependencies, and original eleven-case inventory unchanged. The recorded worker
and driver hashes are respectively
`51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f`
and `f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e`.
Confirm actual bytes and environment before future execution.

The inventory retains 10k/1,024-row and 1M/64,000-row full/projected cases,
five 1M null/repeated-reference/fragmentation challenges, and two projection
controls. Each side's native module is sealed independently; actual dependency,
Polars and Tokio thread observations must match, not just environment strings.

The original ready-to-Arrow timer includes input/watermark enqueue, bounded
tasks/channels, backpressure, output gathering/delivery, and final Arrow
concatenation. Compile/startup, EOF cleanup, status and independent all-column
oracle remain outside that timer. Every sample keeps the unchanged full
schema/order/null/reuse oracle, pending-left-zero and task-count-zero checks,
awaited managed termination, and v3 manifest/delivery evidence. This matrix's
terminal manifest is not an additional nonterminal restore proof; the direct
post-timer restore checks above supply separate nonempty-state evidence.

Keep the original source batch backing decisions, projection declarations,
state/workspace/edge budgets, checkpoint interval, concurrency and completion
helpers. The original allocation inventory constant is not an allocator
measurement. The controls are separate projection comparisons, not a
shape-matched ASOF floor. Do not subtract them from ASOF time or add stage
percentage gains.

## Pairing, uncertainty and resource admission

For each accepted performance case collect two independent rounds of ten
adjacent baseline/candidate pairs, alternating AB then BA order. Record four
warm-ups and 40 successful measured observations, plus every failed or rejected
attempt. The Rust diagnostic uses fresh processes per observation; the frozen
end-to-end fixture retains its original resident workers. These scopes cannot
be pooled, and preflight/warm-up observations are not performance samples.

Use the maintained `paired_round` exact order-statistic interval on
`100 * (candidate / baseline - 1)` per aligned pair. At ten pairs its conservative
95% construction selects the second and ninth ordered values, with 97.85%
coverage under the independence/common-distribution assumption. Alternation
does not prove that assumption. Preserve both round medians and intervals;
use the existing +5% regression and -5% improvement rules with their endpoint
tolerance. Do not substitute a ratio of aggregate medians or choose a faster
round. P50 per side is descriptive; p95/p99 from twenty observations per side
are sparse-tail diagnostics, not a reliable tail-latency guarantee.

Preflight both exact releases with the complete shape, post-timer oracle,
cleanup and process exit before running a matrix. For each larger shape use a
10k → 100k → 1M ladder at its fixed batch policy, key/payload widths, variant,
limits and lifecycle. The original 10k/1,024-row case is not a same-shape memory
observation for a 1M/64,000-row challenge. Separate resource-bridge cases may
use the same frozen constructor with smaller row counts and explicit new
preflight-only identities; they must not mutate or replace the original eleven
timed cases. Each original large case still requires an actual full-shape
preflight before acceptance.

Preserve original `MemAvailable` kB lines and convert using 1,024 bytes per kB.
Read process `VmHWM`, `VmRSS` and `VmSwap` after full oracle/lifecycle work,
including native workers. Screen next shapes using the disclosed row ratio and
1.25 factor against 70% current available RAM, then confirm the actual next
shape. The prediction is a heuristic, not an observed peak or upper bound.
For two resident end-to-end workers conservatively sum both observed peaks;
for strictly sequential fresh Rust workers use the maximum resident worker
peak, record that topology, and wait for exit before launching the next.
Require observed worker `VmSwap == 0`; subtract no interpreter or native floor.

Whole-worker preflight wall time includes process startup, fixture construction,
warm-up/sample, full oracle, lifecycle, IPC and exit. Estimate execution cost
using the actual process topology and full observed costs, rather than the
callback timer. The frozen resident case keeps its existing
`2 * complete two-worker preflight + 18 * sum(complete sample costs)` estimate;
the fresh-process diagnostic requires its own 40-worker cost estimate. No user
duration budget exists. Resource refusal, lifecycle/oracle failure, fingerprint
mismatch, unsupported coverage and interrupted attempts remain visible in
their unique raw directories; none can be relabeled a speedup.

## Handoff and acceptance limits

Future execution requires matching release/probe seals and functional preflight,
then an explicit team quiet grant after builds, tests and heavy artifact scans
stop. Preserve probe/harness/driver and source identities, command arguments,
loaded binaries, actual thread/environment observations, pair ordering,
timestamps, stdout/stderr, all snapshots/oracles, manifests, resource records,
exit statuses and every attempt. Final independent evidence review must
recompute paired results and compare the full accepted inventory.

Report admission and end-to-end results separately. A callback median below
100 ms is an admission diagnostic observation. FR16's full-work target is
reached or unreached only from a declared 1M ASOF output case that preserves
its complete work and correct result; an unavailable/failed/unmeasured case is
unverified. No source allocation saving, hypothetical timing estimate or
unobserved target is promoted to measured performance acceptance.
