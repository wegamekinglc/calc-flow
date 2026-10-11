# Issue 406: fresh v3 acquisition

Status: implementation specification incorporating independent review. Builds and
measurements remain separate stages under the user's existing task authorization.

## Scope and fixed inventory

Rebuild and measure the final candidate against exact main
`588726c8bce0f1c57b90077064834c4a472ee5f2` using one final common harness. Preserve
the eight targets, fixtures, clocks, oracles, mode semantics, and statistical
rules in [the acceptance protocol](issue406-performance-acceptance.md). This is
one Issue 406 acquisition, not a general measurement framework.

| Row   | Target workload                    | Target                                         | Cases   |
|-------|------------------------------------|------------------------------------------------|---------|
| 1     | Sealed Join, 1M quotes             | Single CPU ≤60 ms; 32 CPUs ≤15 ms              | 6       |
| 2     | Sealed Join, 10M quotes            | Single CPU ≤600 ms; 32 CPUs ≤150 ms            | 6       |
| 3     | Retained Join, 1M→4M quotes        | Scaling ≤4.4×                                  | 6       |
| 4     | Join capture and restore per row   | Capture ≤0.1 µs; restore ≤0.2 µs               | 2       |
| 5     | Join compaction handler latency    | Compaction handler ≤2× steady handler          | 2       |
| 6     | ASOF, 1M rows per side             | Single CPU ≤100 ms; 32 CPUs ≤30 ms             | 6       |
| 7     | ASOF, 1024-row message overhead    | Matched residual ≤2× measured projection floor | 8       |
| 8     | Join, 1024-row handler cost        | ≤0.05 ms                                       | 2       |

The current `acceptance_inventory()` and `collection_groups()` evaluate to
**38 semantic cases, 36 physical groups, and 64 semantic primary arms**.
Rows 4 and 5 each share one invocation between two metrics, giving 60 physical
primary arms. These counts do not include the two retained-OFF controls from
`cffcf57be9c5cada667c6109156bcbfe5b084f27`. Their two rounds of one warmup plus ten
samples create **44 control reference pairs**, each referencing the existing
primary candidate observation rather than collecting the candidate twice.

## Exact identity and build boundary

1. Add `calc-flow.issue406-execution.v3` as a separate plan policy. An external,
   immutable reviewed seal pins full final candidate product SHA `P`, full
   harness SHA `H`, exact main SHA, the cff control SHA, and every source/build/
   installed-artifact digest. Both `P` and `H` must resolve to clean revisions;
   `H` is also the execution revision. Never use a moving branch, derive `P`
   from an untrusted observation, or require a source file to contain its own
   future commit SHA. Later source edits invalidate the seal.
2. Keep v2 validation and its `HISTORICAL`, `FROZEN`, witness, exact-addition,
   and rustdoc compatibility restrictions unchanged. V3 must not call those
   exceptions or relabel older measurements. For every new native/traced build,
   the attested build harness is exactly `H`, with `compatibility=None` and
   `helper_reconstruction=None`.
3. Build a fresh ordinary `P` wheel without instrumentation. Exact main and cff
   ordinary wheels may be reused only with their immutable descriptors, complete
   source and installed-member inventories, ordinary compiler/profile audits and
   matching dependency/environment identities validated unchanged. Preserve their
   actual original build identities; take fresh observations only.
   Build both native targets from main with `benchmark-internals`, and from `P`
   with enabled and checkpoint-OFF profiles. Build fresh traced row-7 wheels
   for main and `P` with the same common overlay from `H`. Seal effective source,
   compiler invocations, release profiles, locks, flags, wheels, executables,
   and installed package members. Preserve current dependency/profile parity,
   ordinary-wheel audits, and absence of `cfg(test)`. The separately sealed cff
   ordinary control retains its actual identity and matching build settings.
4. Main remains enabled-only: its context has no `with_checkpointing(bool)`.
   Keep main OFF unsupported and candidate OFF absolute-only, except the two
   explicit cff controls. Do not synthesize an OFF method, copy candidate
   operators, change main's pool, shrink fixtures, or alter timing boundaries.

## Portable helper boundary

The current overlay constructs successfully for both products, but copied
`benchmark_internals::join_checkpoint_history_4m()` calls
`operator::run_checkpoint_history_diagnostic()`, which main lacks. Copying the
diagnostic would also require missing `StreamRuntimeState::initialize_benchmark_pool`
and `DataFusionRuntime::with_benchmark_pool`; it is not an isolated helper copy.

Version the new overlay recipe explicitly, preserving historical insertion and
witness validation unchanged. Introduce a separate opt-in
`benchmark-checkpoint-diagnostic` feature depending
on `benchmark-internals`. Gate the 4M diagnostic wrapper, its native component
variant/parser/CLI branches, and its production-side benchmark module/re-export
behind that feature. Declare it consistently in the common overlay, but never
enable it for acceptance native binaries or traced wheels. Preserve the
candidate-only correctness diagnostic through its explicit feature invocation.
Ordinary wheel behavior and default production APIs do not change.

The row-4/5 calls already use the same checkpoint API sequence on both products.
The recorded main failure originates in IPC credit admission; main's production
runtime fixes its pool at 1 GiB. The current helper does not establish a fix.
Successful numerical main results for those rows are not assumed.

## Fixed collection and failure outcome

Retain two rounds, one discarded warmup per arm per round, ten measured samples
per round, alternating adjacent pair order, the fixed 32-CPU set and existing
single/parallel affinity and thread controls. Preserve resource guards, full
oracle checks, stage clocks, and the existing regression rule.

Retain the v2 envelope exactly: 5,400 seconds total; collection cutoff at 4,800;
300 seconds reserved for settlement, with its absolute deadline at 5,100;
5-second startup and final-seal allowances; row-5 cumulative limits of 360
seconds per arm and 720 seconds for its physical group. Preserve finite
precharge validation `0 <= precharged_seconds < 4795`, accounting for preparation,
correctness, setup, warmup, process startup, and cleanup. Build time is reported
separately. No budget extension, added cases, extra samples, or favorable retry.

A first baseline warmup ending in a workload failure may exhaust only that
physical arm and permit the remaining groups to be attempted within the fixed
budget. Before continuing, preserve independently identified diagnostic evidence:
exact v3 product/harness/artifact identities, invocation, case and slot, stdout,
stderr, structured error/status when available, process exit, and verified owned
cleanup with an empty process group. Record the failed slot and every remaining
unattempted slot; link both semantic metrics when a physical group is shared.
Do not retry the failed arm or manufacture sample values, zeros, infinities,
speedups, paired statistics, or a successful lifecycle from partial clocks.

This is an incomplete v3 observation, never the historical known-binary
exception. `issue406_failures.py` pins old binary hashes and panic locations;
new hashes or locations must not be substituted into that exception. Integrity,
provenance, controller, resource-guard, budget, and cleanup failures still stop
collection; the permission to continue is limited to a settled baseline workload
failure, not an arbitrary caught exception. Use a closed classifier for the known
runtime EOF shutdown-sentinel failure and checkpoint IPC-credit admission failure,
and the exact recorded retained-ON early checkpoint failure whose cause remains
unresolved. The latter requires its closed 10-key early-failure raw shape, exact
structured checkpoint/source/operator statuses and RuntimeError→CycleValidationError
chain, with no oracle/lifecycle/timing-success fields. Each classification requires
exact phase/reason and sealed invocation evidence. Natural exit alone is
insufficient. Unknown exceptions, oracle/assertion failures, malformed evidence,
signals, cancellation and timeouts stop collection. Candidate/control failures
retain the existing stop policy.

Apply classification to returned failed lifecycle observations as well as raised
failures. Only the first baseline warmup can exhaust an arm; a later failure stops
collection. Preserve diagnostic clocks outside the valid timing ledger. V3 final
validation rejects every failed lifecycle even if all sample counts are present.

If a later native replay depends on an exhausted row-7 E2E arm,
record every dependent replay slot as unattempted because its source observation
failed. Do not launch replay, substitute traces or invent a second workload failure.
Continue unrelated groups and retain incomplete overall status.

All eight rows remain in the final report, with supported/unsupported,
attempted/failed/unattempted, and valid/incomplete states distinguished. Attempting
all eight rows does not complete performance acceptance. Any required missing or
invalid arm leaves overall acquisition **incomplete**, overall acceptance
**inconclusive**, and the CLI unsuccessful; valid unaffected absolute results may
still be reported with their scope. Settling processes never upgrades that result.

## Minimum implementation surface

- `scripts/benchmark_suite/issue406_fresh_v3.py` and its focused test module:
  the exact v3 identity/envelope policy and independent failed-arm records.
  Reuse the fixed inventory and owned runner rather than creating a new runner.
- `issue406_run.py`, `issue406_fresh.py`, and `issue406_fresh_report.py`, with
  their existing focused tests: explicit v3 routing, plan-bound candidate
  identity, continued collection after a settled baseline failure, and truthful
  incomplete reporting. Keep v2 behavior and bridge policy unchanged.
- `issue406_harness.py` and `scripts/test_issue406_harness.py`: the exact common
  diagnostic-feature declaration and fresh attestation validation. Keep
  `issue406_reuse.py` and historical failure exception contents unchanged.
- `crates/calc-flow/Cargo.toml`, `src/benchmark_internals.rs`,
  `benches/support/native_join_components.rs`, `benches/stream_join_perf.rs`,
  `src/operator/mod.rs`, and `src/operator/join.rs`: diagnostic feature gates
  only. Adjust existing diagnostic tests/invocation documentation as needed;
  do not change operator implementations or add main compatibility stubs.

## Focused red/green acceptance

Write and observe each focused failure before implementation:

1. Pure plan tests confirm 38/36/64 counts and 44 references; reject altered
   cases, samples, budgets, main OFF, stale `P`/`H`, missing hashes, non-null historical
   compatibility/reconstruction fields, and mismatched build profiles.
2. Helper tests construct identical common helper bytes over exact main and
   candidate source fixtures; acceptance features exclude the diagnostic,
   while its explicit candidate invocation remains available. At the separately
   authorized build stage, compile the enabled main and candidate acceptance
   targets and traced bindings; static overlay success is not compile evidence.
3. Fake-runner tests inject a newly identified first-main workload failure,
   verify one attempt and 21 unattempted slots for that arm, continue through
   all 36 groups, preserve the 44 control references, and produce no timing for
   the failed arm. Cover returned failed warmups and dependent row-7 replay slots.
   Reject a full-count ledger containing any failed lifecycle. Verify the new
   evidence is rejected by the old binary
   exception and cannot produce successful overall acceptance or exit status.
4. Focused failure tests stop on provenance/oracle validation errors, candidate
   failures, resource exhaustion, or unproved cleanup; exercise cancellation
   during spawn/read/round cleanup under the unchanged absolute deadline.
5. Focused report tests retain every row and mark incomplete/inconclusive even
   when collection reaches the final group. Existing v2 bridge and exact old
   failure tests must retain their original acceptance/rejection behavior.

Run only affected script tests and format/lint checks for implementation loops.
Native checks and actual acquisition require the later authorized build and
measurement stages; this specification performs neither.
