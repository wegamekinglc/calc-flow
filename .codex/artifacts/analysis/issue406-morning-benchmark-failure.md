# October 12 morning benchmark investigation

## Current repair outcome

The one authorized [complete repair comparison](https://github.com/wegamekinglc/calc-flow/actions/runs/38228629578)
succeeded: 22 shards, 809 unique results, zero errors and zero confirmed
regressions. Main `e5380992` versus measured product `6978f3bf` gives LOW 10M
group-by 744.856 / 686.566 ms (-7.8257%, `improved`). Source head `db7945ba`
has 34 successful checks and four successful external contexts; combined Rust
line coverage is 92.96%. Original-wheel regression verdicts and their unproven
cause remain documented below. All 150 inconclusive and three informational
slowdown results remain visible; PR #412 merged as `8f890f28` with all 38 checks green and final specialist
approval. The user has resumed the wider eight-target development objective;
see the [next ASOF increment](issue406-asof-admission-bulk.md) for current work.
The later [18:00 scheduled full run](https://github.com/wegamekinglc/calc-flow/actions/runs/38235513830)
also completed all 22 shards and 809 unique rows without errors or confirmed
regressions, retaining 149 inconclusive results. Its run head is `e5380992`;
this successful scheduled run does not replace the original failed-run evidence
or the eight-target OFF/LOW/ON acceptance measurements.

## Confirmed failure

[Scheduled run 38178694560](https://github.com/wegamekinglc/calc-flow/actions/runs/38178694560)
started at 06:00:49 Asia/Shanghai and completed at 08:11:15. The complete report
contains 809 result rows and exactly one error:
`engines/10000000/calc-flow-stream/group_by: repeated regression exceeds +5%`.
The engine shard failed its regression check; the aggregate gate propagated that
failure. This was neither a missing-result failure nor a failed correctness oracle.

| CI observation             | Value                                      |
|----------------------------|--------------------------------------------|
| Baseline                   | `89e511416d5793e11cefd74b1a7c09aae96f81bd` |
| Candidate                  | `8ee5b71b7b862d4e44bd0be390d182cf950d17f7` |
| Baseline / candidate p50   | 853.563 / 953.112 ms                       |
| Combined p50 change        | +11.663%                                   |
| Round 1 paired median / CI | +11.740% [+11.251%, +12.360%]              |
| Round 2 paired median / CI | +11.583% [+11.330%, +11.790%]              |

Each round contains ten adjacent AB/BA pairs with successful oracles. Both lower
confidence bounds exceed the maintained +5% threshold, so the original run is a
valid regression verdict under that rule. The exact order-statistic interval is
nominally 95%, conservatively 97.85% at ten pairs under its iid assumption.
No threshold, sample filtering or verdict was changed.

This historical group_by case uses checkpoints enabled with a 24-hour interval;
its ready-to-output timer ends before EOF and the terminal checkpoint, while the
worker still requires successful job completion. It is a low-frequency case,
not a checkpoint-OFF measurement. OFF/LOW/ON acceptance remains separate.

The original native SHA256 values are:

- Baseline: `48e63a99052bece0cf59309b9057e3e950a8673383a0fbf3b6ca5e1cdc5454a6`.
- Candidate: `ef24e1d8fefb075909f602a054b03a621200abe4230823fd8259ad2b3a9defa5`.

Both release manifests record Rust 1.88.0, Python 3.13.16, equal Cargo lock hashes
and clean exact source revisions. The runner exposed four logical CPUs; both
arms used Tokio/Polars pools of 32, NumPy 2.5.2 and PyArrow 24.0.0. These thread
settings are existing catalog settings, not a newly imposed runtime restriction.

## Narrowing the cause

The source delta changes ASOF key admission and tests. The window/group aggregation
implementation, benchmark inputs and group_by timing adapter are unchanged.
The same morning run's 1M group_by pair was -0.698%, with both round intervals
below +5%. Other 10M paths did not trigger regression. The preceding manual full
run 38164813243 reported 10M group_by 770.869 / 771.040 ms (+0.022%); that run has
different artifacts and host history and cannot invalidate the morning result.

Downloaded the original wheels and verified both wheel/native seals using the
maintained release loader. A fixed local diagnostic reused the maintained
`measure_case` collector, oracle, alternating pairs and verdict, with A/A, B/B,
then A/B. Each comparison had two rounds of ten pairs and one warmup per side per
round. The recorded total budget was 600 seconds, including creation, startup,
oracles and cleanup; no builds were needed. Collection completed in 119.327 seconds
with no observed native-build overlap.

| Local diagnostic | First arm p50 | Second arm p50 | Change  | Verdict                 |
|------------------|---------------|----------------|---------|-------------------------|
| A/A              | 489.122 ms    | 489.322 ms     | +0.041% | no-confirmed-regression |
| B/B              | 494.282 ms    | 494.593 ms     | +0.063% | no-confirmed-regression |
| A/B              | 493.931 ms    | 498.305 ms     | +0.885% | no-confirmed-regression |

A/B paired intervals were +0.635% [-1.219%, +2.336%] and +0.509%
[-0.456%, +2.745%]. This local WSL host (Python 3.13.9, affinity 0–3 on a 32-CPU
host) is not the original GitHub environment. It did not reproduce +11.66%; it
does not establish equivalence or explain the original cause. Artifact seals
match the morning wheels; host/interpreter differences remain explicit.

## Follow-up and limits

Triggered exactly one existing `suite-recovery` run,
[38193911010](https://github.com/wegamekinglc/calc-flow/actions/runs/38193911010),
for the failed `engines-10000000` shard. It reuses the original run's sealed wheels
and exact source refs, preserves the complete shard and unchanged gate, and keeps
resource diagnostics. It does not rebuild or substitute the current ASOF feature.
It completed with all 51 cases and no missing/oracle errors. Only the original
10M stream group_by case regressed again:

| Recovery observation       | Value                        |
|----------------------------|------------------------------|
| Baseline / candidate p50   | 860.439 / 947.314 ms         |
| Combined p50 change        | +10.097%                     |
| Round 1 paired median / CI | +9.951% [+9.255%, +10.574%]  |
| Round 2 paired median / CI | +10.146% [+8.905%, +10.633%] |

Both original native seals and common harness hash match. All 197 host resource
snapshots report zero swap usage; minimum available host memory was 8,543MiB.
The group_by snapshot had 14,196MiB available and a worker RSS of 827,580KiB.
These snapshots show no memory pressure explanation for the repeated regression;
they are not continuous per-worker peak measurements. The systemd unit's reported
3MiB peak does not describe the workers and is not used as their memory evidence.
No additional full-shard retries are planned.

The dispatched workflow definition comes from release main `8ce5380b`, advanced
after the morning run. The recovery job explicitly checks out candidate
`8ee5b71b` and baseline `89e51141`; its product inputs remain the original refs.
The workflow run head must not be mislabeled as the measured product.

## Fixed GitHub diagnostic and remaining uncertainty

The single five-case [diagnostic 38209692076](https://github.com/wegamekinglc/calc-flow/actions/runs/38209692076)
completed at harness commit `82306c6f` in 179.518 seconds, within its 600-second
budget, with no observed native-build overlap. It reused the original wheels and
common harness. All 120 paired observations passed their oracles, all twelve
paired worker completions succeeded, all fourteen worker input IPC hashes matched,
and all fourteen exit records were zero. Profile samples remain separate.

| GitHub diagnostic | First arm p50 | Second arm p50 | Change  | Maintained verdict      |
|-------------------|---------------|----------------|---------|-------------------------|
| A/A               | 476.592 ms    | 475.786 ms     | -0.169% | no-confirmed-regression |
| B/B               | 487.065 ms    | 491.649 ms     | +0.941% | inconclusive            |
| A/B               | 470.508 ms    | 468.189 ms     | -0.493% | inconclusive            |

A/B paired intervals are +0.145% [-4.010%, +9.737%] and -0.784%
[-2.793%, +0.445%]. B/B round one's upper bound is +7.252%. These intervals do
not establish equivalence or erase either earlier regression verdict. The
collector's success means valid collection, not that performance passed.

The new runner identifies an AMD EPYC 9V45, family 26/model 2, with two physical
cores/four threads and 32MiB L3. Kernel, Python and dependency fingerprints match
the failed runs, but the original/recovery reports did not retain CPU model/cache
information. Their baseline was about 854–860ms; this baseline was 471ms. A CPU
model difference is plausible but cannot be proven retroactively from missing
metadata. Scheduled shards now retain `host-cpu.json` via `lscpu --json` before
measurement; this does not change the common measuring harness or its gate.

All 692 diagnostic resource observations show zero swap and at least
14,273,576,960 bytes available. Sampling produced 1,228 baseline and 1,414
candidate CPU samples at 49Hz, with no recorded loss/throttling. The two 30-second
windows completed 44/52 oracle-checked workloads. The original stripped native
modules and their process maps remain the address source.

Offline grouping by the original ELF unwind-table function boundaries gives:

| Original native range start, A / B | Self samples, A / B | Share of all CPU samples, A / B |
|------------------------------------|---------------------|---------------------------------|
| `0x15d14c0` / `0x15e0e60`          | 392 / 391           | 31.92% / 27.65%                 |
| `0x1364920` / `0x1905280`          | 168 / 205           | 13.68% / 14.50%                 |
| `0x1513570` / `0x1522f10`          | 115 / 133           | 9.36% / 9.41%                   |
| `0x1394130` / `0x1935770`          | 96 / 107            | 7.82% / 7.57%                   |
| `0x15140a0` / `0x1523a40`          | 64 / 74             | 5.21% / 5.23%                   |
| `0x13672d0` / `0x1907ac0`          | 48 / 61             | 3.91% / 4.31%                   |

The first range has 1,142/1,306 inclusive samples (93.00%/92.36%). Its most
sampled instruction is at the same function offset `+0x21e5` in both products,
with the mapping described below still an inference. Under that inferred mapping,
the offset lies in string-key lookup code and makes per-row hash work a lead for
a future reproduction. Samples include work outside the paired timer; percentages
alone cannot assign the earlier extra 10% to a particular function or exclude
other causes.

As an explicitly inferred mapping, the first range matches the ordinary local
symbolized `WindowAggregateOperator::process_data` future's 3,746 instructions at
the same offsets after normalizing external calls and RIP-relative relocations.
This is not a matching executable layout or proven original symbol table. Short
prefix matches for other ranges suggest hash/SipHash, window assignment and key
equality helpers, but prefixes alone are insufficient to resolve names. No
profile-build timing or false claim of resolved original Rust frames is used.

This faster-host A/B comparison is inconclusive. Stop the fixed collection here;
do not expand/retry until it produces a favorable result. A useful next causal
experiment requires reproducing the slowdown while retaining host identity and
sampling on that same host. No production optimization is selected from the
current non-reproducing profile alone.

Root cause attribution is still open. A host-dependent binary/code-layout effect
or runner behavior is a hypothesis, not an established finding. No production
change or benchmark-gate relaxation is justified by these results alone.

Raw logs, manifests, original reports, failed-case samples, local fixed plan,
per-comparison evidence and resource trace are preserved under the main checkout's
`target/issue406-morning-benchmark-failure/`.
The fixed GitHub diagnostic is artifact `11728218142`, preserved under
`github-diagnostic/` there; its outcome, raw perf streams, maps, IPC and seals
remain available. The analysis does not mark the original failure resolved.

## User-authorized repair follow-up

The user subsequently requested further fixes and a fully runnable complete
benchmark. This is a new repair/verification phase; the completed original-wheel
five-case diagnostic is not retried and the wider eight-target goal stays paused.

The candidate replaces standard SipHash in window aggregation's four temporary
lookup maps with the workspace's existing randomized AHash. These maps retain
exact key equality and are never serialized or traversed to determine update
order. Group IDs still follow first occurrence; accumulator updates retain input
order and install/encode still consume a BTreeMap. Existing canonical keys, state
budgets and checkpoint formats remain unchanged. This is a testable response to
the hash-processing hotspot, not proof of the historical regression's cause.
The 35 focused window tests pass before and after the internal hash refactor
(0.28 / 0.31 seconds; one pre-existing ignored test). No hasher-type mirror test
was added. Source builds are separate from performance measurements.

PR #412 head `86d0a3bd` Windows CI completed with 2,029 Rust tests passing and four
failures: writer-v2 checkpoint recovery, queued launch cancellation, unstopped SQL
queue draining, and checkpoint-soak cancellation. Linux gates and Codacy passed.
Source inspection found that the sensitive-log test starts Cargo again without
its parent's features while the parallel lib tests execute. This creates a
second native build; it is a resource-contention lead, not proof that it caused
all four timeouts.

A focused regression assertion requires the child to use exactly its parent's
executable. It fails under the old Cargo launch: parent `calc_flow-00fd9a3140990999`
versus child `calc_flow-242dcf7e84a09783`. The child recompiles the native test
library, and the RED test takes 181.80 seconds after its parent's build; nested
Cargo also rewrites only the lockfile's `flatbuffers`/`fs2` ordering. That incidental
worktree change was restored, preserving the unrelated root worktree lockfile.
The fix starts `current_exe` from an awaited `spawn_blocking` task, which owns
the child through captured-output completion; all original
redaction/sentinel/source-location checks remain. An intermediate compile check
rejected `tokio::process`, whose feature is not enabled; the final approach uses
the existing dependencies and does not add process/signal crates. Runtime timeouts and the Windows test concurrency remain unchanged.

The next acceptance check is one complete maintained workflow dispatch comparing
main `e538099228b54882fe3b9fcc72d802e9998221ed` against the reviewed repair commit.
It includes all 22 shards, every maintained case and the existing correctness,
provenance, completeness and regression gates. The fixed sample counts and total
260-minute dispatch budget (builds, queues and cleanup included) are recorded in
the specification. No production speedup or full-suite success is claimed before
those artifacts exist.

The same-executable child regression is now GREEN in 0.01 seconds after a
separate 71-second rebuild. The original RED spent 181.80 seconds in its nested
build/child before failing. These are test-harness observations, not product
benchmark timings. Required Windows CI must still validate its original parallel
configuration and the four previously timed-out tests.

Scoped all-feature core/lib/test Clippy passes with `-D warnings` in 95 seconds.
Rust formatting, generated-contract and whitespace checks pass; Cargo.lock has
no remaining drift in this repair worktree. No full local regression suite or
local product timing comparison was run; the authorized remote complete suite
and original Windows CI configuration remain the outstanding acceptance gates.

### Complete-run provenance and scanner follow-up

The complete repair comparison was dispatched once as
[run 38228629578](https://github.com/wegamekinglc/calc-flow/actions/runs/38228629578),
created at 09:14:49 UTC. Baseline is main `e538099228b54882fe3b9fcc72d802e9998221ed`;
measured candidate is `6978f3bf773d76f77d60a4d181e2ad88596438d1`. The 260-minute
overall budget ends at 13:34:49 UTC. Catalog/ref resolution and its maintained
support tests passed; final release/measurement/summary results remain pending.

Codacy reported one new audit finding on the test's `current_exe` call. The
[upstream rule](https://github.com/semgrep/semgrep-rules/blob/develop/rust/lang/security/current-exe.yml)
matches that call directly; the [Rust security guidance](https://doc.rust-lang.org/std/env/fn.current_exe.html#security)
concerns trusting the path in security-sensitive operations. Here both calls
are under `cfg(test)`, for same-user test reentry and executable identity, without
an authorization decision. Two line-scoped `nosemgrep` comments document this
reviewed exception; no rule or file-wide exclusion changes. Semgrep 1.180.0 with
the upstream rule reports one finding before and zero after, with no scan errors.
The later annotation/review increment changes only test comments and documents;
keep the original measured candidate SHA and seals rather than relabeling the
run as measurement of the later commit. No benchmark retry is needed for it.


### First repaired Windows result

Windows run [38228597776](https://github.com/wegamekinglc/calc-flow/actions/runs/38228597776)
passed for measured repair commit `6978f3bf`. Its Rust job
`114742244069` took 18m19s including build/setup/cleanup. The core test executable
reports 2,033 passed, seven ignored and two separately selected smoke tests
filtered out in 90.54s; both serial smoke tests then pass in 71.13s. Each of the
four tests that timed out at `86d0a3bd` and the sensitive-log child test reports
`ok`. Python tests, Windows cfg Clippy and the Windows gate also pass. The
workflow's existing test parallelism and timeout values were not changed.

This is a successful validation of the repaired source on that CI runner. It
does not establish that the nested build caused every earlier timeout. The
complete performance suite and the final comment commit's CI remain separate
pending gates. The retained log is
`target/issue406-morning-benchmark-failure/repair-complete/windows-repair.log`.


Linux core job `114742242718` subsequently passed in run
[38228597721](https://github.com/wegamekinglc/calc-flow/actions/runs/38228597721):
all-feature workspace Clippy, precompilation, normal Rust execution and
warning-free rustdoc completed successfully. The combined coverage job also
passed (21m45s), including its maintained coverage gate. macOS smoke had already
passed. The remaining source-head scanner finding is addressed by the reviewed
line-local comments above; its hosted result must be rechecked after pushing
those comments. Complete benchmark measurement is still in progress.


### Subsequent Windows fault-test failure and scoped isolation repair

Comment head `c449ad38` passed hosted Codacy. Its later Windows Rust job
`114753441793` in run
[38231970645](https://github.com/wegamekinglc/calc-flow/actions/runs/38231970645)
failed with 2,029 passed, four failed, seven ignored and two filtered tests in
156.27s. The failed cases are `m5_fault_manifest_rename_restart`,
`m5_fault_manifest_rename_panic`, `m5_fault_retention_restart` and
`m5_fault_source_cut_restart`. This differs from the earlier four failures and
prevents treating the first green run as proof that all Windows instability was
caused by nested Cargo.

For ManifestRename/Restart the job reached only SourcesCut, elapsed 5.504s,
with zero injected fault triggers. ManifestRename/Panic did execute its intended
panic, then failed while checkpointing on restart. Other failures show durable
sink checkpoint errors. These logs establish failure before or outside the
intended oracle, but do not identify the exact Windows filesystem/scheduling
cause. The complete raw log is retained in `repair-complete/windows-comments-rust.log`.

The repair extends the already existing serial Windows checkpoint soak stage
to the entire soak module. Identical `runtime::streaming::soak::` selectors defer
that module from the parallel core command and run it afterward with one test
thread. The remaining core suite stays parallel; all fault assertions, runtime
five-second checkpoint timeouts, step/job deadlines, generation artifact capture
and exit-status forwarding are unchanged. No test is newly ignored. Specialist
log inventory finds 66 executable Windows module cases and three already ignored;
final CI must account for all 69. The existing Linux binary lists 70 module cases
out of 2,048 tests; its selection partitions into 1,978 parallel and 70 isolated
cases with no overlap or omission (Windows has a platform-specific inventory).

The workflow-selection check failed once before implementation, then all four
isolation checks passed. The directly affected Python harness module completed
20 tests in 3.113s, with one pre-existing platform skip. Ruff lint/format passed.
These are configuration/harness checks, not a successful Windows runtime result.
The complete suite's original candidate, dependencies, collector hash inputs and
benchmark timing scopes are unaffected by this CI-only change.

### First fresh repair comparison

The completed 100k engine fragment of run `38228629578` contains 65 cases with
zero errors and passes maintained fragment/provenance validation against main
`e5380992` and candidate `6978f3bf`. LOW streaming `group_by` p50 is
7.4830205 ms / 6.770371 ms, a -9.5236% change. Its two round intervals are
[-12.1083%, -7.1205%] and [-10.7744%, -8.6412%], with maintained verdict
`improved`. It retains the 24-hour checkpoint interval and terminal completion
outside the ready-to-Arrow timer. This is a partial new-product comparison;
the complete report and the original failing 10M workload remain pending.


### Windows isolation runtime validation

Isolation head `db7945ba` passed Windows run
[38234579578](https://github.com/wegamekinglc/calc-flow/actions/runs/38234579578).
Rust job `114762066124` reports 1,969 core tests passed, four ignored and all 69
soak cases deferred in 51.42s. Its serial stage reports exactly 66 passed, three
ignored and 1,973 other core cases filtered out in 48.40s. Independent extraction
of test names from the retained log confirms 69 distinct soak cases: 66 `ok`,
three existing `ignored`, no failures. Thus the combined core inventory is
unchanged at 2,042 tests; all 2,035 non-ignored cases execute. PyO3, connector,
Python, Clippy, process-cleanup and the required Windows gate also pass.

The recorded log is `repair-complete/windows-isolation-rust.log`. This validates
the scoped scheduling repair on this runner with the unchanged deadlines and
oracles; it does not prove the exact earlier filesystem/scheduling cause or
promise absence of future CI instability.

Linux run
[38234579540](https://github.com/wegamekinglc/calc-flow/actions/runs/38234579540)
then passed on the same source head. Its core lib execution reports 2,041 passed
and seven existing ignored tests in 177.58s; Clippy, precompilation, the remaining
Rust targets, three isolated PyO3 executions and warning-free rustdoc pass.
The combined Rust/connector coverage job takes 23m52s and reports 92.96% line
coverage (176,363 lines, 12,415 missed), above the unchanged 90% floor. The
retained logs are `repair-complete/linux-isolation-core.log` and
`repair-complete/linux-isolation-coverage.log`.

The final source-head snapshot contains 34 successful check runs and four
successful external contexts, including Codacy, Windows, macOS and every
Coveralls context; none are pending, cancelled or failed. At that interim snapshot, the full benchmark
report was still pending. Twenty completed shards had 640 validated
result rows, zero execution errors and zero confirmed regressions, with 146
inconclusive results and three informational slowdowns retained. The 10M engine
and Rust fragments were pending, so that partial inventory did not establish
complete-suite acceptance; the subsequent final result follows.

## Completed main-versus-repair suite

[Complete suite 38228629578](https://github.com/wegamekinglc/calc-flow/actions/runs/38228629578) succeeded once in 117m20s, including queues, builds, setup, measurement, transfer and cleanup, within its fixed 260-minute budget. Both ordinary locked release wheel builds passed (22m33s / 22m11s). All 22 shards and the aggregate gate passed: **809 unique results, no missing cases, no errors and no confirmed regressions**. Coverage is 409 engine, 26 warm, 248 Python, 118 Rust, five Studio, two frontend and one lifecycle results. The downloaded consolidated JSON exactly matches the 22 independently validated fragments. [Full 809-row report and JSON](https://github.com/wegamekinglc/calc-flow/actions/runs/38228629578/artifacts/11742858650) retain every case.

The maintained labels remain visible: two improved, 62 no-confirmed-regression, 150 inconclusive, 221 external references, 355 informational, three informational slowdowns and 16 validated metrics. This does not establish equivalence for the inconclusive rows.

Commits after `6978f3bf` change only test comments, Windows CI scheduling/configuration checks and evidence documents. Product code, dependencies, features and benchmark harness remain equivalent to that measured candidate. The original measured SHA and wheel/native seals remain authoritative; the products are not relabeled as later-commit wheels.

### Fresh main-versus-repair comparisons

| Case                                | Checkpoint / scope      | Main ms  | Candidate ms | Change | Maintained verdict      |
|-------------------------------------|-------------------------|----------|--------------|--------|-------------------------|
| group_by 100,000                    | LOW 24h / ready→Arrow   | 7.483    | 6.770        | -9.52% | improved                |
| group_by 1,000,000                  | LOW 24h / ready→Arrow   | 78.150   | 73.261       | -6.26% | no-confirmed-regression |
| group_by 10,000,000                 | LOW 24h / ready→Arrow   | 744.856  | 686.566      | -7.83% | improved                |
| join 1,000,000                      | LOW 24h / ready→Arrow   | 115.125  | 115.866      | +0.64% | no-confirmed-regression |
| join 10,000,000                     | LOW 24h / ready→Arrow   | 935.282  | 923.712      | -1.24% | no-confirmed-regression |
| asof_join 1,000,000                 | LOW 24h / ready→Arrow   | 229.265  | 228.689      | -0.25% | no-confirmed-regression |
| asof_join 10,000,000                | LOW 24h / ready→Arrow   | 2105.536 | 2101.136     | -0.21% | no-confirmed-regression |
| join 1,000,000 / batch 1,024        | LOW 24h / ready→Arrow   | 439.490  | 436.722      | -0.63% | no-confirmed-regression |
| join 10,000,000 / batch 1,024       | LOW 24h / ready→Arrow   | 3744.371 | 3765.791     | +0.57% | no-confirmed-regression |
| asof_join 1,000,000 / batch 1,024   | LOW 24h / ready→Arrow   | 955.406  | 959.170      | +0.39% | no-confirmed-regression |
| asof_join 10,000,000 / batch 1,024  | LOW 24h / ready→Arrow   | 8492.758 | 8536.933     | +0.52% | no-confirmed-regression |
| projection 1,000,000 / batch 1,024  | LOW 24h / ready→Arrow   | 415.240  | 425.594      | +2.49% | inconclusive            |
| projection 10,000,000 / batch 1,024 | LOW 24h / ready→Arrow   | 3765.913 | 3727.965     | -1.01% | no-confirmed-regression |
| join 1,000,000 / batch 64,000       | ON 100ms / ack+recovery | 259.762  | 264.458      | +1.81% | no-confirmed-regression |
| join 1,000,000 / batch 1,024        | ON 100ms / ack+recovery | 573.255  | 575.710      | +0.43% | no-confirmed-regression |
| asof_join 1,000,000 / batch 64,000  | ON 100ms / ack+recovery | 379.126  | 377.310      | -0.48% | no-confirmed-regression |
| asof_join 1,000,000 / batch 1,024   | ON 100ms / ack+recovery | 2306.954 | 2403.740     | +4.20% | inconclusive            |
| projection 1,000,000 / batch 1,024  | ON 100ms / ack+recovery | 536.713  | 542.488      | +1.08% | no-confirmed-regression |

Measured main `e538099228b54882fe3b9fcc72d802e9998221ed` and candidate `6978f3bf773d76f77d60a4d181e2ad88596438d1`, using two rounds of ten paired samples for historical engine/warm cases. Every throughput row here, including exact-cursor 1,024-row batches, keeps checkpointing enabled with a 24-hour interval (LOW). Terminal capture is outside the ready-to-Arrow timer, but successful job completion is required. The adapter field `checkpoint_interval_millis=None` selects the throughput protocol; it does not disable the runtime default. This suite supplies no new true-OFF comparison. ON is the paced 100ms checkpoint acknowledgment/recovery contract. Their timing scopes are distinct; subtracting them does not isolate checkpoint cost.

10M group-by is 744.856435 ms / 686.566208 ms (-7.8257%); its round intervals are [-8.7330%, -5.3521%] and [-10.2687%, -6.9408%], with maintained verdict `improved`. The 100k result is also `improved`. At 1M, the -6.2569% p50 difference retains verdict `no-confirmed-regression`: interval highs are -4.7499% and -4.9600%. The 10M runner records AMD EPYC 9V74 with four logical CPUs. Earlier failed runs lack comparable CPU metadata, so this new main-versus-repair result does not prove the historical slowdown's cause or erase those earlier verdicts.

The retained complete report is `repair-complete/report/summary.json` and
`summary.md`; all raw shards are under `repair-complete/available-shards/`.
`complete-validation.json` records the recomputed inventory, labels and group-by
intervals. The collector identity is
`e30672098c018ef1a650492403bf5ae4813d0182e2b4921cb9233b4d26b8b2e6`.
The successful run spans 09:14:49–11:12:09 UTC, 7,040 seconds including builds
and collection. No second full dispatch, filtered samples or relaxed gate follows.

This complete maintained suite does not rerun the separate eight-target native
acceptance inventory. Sealed 1/32-CPU kernels, 1M capture/restore and compaction,
matched retention scaling, full 1,024-row callbacks and traced harness residuals
retain their prior evidence and paused status. Scheduled Criterion capture/restore
cases use different row counts/scopes and cannot be substituted for those goals.
The PR and final review retain every user target and OFF/LOW/ON distinction.

### Final review correction to small-batch checkpoint labels

The first derived comparison table incorrectly labeled six 1,024-row throughput
results OFF. Final source review found that `benchmarks/engine_stream.py`'s
`_start_replay_job` passes a 24-hour interval and does not set `checkpointing=False`;
`StreamRuntimeConfig.checkpointing` defaults to true. `_run_throughput_stream`
also requires terminal completion after the timed output. These cases are LOW,
not OFF. The analysis, PR comparison and derived tables now say LOW consistently;
the original measurements, immutable raw artifacts and verdicts are unchanged.
The original catalog's `checkpoint_interval_millis=None` describes the throughput
variant rather than proving checkpoint disablement. No new true-OFF result is
claimed. ON 100ms acknowledgment/recovery measurements retain their own scope.
The eight-target inventory header distinguishes enabled checkpoint components
from OFF throughput goals, and preserves its historical evidence separately.
