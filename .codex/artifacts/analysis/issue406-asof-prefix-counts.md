# Issue 406: ASOF prefix dictionary counting

This increment follows the independently reviewed admission optimization in
PR #413, merged as `5fef1a31abb67dacc4d3e243241090e177396b89` after all 38 checks
passed. Its initial source base was `8f890f28`; it is now rebased onto that exact
merged main. The complete eight-target objective remains active. The
[specification](../specs/issue406-asof-prefix-counts.md) records the profile
selection, scratch constraints and verification scope.

## Implementation and regression evidence

`ChunkRun::visit_keys` previously copied and sorted one `u32` dictionary ID per
selected row merely to count each key. A recorded failing test isolates this
scratch by reusing already populated prefix maps. The old one-key, 1,024-row
case allocates 4,096 bytes instead of the required zero bytes. Its focused RED
exits 101, with one failed test and 2,047 filtered out; the build takes 188
seconds. The raw log is retained under
`target/issue406-asof-prefix-counts/prefix-counts-red.log`.

The replacement uses a 64-slot stack histogram for small dictionaries. Larger
dictionaries use a heap histogram only with at most half as many slots as
selected rows, keeping its scratch within the old row-ID copy on supported
platforms. Sparse dictionaries and small cuts keep sorting. Visits retain
ascending dictionary-ID order; zero counts are skipped before resolving retired
slots. Stored columns, funding, journals and commit boundaries are unchanged.
The scratch threshold does not limit supported key counts or runtime memory.

The existing test registration now covers 1, 64, 65, 128, 512 and 1,024 keys,
including both histogram paths and sorting. A partially consumed 128-slot
dictionary exercises retired entries, nonzero starts, empty/tiny cuts, the
histogram selection boundary, rowwise owner counts, projected inventory and
checkpoint journals. The focused GREEN passes; all 341 serial ASOF tests pass
in 24.78 seconds. Scoped Clippy passes in 91 seconds. Formatting, spec references,
generated-contract drift and whitespace checks pass. Specialist source review
is Approve. The acquired release comparison follows below; required CI remains
pending.

After rebasing onto `5fef1a31`, all 341 serial ASOF tests pass again in 35.54
seconds. Measurement setup review exposed a separate harness bug: every ASOF
1,024-row case tried to export a matched fixture and call native dispatch-trace
methods, which ordinary production releases do not expose. Both operations
now belong only to row 7's matched-harness case. Row 6 small-batch controls use
ordinary processing timing. Two focused tests fail before this correction;
all 23 tests in the affected Python module pass afterward, including existing
row 7 trace, settlement, ownership and failure-evidence checks. Scoped Clippy
also passes on the rebased source in 167 seconds. Measurement setup review is
Approve after four adapter tests cover valid ON evidence, rejected manifest/
window/cursor mutations and mismatched compiled dependency identities.

The final CI exposed an unrelated, pre-existing race in the Join retained-key
scratch test. The parity log records an unsigned subtraction overflow after
the second key batch is dropped. Admission's string-copy worker uses a shared
process service; another parallel test can trigger retirement and refund its
16,384-byte generation between this test's global pool snapshots. The test now
uses the existing isolated service fixture and binds its job to that service.
All original allocation, admission, scratch lifetime and final refund assertions
remain unchanged. Runtime source and the measured products are unchanged.
The failing CI log is retained as `ci-parity-failure.log` in this increment's
evidence directory. Independent review confirms the diagnosis and narrow scope;
the focused regression and all 11 native-lookup module tests pass. Scoped
all-feature core/test Clippy passes in 86 seconds; formatting and whitespace
checks pass. Required CI on the corrected head remains pending.

A follow-up audit finds the same shared-service path in this module's allocation/
alias-funding and exact string-admission tests. Both now bind their case loops
to the same isolated fixture policy; each iteration still awaits job cleanup.
The module's 125 assertion macros and 11 test registrations remain unchanged across
this preventive increment. Neither test has an observed CI failure; the recorded
RED above belongs to the retained-key scratch test. All 11 affected-module
tests pass with eight test threads in 0.30 seconds; scoped all-feature core/test
Clippy passes in 25.27 seconds. Formatting and contract/whitespace checks pass.
Required CI on the final corrected head remains pending.

## Performance scope

The ordinary-release comparison below measures this increment. Allocation
assertions alone do not establish an elapsed-time improvement. Preserve
PR #413's original measured binaries and its interrupted ON result separately.
Current-main release comparisons must retain true OFF, LOW and advancing-epoch
ON distinctions, fixed paired counts, all correctness/lifecycle checks and the
inclusive acquisition budget. The [complete benchmark for PR #413](https://github.com/wegamekinglc/calc-flow/actions/runs/38254036436)
finished all 26 jobs and 22 shards. Its 809 unique results have no errors or
confirmed regressions, retaining all 154 inconclusive and three informational
slowdown results. Final independent evidence review is Approve. Its exact
products remain main `8f890f28` and candidate `09894223`; the measured binaries
are not relabeled as the later squash or newer main dependencies. The new
comparison rebuilt both arms from merged main and the prefix candidate.

The [October 13 06:00 scheduled run](https://github.com/wegamekinglc/calc-flow/actions/runs/38314145431)
also succeeds at exact head `748eec89`, before the PR #413 merge. Its 24 jobs
cover all 20 shards in main's maintained post-Studio-split catalog, with 802
unique successful results, zero errors and zero confirmed regressions. The
independent Studio split removed its Studio backend and frontend shards;
the received shard set exactly matches the current catalog. All 156 inconclusive
and five informational slowdown results remain visible. This is complete-suite
operability evidence, not a prefix-candidate or true-OFF acceptance measurement.

## Fixed paired ordinary-release comparison

Baseline is exact merged main `5fef1a31abb67dacc4d3e243241090e177396b89`;
candidate product and execution harness are
`b77cd8bf2b6d9b9598a5441ab849013e48060cf2`. These remain the measured
identities after later documentation commits. The baseline/candidate native
SHA-256 values are
`99bdd4d17ad1a2d2d59ea083c02245c86aa1f875c2d83ddd4477942fdfb775cf` /
`7df6e7267390d6bf200a582657d95a57fd3d4e8c55b7aa47b75dfdfd11c7d538`.
Both ordinary production wheels retain symbols and contain no
`benchmark-internals` features. All 294 compiled artifact feature/profile
identities, compiler, environment and normalized lock identities match.
Builds took 796.214 / 375.393 seconds, recorded separately from acquisition.
The default runtime pool remains unbounded.

The fixed selection has four cases, each with two rounds of ten adjacent
AB/BA pairs and one warmup per arm per round. Frozen execution-plan SHA-256:
`62fbd0617977850b7584c662c42c9b7f6ff870952bebb245593f5ff3a00fb74d`.
The inclusive 600-second budget starts at candidate build completion and charges
audit, preflight, fixture creation, warmup, correctness, process lifecycle and
sealing; actual acquisition is 149.886 seconds. All 176 raw observations have
full-output oracles, completed lifecycles and cleanup. The 132 OFF observations
have zero checkpoint state files; all 44 ON observations have actual advancing
processing epoch 1 with corresponding nonterminal manifests and recomputed
processing windows. No worker was forcibly killed or survived settlement.
All 390 output seals and every paired statistic were independently checked.

| Workload                         | Main 5fef1a31 p50 | Candidate b77cd8bf p50 | Round changes / CI95                              | Verdict                 |
|----------------------------------|-------------------|------------------------|---------------------------------------------------|-------------------------|
| ASOF 1M/input, 64k, OFF, 1 CPU   | 142.807 ms        | 141.100 ms             | -4.13% [-6.31%, +0.85%]; -1.10% [-7.01%, +3.16%]  | no-confirmed-regression |
| ASOF 1M/input, 64k, OFF, 32 CPUs | 173.554 ms        | 164.884 ms             | -3.92% [-7.37%, -1.79%]; -5.06% [-7.42%, -2.23%]  | no-confirmed-regression |
| ASOF 1M/input, 1024, OFF, 1 CPU  | 601.568 ms        | 602.688 ms             | +0.39% [-1.98%, +10.15%]; +2.39% [-1.92%, +6.82%] | inconclusive            |
| ASOF 1M/input, 64k, ON, 1 CPU    | 143.870 ms        | 137.799 ms             | -3.61% [-12.55%, -0.99%]; -5.05% [-7.55%, -2.19%] | no-confirmed-regression |

Intervals are the maintained conservative CI95 order-statistic intervals
(actual coverage 97.85% for ten iid pairs). The maintained rule confirms a
regression only when both rounds' lower bounds exceed +5%; an interval crossing
+5% is inconclusive. WSL2 host isolation remains unverified. The one-CPU OFF
intervals include zero: its lower aggregate median does not establish a gain.
The 1,024-row control is inconclusive and is not evidence of equivalence.
The 32-CPU medians are lower, but both absolute OFF targets (100 / 30 ms) remain
unmet. The ON arm measures a distinct checkpoint configuration; these independent
configuration medians do not identify checkpoint cost or a checkpoint speedup.
LOW is not remeasured in this increment, and older LOW results are not relabeled.
Row 6's small-batch control does not measure row 7's matched overhead or row 8's
full Join callback.

The candidate's [complete suite](https://github.com/wegamekinglc/calc-flow/actions/runs/38334954225)
has been triggered against exact main `5fef1a31`. Its final result and this
branch's required CI are pending. Earlier complete-suite successes above remain
separate evidence; they do not fill missing candidate acceptance measurements.

| Original workload               | Single-thread target         | Parallel target | This increment  |
|---------------------------------|------------------------------|-----------------|-----------------|
| Sealed Join 1M                  | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not measured    |
| Sealed Join 10M                 | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not measured    |
| Retained Join 4M / 1M           | ≤4.4×, OFF                   | —               | Not measured    |
| Join capture / restore per row  | ≤0.1 / ≤0.2 µs, enabled      | —               | Not measured    |
| Join compaction handler ratio   | ≤2× steady, enabled          | —               | Not measured    |
| ASOF 1M per input               | ≤100 ms, OFF                 | ≤30 ms, OFF     | Measured; unmet |
| ASOF 1,024-row harness overhead | ≤2× matched projection floor | —               | Not measured    |
| Full Join 1,024-row callback    | ≤50 µs                       | —               | Not measured    |

LOW/ON throughput goals remain at least 90%/75% of the corresponding OFF
goal. For ASOF this means OFF ≤100 / 30 ms, LOW approximately ≤111.11 / 33.33 ms,
and ON approximately ≤133.33 / 40 ms at one / 32 CPUs. These are separate
configuration targets, not achieved results; this increment's single-CPU ON
median also exceeds that absolute ON budget. The full suite and native
eight-target acceptance have distinct scopes; neither supplies missing
measurements for the other. All eight targets remain active.
