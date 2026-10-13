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
generated-contract drift and whitespace checks pass. Final specialist source
and acquired-evidence review is Approve. This increment's required CI and release
measurements remain outstanding.

## Performance scope

There is no release timing result for this increment yet. Source work and
allocation assertions do not establish an elapsed-time improvement. Preserve
PR #413's original measured binaries and its interrupted ON result separately.
Current-main release comparisons must retain true OFF, LOW and advancing-epoch
ON distinctions, fixed paired counts, all correctness/lifecycle checks and the
inclusive acquisition budget. The [complete benchmark for PR #413](https://github.com/wegamekinglc/calc-flow/actions/runs/38254036436)
finished all 26 jobs and 22 shards. Its 809 unique results have no errors or
confirmed regressions, retaining all 154 inconclusive and three informational
slowdown results. Final independent evidence review is Approve. Its exact
products remain main `8f890f28` and candidate `09894223`; the measured binaries
are not relabeled as the later squash or newer main dependencies. The new
comparison will rebuild both arms from merged main and the prefix candidate.

The [October 13 06:00 scheduled run](https://github.com/wegamekinglc/calc-flow/actions/runs/38314145431)
also succeeds at exact head `748eec89`, before the PR #413 merge. Its 24 jobs
cover all 20 shards in main's maintained post-Studio-split catalog, with 802
unique successful results, zero errors and zero confirmed regressions. The
independent Studio split removed its two frontend shards from this repository;
the received shard set exactly matches the current catalog. All 156 inconclusive
and five informational slowdown results remain visible. This is complete-suite
operability evidence, not a prefix-candidate or true-OFF acceptance measurement.

| Original workload               | Single-thread target         | Parallel target | This increment |
|---------------------------------|------------------------------|-----------------|----------------|
| Sealed Join 1M                  | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not measured   |
| Sealed Join 10M                 | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not measured   |
| Retained Join 4M / 1M           | ≤4.4×, OFF                   | —               | Not measured   |
| Join capture / restore per row  | ≤0.1 / ≤0.2 µs, enabled      | —               | Not measured   |
| Join compaction handler ratio   | ≤2× steady, enabled          | —               | Not measured   |
| ASOF 1M per input               | ≤100 ms, OFF                 | ≤30 ms, OFF     | Not measured   |
| ASOF 1,024-row harness overhead | ≤2× matched projection floor | —               | Not measured   |
| Full Join 1,024-row callback    | ≤50 µs                       | —               | Not measured   |

LOW/ON throughput goals remain at least 90%/75% of OFF. The full suite and
native eight-target acceptance have distinct scopes; neither supplies missing
measurements for the other. The next fixed selection should include a maintained
1,024-row-batch OFF control alongside 64k-batch OFF one/32-CPU and ON one-CPU
cases, because short runs exercise a different histogram-to-sort tradeoff.
