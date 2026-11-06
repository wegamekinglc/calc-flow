# Ranked grouped SUM candidate experiment

The runtime prototype is rejected. It removes one candidate-map lookup per
touched group during partial merge, but neither dense workload achieves the
predeclared maintained `improved` verdict. Delivery restores the runtime and
preserves the experimental source, tools, profiles and complete evidence.

| Case              | Baseline P50 ms | Candidate P50 ms | Change | Verdict                 |
|-------------------|-----------------|------------------|--------|-------------------------|
| Dense, one CPU    | 171.230         | 171.482          | +0.15% | no-confirmed-regression |
| Dense, four CPUs  | 152.138         | 147.029          | -3.36% | no-confirmed-regression |
| Sparse, one CPU   | 21.353          | 21.074           | -1.31% | inconclusive            |
| Distinct, one CPU | 88.291          | 83.124           | -5.85% | no-confirmed-regression |
| Repeat2, one CPU  | 96.154          | 91.760           | -4.57% | no-confirmed-regression |

These pooled medians are descriptive. The maintained decision uses both
rounds' paired-median intervals and its existing 5% rule. Dense one-CPU
intervals are `[-2.792%, +3.568%]` and `[-7.275%, +3.822%]`; dense four-CPU
intervals are `[-13.152%, +2.990%]` and `[-6.112%, +1.648%]`. Neither confirms
the required improvement. Even the distinct control's negative pooled change
does not meet the exact `improved` rule: its interval upper endpoints are
approximately -1.735% and -1.719%, rather than below -5%.

`no-confirmed-regression` is the maintained verdict, not an equivalence or
speedup claim. Sparse remains inconclusive, with a second-round interval
extending to +18.248%; it is not a confirmed regression under the same rule.
No selected case has a confirmed regression, but the missing dense improvement
is sufficient to reject the prototype. The [sealed report](sum-candidate-ranks-20261106/README.md)
retains all ten intervals and their coverage.

## Hypothesis and implementation

The prototype applies only to the existing trusted single grouped Int64/UInt64
SUM admission, `inline_sum.is_some()`. A small integer-slot-to-rank map indexes
a contiguous candidate Vec. Partial merge uses its existing rank directly.
Full snapshots still use slot lookup. Transaction preparation consumes moved
partial slots zipped with payloads before the later prepared-state commit.
Storage scales with touched groups, not historical groups.

There is no second string-key dictionary, new threshold, worker, public
configuration or checkpoint representation. Existing candidate fees remain
unchanged. Global, mixed, floating and ordered aggregates such as LAST_VALUE
keep the legacy path. Output ORDER BY and HAVING do not themselves make a SUM
an ordered aggregate. The earlier inline-state, finalization and typed-output
improvements remain intact. This experiment does not introduce parallel SUM
updates or change the batch partition policy.

Payload capacity follows the small map's actual capacity. The checked
prepayment covers simultaneous old/new map and payload Vec growth, moved
slots and overlapping partial/coordinator state. It uses actual target layouts
and the pinned map's control/alignment bound. A post-allocation capacity check
supplements the prepayment proof. Owned iteration adds no Option payload
array, slot copy, sorting pass or retained complete partial lease.

## Component evidence and correctness

An actual baseline RED observes 256 candidate lookups in the inline merge.
The final ranked route observes zero; the mixed-aggregate positive control
retains 256. Independent value, schema and first-appearance checks pass before
the operation-count assertions.

The allocation probe touches 256 groups over 256 committed seed groups,
including 128 new groups, and emits 384 output groups. Immutable inputs and
the prebuilt plan are outside the counter. Ranked versus forced-legacy values
on the final source are:

| Metric                     | Forced legacy | Ranked    |
|----------------------------|---------------|-----------|
| Held allocated bytes       | 99,744        | 97,696    |
| Lifecycle allocated bytes  | 701,033       | 692,809   |
| Peak live allocated bytes  | 422,209       | 411,697   |
| Lifecycle allocation count | 630           | 638       |
| Peak live allocation count | 402           | 404       |
| Conservative prepaid peak  | 2,689,680     | 2,689,680 |

Allocation counts increase. All counted live allocations and reservations
settle completely. These counters establish no speedup, state-budget reduction
or process-RSS guarantee. Retained bytes are equal at unfavorable capacity
points 257 and 513. Source capacity algebra predicts equality at 10,000
touched groups on this host: map buckets and moved-slot capacity are both
16,384. The favorable 256-group probe is not a general memory-reduction result.

Current forced-legacy lifecycle bytes are 96 above the earlier RED. Map/store
sizes are 64/112 bytes and the current boxed preparation future is 3,288 bytes.
The old future was not measured. Its inferred size is not direct evidence;
the wrapper/async layout is consistent with, but does not prove the cause of,
that constant difference. Generic payload/map fees remain unchanged.

Thirty focused tests pass, including rank mapping, signed/unsigned NULL and
wrapping sums, first appearance, historical and new keys, WHERE/FILTER,
NULL/empty/NUL/Unicode keys, held outputs, recovery and generic paths. New
controls cancel a real admitted transaction at merge yield and verify exact
refunds and unchanged installed state. A complete update refuses at its
measured custom-pool peak minus one byte and retries at the exact peak.

Independent review found that an empty primitive-native partial could transfer
a capacity-four slot Vec without any candidate credit. An actual one-test RED
reproduced it; the fix drops empty storage before releasing partial credits.
The final regression passes. Scoped strict Clippy, formatting, whitespace and
schema checks pass; all 67 changed functions have CCN at most eight.

The separate [component archive](sum-candidate-ranks-20261106/component-evidence.json.gz)
contains twenty original source-verification records/logs, the layout proof
and seven final public-source identities. It does not invent unavailable
original RED test source from its hash. These tests concern the preserved
experimental source; they do not imply that ranked storage is shipped.

## End-to-end resources and CPU profiles

The [sealed resource table](sum-candidate-ranks-20261106/README.md) reports CPU
time, P95 and whole-process RSS for every case. Four-CPU average active-core
medians are 1.325/1.347, and CPU-time medians are 199.532/195.952 ms. The one-CPU
dense RSS medians are 225.87/226.12 MiB; four-CPU values are 239.72/240.09 MiB.
RSS includes preloaded inputs and all retained outputs, not isolated operator
heap or a state-budget charge. These observations establish no memory gain or
new multicore scaling result.

Both dense single-CPU profiles pass their five-window and timestamp/loss
gates. Baseline/candidate have 798/840 accepted user-CPU samples, with 142/167
unresolved leaf samples. All 2,205/2,322 raw timestamps match imported clocks;
maximum alignment error is below `6e-8` ms. RowConverter inclusive shares are
14.41%/14.64%; `filter_record` shares are 15.54%/16.55%. The maintained generic
hash-map target is 10.40%/6.67%, while hashing is 5.26%/7.26% and the outer
candidate routine is 6.52%/7.86%.

These inclusive CPU shares overlap and are not wall-time phases. Generic
hashing/map symbols combine multiple callers. Missing or inlined frames and
zero matched target samples do not establish zero runtime cost or an isolated
ranked-path hit. Source admission and focused operation tests prove the
functional path separately. The profile shift does not identify a causal
speedup; the complete timing evidence rejects the runtime under the frozen rule.

## Frozen comparison and provenance

Five cases use two rounds of ten alternating pairs: 200 formal observations,
twenty warmups and twelve profile jobs. Dense uses 10,000 zero seed rows plus
twenty 100,000-row source batches, filtered to 80,000 SQL rows each. Sparse
uses the same seeds plus twenty 100-row batches/80 SQL rows each. Distinct uses
twenty all-trade 10,000-row complete key cycles; Repeat2 uses two cycles per
20,000-row batch. Repeat2 is a cardinality control, not an admission threshold.

Every job checks twenty-one complete snapshots against independent arithmetic,
schema and input-identity oracles. The timer starts after IPC preload and covers
graph construction, planning, execution, retained output collection, natural
end and cleanup. Oracles are outside that timer but inside the total budget.
DF target/input partitions remain one; CPU affinities are `[0]` or `[0,2,4,6]`
with matching Tokio workers, and external helper pools remain one. Checkpoints
are disabled. Host load and power outside the WSL guest remain unobservable.

The immutable collector stop is **143.80610578000778 seconds** of one 600-second
budget, with zero retries. It includes fixtures, preflight, warmups, correctness,
profiles/decoding, resource statistics, local collector publication, source/
binary/helper post-seals and owned cleanup. Subsequent read-only integrity
checking, exact protocol archiving and settlement verification completed at
**235.56555740401382 seconds** from the same original start, still below 600.
The original collector manifest is unchanged; the separate handoff records
this longer archival tail. There is no second campaign or residual-time test.

All 232 benchmark jobs (200 formal, twenty warmup and twelve profile jobs)
validate all twenty-one snapshots, for 4,872 complete snapshot checks. All 223
physical invocation receipts confirm reaping: 220 timing invocations, two
six-job profile invocations and one preparation invocation. Natural completion,
zero remaining tasks or task errors and empty edges pass across every job.
The final handoff records no remaining native build or measurement processes.

The baseline is public main at study start,
`1fcafde11a247b1be275d3cb6ab41a59476d6832`. The experimental candidate is
[ec1a629f](https://github.com/wegamekinglc/calc-flow/commit/ec1a629f3a4d7e12f53407f795666d58cee91bb8),
tree `7ae05ed4d3ed0fcb9db4445cb81aa73f6ad70739`. Both use common657 from
public `84186554b078e25e71003a74fedc91c63b2b7f70` as their sole native overlay.
The patched Arrow buffer/data/schema crates are 59.3.2; other Arrow crates
remain 59.2. Rust 1.94, locks, features, flags, actual compiler environments
and 292 registry artifact identities match. Fresh core/example records and
all 1,010/1,012 source inputs are verified against public Git blobs.

The candidate normal Release build succeeds in 805.830162640952 seconds,
with binary SHA256 `d2ce2a6644ab0934a8cf46e8f3c8d370eea3ba610fe91dbfa084fc96c176dd61`.
The baseline log reports 21m23s, rounded by Cargo. Its original driver return
code and precise duration were not retained and remain null. Original compiler
success/fresh records, source pre/post and public-blob equality, actual
environment, depfiles and copied binary seals support independent recovered
admission; recovery took 8.132 seconds. The baseline binary SHA256 is
`da0af0d5db0485b1f7ebeb9ddb935bde831d5e813596343e652f9c61f387d128`.
A prior 145.409-second shared-target lock attempt was aborted before an owned
rustc ran; its stale-product cleanup had run. Other native work existed during
baseline compilation, so these build costs are not controlled build benchmarks.

The [protocol](sum-candidate-ranks-20261106.protocol.json) SHA256 is
`75d53e62d98989e28c598bf4b43f402512adb32648363403647be23e3b4cecee`.
Build receipt hashes are `e781654b...` and `2bc036a6...`. The
[complete evidence](sum-candidate-ranks-20261106/complete-evidence.json.gz)
embeds portable compiler/log/environment/depfile text, source/public identities,
all observations and lifecycle records. Large original perf files remain at
recorded local paths with hashes; full imported stacks are public. Thirteen
frozen helper hashes match. Permanent source trees and copied binaries remain
preserved. Reproduction uses those two public revisions, common657, maintained
`sum_state_build` and `sum_candidate_ranks`, with a newly sealed build/protocol
packet and a new budget; original evidence stays unchanged.

## Delivery and next direction

All seven prototype Rust paths are restored to exact baseline contents or
removed when absent there. Delivery then integrates main
`70e88cc70016a24807a8cbe46287070a9a501717`, preserving its separate ASOF change.
The entire delivered core equals that main revision. No candidate-rank runtime,
configuration, dependency or checkpoint change is shipped. Experimental P
remains available in history. The integrated product is not performance
measured; all numbers above remain bound to the frozen 1fc/P comparison.

The [source-verification manifest](sum-candidate-ranks-20261106/source-verification-manifest.json)
binds the unchanged acquisition manifest/protocol and separately packaged
component evidence and archival handoff. It does not amend collector outputs.
Required CI and final specialist review validate the delivery separately.

Nullable/UInt64/generic, finite, checkpoint and tight-budget latency remain
unmeasured. This experiment does not justify changing thresholds, charging
less memory or adding SUM workers. A next bounded hypothesis is the repeated
Arrow filtering/projection work visible in the fresh profiles. Audit selection
materialization and CASE evaluation first, preserve Arrow/caller immutability
and cancellation, then measure one isolated prototype against a new current-main
baseline. Key encoding still consumes samples, but the earlier rejected
dictionary experiment does not become accepted here. These are future research
directions, not measured improvements or extra tests in this campaign.
