# Composite UTF-8 SUM key experiment

The runtime prototype is rejected. It reduces canonical key encoding and
allocated bytes in a focused probe, but neither selected dense workload has
the predeclared maintained `improved` verdict. All five cases are
`inconclusive`. The delivery restores the existing runtime and retains the
experiment's source, tools, profiles and complete evidence for reproduction.

| Case              | Baseline P50 ms | Candidate P50 ms | Change  | Verdict      |
|-------------------|-----------------|------------------|---------|--------------|
| Dense, one CPU    | 185.134         | 189.210          | +2.20%  | inconclusive |
| Dense, four CPUs  | 172.947         | 180.635          | +4.45%  | inconclusive |
| Sparse, one CPU   | 22.838          | 22.725           | -0.49%  | inconclusive |
| Distinct, one CPU | 99.598          | 99.617           | +0.02%  | inconclusive |
| Edge, one CPU     | 108.790         | 121.376          | +11.57% | inconclusive |

These are pooled elapsed-time medians from one frozen cohort. The verdict
comes from both rounds' paired-median intervals, not the pooled change. In
particular, the edge intervals are approximately `[+4.884%, +19.548%]` and
`[+5.573%, +14.085%]`. The first lower endpoint does not exceed the maintained
5% regression threshold, so this case is not a confirmed regression under
that rule. Inconclusive results establish neither equivalence nor improvement.
The absence of a dense improvement rejects the runtime under the frozen
acceptance contract. The [sealed report](sum-columnar-keys-20261105/README.md)
contains every interval and its coverage.

## Hypothesis and prototype scope

The prototype uses a transaction-wide column dictionary for exactly two plain
UTF-8 column keys and one trusted built-in Int64/UInt64 SUM. It retains
per-row hashing while canonicalizing first-seen distinct keys. Native IDs are
temporary; persistent keys and checkpoint encoding remain canonical.

The selected scalar `GroupValuesColumn<true>` route preserves first-appearance
IDs. The generic factory's vectorized collision route does not satisfy this
admission contract. After each existing 8,192-row chunk, newly seen groups pass
through the existing ordered group admission and SUM candidate path before the
next chunk. The dictionary is dropped before history merge and output creation.

The selector requires the exact default unbounded engine pool, an explicitly
configured compact group cap G, default compact byte allowance, no SQL WHERE
or eager input checks, and `G <= N/2`. The half-encoding floor was derived
before timing. Both SQL budgets still default to None. The required byte
allowance is the runtime's JSON-safe default sentinel,
9,007,199,254,740,991 bytes; an ordinary 64 MiB compact budget does not activate
the prototype. Unsupported pools, budgets, types and aggregates use the
existing path. These are experimental conditions, not a newly shipped API.

Chunk-local deduplication would encode 78,488 of the maintained dense input's
80,000 SQL rows: only 1.89% fewer. A transaction-wide dictionary can reduce
that fixture to 10,000 canonical rows. This 87.5% encoding-work reduction
motivates the experiment; it is not an elapsed-time speedup.

## Component evidence and correctness

The allocation probe includes the update selector, header census, dictionary,
take/canonical conversion, partial state, coordinator and commit. Immutable
input arrays and the prebuilt plan are outside the counter. It uses 32,768
rows, 256 groups, two long strings with Unicode/NUL and a late NULL key, plus
full output and independent distinct-key checks.

| Metric                     | Legacy    | Candidate  |
|----------------------------|-----------|------------|
| Canonically encoded rows   | 32,768    | 256        |
| Canonically encoded bytes  | 6,737,152 | 52,634     |
| Total allocation count     | 1,679     | 1,726      |
| Total allocated bytes      | 8,041,056 | 1,002,262  |
| Peak live allocated bytes  | 2,070,541 | 672,331    |
| Held allocated bytes       | 236,224   | 236,224    |
| Conservative prepaid bound | 9,263,620 | 63,274,312 |

Allocation count rises by 47. The full-input/growth-overlap prepayment is
intentionally conservative and increases despite the lower actual allocated
peak. These counters establish a component change, not a reservation
improvement, bounded-pool acceleration, process RSS reduction or overall gain.
The long-key probe differs from the short-key stock workload.

The capacity proof covers `K=min(N,G+R)`, `R<=8,192`, including the remainder
of a chunk that exceeds the group budget. It covers native builder growth and
map rehash overlap, row/ID mappings, take and canonical conversion, and
simultaneous coordinator state. Checked capacity, UTF-8 offset and layout
limits decline the optional path before interning. The header census checks
cancellation per record batch and yields every 128 batches.

The final layout receipt reports a 1,040-byte plan and 424-byte input candidate
container, against its 416-byte legacy equivalent. The optional key plan is
24 bytes; Group and Candidate remain 96 and 128 bytes on this host. Tests
verify the existing 4,096-byte structural floor relative to target layout.

Ten new tests and twenty-eight affected existing controls passed using one
fresh test executable. They cover tuple identity, NULL/empty/Unicode/delimiter
keys, first appearance, FILTER versus WHERE admission, wrapping/null integer
behavior, local versus historical ranks, held snapshots, discard/refusal/retry,
cancellation, restore, group/byte limits and unsupported-path fallbacks.
The maintained 80,000-row SQL shape encodes 10,000 distinct keys in the
focused path test. Strict Clippy, formatting, whitespace and generated-schema
checks passed; all 64 inspected affected functions have CCN at most eight.
These checks concern the preserved experimental source, not a shipped
columnar-interner implementation.

## End-to-end resources and profiles

| Case              | CPU P50 ms, baseline to candidate | Whole-process peak RSS P50 MiB, baseline to candidate |
|-------------------|-----------------------------------|-------------------------------------------------------|
| Dense, one CPU    | 185.128 to 189.204                | 226.99 to 227.37                                      |
| Dense, four CPUs  | 221.300 to 226.445                | 240.67 to 243.57                                      |
| Sparse, one CPU   | 22.838 to 22.726                  | 66.12 to 66.51                                        |
| Distinct, one CPU | 99.592 to 99.611                  | 83.17 to 83.24                                        |
| Edge, one CPU     | 108.783 to 121.369                | 94.68 to 96.80                                        |

Average active-core medians in the four-CPU case are 1.280 and 1.272. The
whole-process RSS includes preloaded inputs and twenty-one retained complete
outputs. It is not an isolated operator working set, a state-budget charge or
an engine memory guarantee. These resource medians do not establish a gain.

Both required dense single-CPU profiles pass the five-window and timestamp/loss
gates. Baseline/candidate have 876/963 selected user-CPU samples and 173/174
unresolved leaf samples. All 2,319/2,503 raw timestamps match imported clocks,
with maximum error below `6e-8` milliseconds. RowConverter's inclusive share
falls from 13.13% to 2.28%. The candidate profile includes
`ByteGroupValueBuilder::equal_to` at 7.48% and `group_rank` at 22.01%. Their
absence from the baseline's retained top entries does not imply zero cost.

The three exact interner-route anchors have no standalone symbols in either
sealed nm inventory. The [route record](sum-columnar-keys-20261105/profile-route.json)
therefore marks both arms nonseparable/inlined, with null counts. It does not
prove a Release interner hit, zero cost or absence of the path. Same-shape unit
tests and source admission establish functional eligibility separately.
Inclusive shares overlap and are not phase wall times. A shift from encoding
toward hashing/equality is a plausible explanation, not an isolated causal
measurement. The overall timing evidence does not establish a speedup.

## Frozen comparison and source identity

Five cases use two rounds of ten alternating pairs each: 200 formal
observations and twenty warmups. Dense input has 10,000 seeds plus twenty
100,000-row source batches, leaving 80,000 SQL rows per business batch after
the upstream filter. Sparse input has the same seeds and twenty 100-row
source batches. Distinct and edge controls select one or two complete
10,000-key cycles per business batch and set all selected rows to trade,
preserving quantities, sides and schema. They exercise `N=G` legacy decline
and the first eligible `N=2G` respectively.

Every run retains twenty-one full snapshots for independent schema, arithmetic
and complete-prefix oracle checks. The timer starts after IPC preload and
covers graph construction, planning, execution, retained output collection,
natural end and cleanup. Oracles are outside that timer but inside the
acquisition budget. All runs select one DataFusion partition and one input
partition; four-CPU affinity is `[0,2,4,6]` with matching Tokio workers. Grouped
SUM updates remain serial. Previous studies are separate timing cohorts.

One 600-second acquisition completed in **147.604826276 seconds**, including
fixtures, preflight, startup, warmup, oracles, resources, profiles/decoding,
analysis, local artifact publication, post-seals and cleanup. Builds are
separate. The [manifest](sum-columnar-keys-20261105/evidence-manifest.json)
contains the final stopped-clock debit; the summary written just before the
publication tail has a slightly earlier elapsed value. There were zero
retries, additional cases or extra observations.

All 232 workload jobs (200 formal, twenty warmup and twelve profile jobs)
validate all twenty-one snapshots, for 4,872 snapshot checks. All 223 native
process receipts confirm reaping. Natural completion, zero remaining tasks or
task errors, finished actors and empty edges pass the lifecycle checks.
Post-acquisition source/binary seals and all eleven frozen helper hashes match.

The baseline is main at study start,
`22f4b9a562bdd087a75d810aeeb2161f29a3d6b7`, unchanged by the final pre-freeze
fetch. Its fresh Release build took 599.736255465 seconds, with binary SHA-256
`73bc770aebdf43271f04d896eb7fbf5d20297bd587977e4c26c0d0d6156a4af6`.
The older P74 product was incompatible with current compiled source and was
not reused or relabeled.

The public experimental candidate is
[e492158c](https://github.com/wegamekinglc/calc-flow/commit/e492158c4d7109c779b7b52aac87cf6ffd3e1613),
tree `441f82dfb4558ce0d70613e50958b4da67e70d23`, exactly the reviewed source
and tool tree. Its Release build took 641.063242391 seconds, plus 2.612 seconds
for validation, with binary SHA-256
`db9a9476e975b783790d460b2757e937068fef67e31cf9e4849de95eced66147`.
An external build in a separate target was present early in this build;
these durations are costs, not a controlled build-performance comparison.
No competing native job remained at acquisition readiness.

Both builds use Rust 1.94, identical lock/compiler/features/flags, and the
audited common657 harness from public `84186554`. Core and example actually
compile with `fresh=false`. All 1,008 baseline and 1,013 candidate source
inputs match their public Git blobs except that sole common-harness overlay;
copied dependency records include the new module. Frozen trees and binaries
are preserved. Build receipt SHA-256 values are `14a457da...` and `ffa4f8a1...`.
The frozen [protocol](sum-columnar-keys-20261105.protocol.json) SHA-256 is
`1ba3640f929df2964e330a99c4330bbb5a6736d32cf9d8bb8b064ca119d94422`.
The [complete evidence](sum-columnar-keys-20261105/complete-evidence.json.gz)
retains full receipts, compiler/dependency proof, source/helper seals, raw
observations, clocks, resources and cleanup. Large original perf data stays
at recorded local paths with hashes; full imported stacks are public.

Reproduction uses the public baseline and experimental candidate above, with
the common657 example from `84186554b078e25e71003a74fedc91c63b2b7f70` as the
sole overlay. The maintained `scripts.benchmark_suite.sum_state_build` builder
and `scripts.benchmark_suite.sum_columnar_keys` collector record exact commands
in the archived receipts. New builds require their own receipts and a newly
sealed protocol before collection; the original protocol and evidence remain
unchanged.

## Delivery and remaining questions

The seven prototype Rust paths are restored to the exact baseline: the two
modified existing files are restored and five new private modules are removed.
The measured candidate remains available as experimental history. This PR
ships research tools and evidence; it does not enable columnar interning or
change configuration, checkpoint encoding or thread policy.

Nullable/UInt64/generic/checkpoint/finite latency, tight-byte pools and
near-below-edge latency remain unmeasured. The long-string allocation result
does not establish a speedup for a long-string workload. A further study
could remove redundant canonical candidate/rank lookups in the existing path,
or separately investigate Arrow expression filtering. These are hypotheses,
requiring new source-specific evidence. Changing thresholds or expanding cases
using this campaign's unused time would not validate this prototype.
