# Bounded batch-parallel SUM experiment: rejected

The prototype failed its frozen performance contract. It used real parallel
CPU work but confirmed slower dense execution with both four and eight CPUs.
The prototype was based on merged
`c9afa829ff025280ab346667582c62dc26a34785`. The archive delivery preserves
the target branch's runtime and API; its final review checks against
`d5e05c31110ebb1e6bfb8fa3d63359f50010f16e`. Prototype source remains
available at public
[fe0e86ad](https://github.com/wegamekinglc/calc-flow/commit/fe0e86ad220bd88ae3aeb1ae6c384e3a72e0a792).

The delivery preserves the measurement tools, CI tests, protocol, and evidence.
It does not enable the prototype's local partial workers.

## Fixed comparison and decision

| Case            | Baseline ms | Candidate ms | Change  | Verdict      |
|-----------------|-------------|--------------|---------|--------------|
| Dense, 1 CPU    | 353.969     | 361.425      | +2.11%  | inconclusive |
| Dense, 4 CPUs   | 376.225     | 435.468      | +15.75% | regression   |
| Dense, 8 CPUs   | 375.070     | 444.834      | +18.60% | regression   |
| Sparse, 1 CPU   | 33.471      | 34.572       | +3.29%  | inconclusive |
| Hot key, 8 CPUs | 112.540     | 127.855      | +13.61% | inconclusive |

These are pooled wall-time medians. Acceptance uses the maintained two-round
paired-median rule, with ten alternating pairs per round and a 5% regression
threshold. Both dense cases have round interval lower bounds above +5%.
No dense target has an `improved` verdict, so acceptance is false.
The hot-key median is also slower, but its second interval crosses the
threshold; its verdict remains inconclusive. Inconclusive results establish
neither equivalence nor a speedup.

The [sealed report](sum-batch-parallel-20261104/README.md) retains every round
interval, CPU/RSS result, and the complete 200-row observation table. Its nine
files are preserved exactly as generated, including the eight hashes in the
[manifest](sum-batch-parallel-20261104/evidence-manifest.json).

## What the profiles establish

The candidate has 2,302 selected user-CPU samples across five processing
windows. Every window contains five different TIDs attributed to the exact
SQL `ParallelCpuWork::run` symbol; that symbol accounts for 558 samples
(24.24%). The baseline has 1,792 samples and no such SQL worker samples.
The common graph filters each 100,000-row business input into one 80,000-row
SQL record: ten legacy 8,192-row chunks, admitting five units with eight
available logical CPUs.

Parallelism did not reduce elapsed work. In the dense eight-CPU case, total
process CPU P50 rises from 428.307 to 587.846 ms, while median average active
cores rises from 1.152 to 1.324. Whole-process peak RSS P50 rises from 247.67
to 271.14 MiB. RSS includes IPC preload, retained snapshots, and validation;
it is not an isolated operator working-set measurement.

Candidate inclusive samples include `group_rank` at 24.72%, local worker
execution at 24.24%, transaction finishing at 18.68%, commit at 12.21%, and
local-key/fragment destruction at 5.13%. Inclusive shares overlap and must
not be added. Exact coordinator helper symbols were inlined; the 33.84%
`other` bucket is not a separately measured merge phase, and missing symbols
do not establish zero cost. About 29% of leaf samples are unresolved in both
arms. All 3,752 baseline and 4,422 candidate raw timestamps match the imported
profile clock, with maximum error below ten nanoseconds.

The results support rejecting this implementation's additional local grouping
and reduction work. They do not isolate a complete wall-time breakdown or
prove that all possible batch-parallel SUM designs regress.

## Provenance, budget, and reproduction

The actual baseline product is
`af400b38511bc7f6255fd3549518095b51b20e0c`, whose compiled-source bytes match
merged `c9afa829`; it is not relabeled as a fresh merged build.
Its binary SHA-256 starts `be89d670`. The freshly built public candidate
`fe0e86ad` has binary SHA-256
`08fbff37a2e1c7ac51b08a117172c8b276bb0c0ab146de61836d74e30e43239d`.
Both arms use the audited common harness SHA-256
`657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`;
its public source is
[84186554](https://github.com/wegamekinglc/calc-flow/commit/84186554b078e25e71003a74fedc91c63b2b7f70).

The [frozen protocol](sum-batch-parallel-20261104.protocol.json), SHA-256
`7455ab73636085a84b2dbd983cc831fcfdc23a37b54f76d63fe3efdcc568b72c`,
records the five cases, source identities, affinity, sample counts, and budget.
The acquisition used the exact driver at public `fe0e86ad`. The archive copy
later adds a documented, rule-specific audit annotation for Git blob SHA-1
identity; its Python AST is unchanged. This later annotation does not replace
the original driver hash sealed with the measurements.
The [complete evidence](sum-batch-parallel-20261104/complete-evidence.json.gz)
includes compiler/build seals, actual core/example `fresh=false` records,
the new-module depfile proof, source inventories, commands, all snapshot/schema
oracles, and raw timestamp text. Original `perf.data` files remain at the
recorded local artifact paths with their sizes and hashes; the public sealed
set contains imported stack counts and clock evidence.

Candidate compilation took 631.784 seconds, with 0.979 seconds of read-only
build validation recorded separately. One acquisition took
224.693588034017 of 600 seconds, including fixture preparation, resource/source
guards, warmups, correctness, profiling, decoding, statistics, local
publication, and cleanup. It completed 200 formal observations and 20 warmups,
plus two profile processes of six jobs each, excluding the first job.
All 223 invocation receipts confirm their owned processes were reaped.
There were no retries or extra observations.

All streams use one DataFusion partition and identical batch/snapshot schedules.
Dense and hot-key streams retain all 21 complete snapshots. Hot-key input
maps 90 of every 100 business rows to an existing key while preserving all
10,000 seed groups. The common harness requires non-null input; nullable
semantics were tested separately in the rejected prototype's Rust suite.
The eight selected logical CPUs report four core IDs on WSL2, and host power
and load are not observable. Compare arms within this cohort; do not combine
these absolute timings with prior measurement cohorts.

## Next bounded hypothesis

The baseline profile still places 40.74% of inclusive samples in chunk updates,
26.45% in transaction finishing, 15.90% in commit/group destruction, and
13.06% in candidate construction. These are CPU samples, not independent
wall-time fractions or a promised speedup.

The next candidate should reduce per-group integer SUM state/result allocation
and destruction on the serial path before adding more workers. Inlining a
single integer state and deriving its result from that state could remove
nested scalar vectors, while preserving the existing checkpoint encoding,
null/wrapping semantics, and transactional output boundary. This remains an
unimplemented hypothesis. It needs focused allocation/correctness tests and
a separately frozen paired measurement task; this experiment's unused time
does not authorize another run.
