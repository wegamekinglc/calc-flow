# Materialized Arrow source: scoped validation

DAL-378 implements the materialized Table/RecordBatch portion of #440 on main
`658527f8354a92163362d51c9ec7de9f68552836`. It does not implement readers,
arbitrary C stream providers, persistent replay or a general connector bridge.

## Measurement plan (fixed before collection)

Compare the existing iterable adapter (baseline) with direct materialized input
(candidate) in the same sealed release wheel. This isolates the input path;
it is not a comparison between different Join implementations or historical
builds. Use the maintained `release_pairs.collect_case` and `evaluate_case`
contracts: two rounds of ten alternating AB/BA pairs per case, a fresh process
per invocation, complete output oracle and immutable input checks. Exactly two
cases, with no retries or expansion to a larger matrix:

- Projection: 512 int64 rows in 64 records of eight rows; verifies every doubled
  value and the complete output schema. This isolates small-batch bridge cost.
- Interval Join: 64 rows per side in eight records per side, one match per key,
  zero-width time bounds and disabled watermarks; verifies all 64 pair identities,
  values, multiplicities and the complete schema. This checks a stateful lifetime.

Each invocation includes fixture creation, graph compilation/preflight,
temporary managed checkpoint setup, all outputs, terminal job cleanup,
correctness checks and garbage collection. Controller wall time also includes
process startup, imports, release identity verification and exit. Record CPU
time and process peak RSS (both process-wide), inner lifecycle time, Arrow
allocator bytes before/after, native task count and Python source callback count.
Input allocation is intentionally inside the lifecycle. Logical edge budgets
are not asserted to bound RSS. Arrow allocated bytes must return to baseline.

Total collection budget: 600 seconds including all worker startup and cleanup;
stop on exhaustion or incompatible evidence. Build time is separate (estimated
3–10 minutes with compatible dependency cache, longer for a cold release build).
Use the existing paired-median +5% rule and report uncertainty. This focused
comparison does not establish a whole-suite performance verdict.

The worker is `benchmarks/arrow_source.py`. A controller calls it with `--case`,
`--side`, `--build`, and `--output`, then uses the controller's process wall time
as the observation's single `samples` value and assigns a unique `worker` ID.
The worker independently verifies the sealed wheel and native hash using
`benchmarks.native_release_identity.release_identity`.

## Functional evidence

Recorded separately from performance collection. The focused Python suite
covers public streams, sliced/multi-record/empty data, schema normalization and
input metadata immutability, row/byte splitting, default and explicit temporal
policies, bounded slow-consumer progress, early exit, metadata and binding reuse,
and foreign-buffer owner release on EOF, sink failure and cancellation.
An inline Rust test checks partial/unread record release, idempotent close,
metadata/schema preservation and rejection of a recovery cursor. Full regression
and coverage remain CI responsibilities.
