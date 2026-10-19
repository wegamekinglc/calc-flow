# Join CPU pipeline

Continue the accepted `80d98ae089c8a317c2a884967b83728c94cb87ff`
implementation, which contains the current local main. Preserve immutable input,
logical ordering, exact pair multiplicity, explicit resource limits, error
precedence, cancellation ownership, and checkpoint recovery. Default runtime
memory remains unbounded. Compute capacity comes from the job's available CPU
capacity and workload, independently of output columns or checkpoint shards.

## Development

1. Add opt-in, job-owned diagnostic stage records for preparation, matching,
   materialization, assembly, state commit, work admission, and fallback.
   Collect at phase/task boundaries; no per-row shared counter. Async spans
   report wall time; only synchronous worker spans report thread CPU time.
   Default release does not include diagnostic hooks or public status fields.
2. Remove both ASOF row-gather column-count restrictions, including the
   single-column gate. Keep shared Arrow output and supported-type fallback.
   Audit reservation lifetimes and replace conservative bounds only with
   allocation-backed bounds. Fund concurrently live parts and concat output.
3. Match contiguous immutable ASOF prefix fragments in workers and construct
   local output plans there. Preserve local source namespaces in segmented
   plans, avoiding a serial row-wise source-ID rewrite. Owner assembly traverses
   fragments and actual span/source/key-owner metadata; fragmented input may
   still produce row-scale span metadata. Preserve null sentinels, schema,
   exact match counts, prefix ownership, and shared spans across fragments.
4. Parallelize interval retained preparation against owned immutable snapshots.
   Preserve source-order admission/ID/dictionary semantics; prepay worker
   credits deterministically. Prepare local retained selections and chunks, validate all results, and
   retain the existing output-acceptance/commit point. Append coordinates,
   expiration preparation and index updates remain on the owner; separate
   diagnostic stages identify their remaining cost.
   Reuse count window caches and make optional allocations yield to mandatory
   state admission. With a bounded or custom runtime pool, these new preparation
   tasks atomically reuse an already funded worker generation or fall back to
   serial preparation. Only the standard unbounded pool permits them to start
   a new generation. Do not clone the full resident index to obtain immutability.

Changes start with observed focused failures. Production work is divided between
ASOF/shared gather, interval state preparation, and diagnostics. One root process
owns native compilation/testing and acquisition. Output pipelining and concurrent
state mutation are outside this first delivery.

## Verification

Use deterministic capacity-planning checks and gated worker-overlap checks, not
timing assertions. Compare serial and parallel paths against an independent
small reference oracle. Cover hot keys, equal-time ties, interval endpoints,
null/late/probe-only admission, fragmented sources, row-ID overflow, deterministic
errors, explicit budget refusal, cancellation, failed output, early iterator
close, and restored legacy/columnar state. Check source owners and reservations
survive worker exit and are then released. Run focused Rust/Python checks,
necessary scoped lint/format checks, and an independent final specialist review.
Full workspace, connector, cross-platform, and coverage gates remain in CI.

## Fixed measurement contract

| Scenario | Rows per side | Batch rows | Logical CPUs | Purpose                         |
|----------|---------------|------------|--------------|---------------------------------|
| ASOF     | 1000000       | 1000000    | 1            | Serial version control          |
| ASOF     | 1000000       | 1000000    | 32           | Parallel preparation and output |
| Interval | 262144        | 262144     | 1            | Serial version control          |
| Interval | 262144        | 262144     | 8            | Intermediate capacity           |
| Interval | 262144        | 262144     | 32           | Full capacity                   |

Compare the sealed default baseline with an exact-source default candidate.
Reuse maintained EngineCase/oracles and prior sealed build infrastructure.
Keep both arms' fixture geometry, explicit limits (including ASOF's 1 GiB),
dependency versions and affinity identical. Each configuration has two rounds
of ten alternating AB/BA pairs and one excluded warmup per arm/round: 200 formal
observations and 20 warmups. Ready enqueue through expected Arrow delivery and
concat is timed; setup, oracle, terminal progression and cleanup still count
toward acquisition budget. Every sample has a fresh checkpoint directory.

Separate diagnostic builds use the same instrumentation settings before/after,
with their own feature/hash provenance. For each revision, the two 32-CPU shapes
have one diagnostic warmup and one observation: eight calls total. Never mix
diagnostic samples into default-release timings. Diagnostic/formal/cleanup
allowances are 90/480/30 seconds, totaling 600 seconds. Builds are recorded
separately (previous release builds took approximately ten minutes each).

Use maintained paired confidence rules: both lower bounds above +5% confirm
regression; both upper bounds below -5% confirm improvement. Report CPU scaling
ratios descriptively, independently of version deltas. Stop on budget exhaustion
or unsuitable environment, settle owned work, and preserve failed/incomplete
evidence. Do not add configurations or samples to obtain a favorable result.
ASOF 8-CPU, small-input timing, sustained state, and broad skew/type/checkpoint
performance are unmeasured by this feature set.
