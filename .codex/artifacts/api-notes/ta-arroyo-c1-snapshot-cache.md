# C1 aggregate snapshot cache implementation

`IncrementalSql::aggregate_output_chunk` reads immutable
`GroupAggregates::result` values, including generic aggregate results and inline
integer SUM. Cache those raw aggregate arrays for every admitted aggregate;
leave HAVING, projection, ordering, and LIMIT on their existing paths.

Prepare touched chunk ordinals from the current `CandidateMap` before building
arrays. Reuse requires the same aggregate schema, first group identity, and row
count. New groups and growing tails rebuild their chunk. A prepared cache belongs
to `Transaction` and replaces the installed cache only in `commit`. Failed or
cancelled updates and failed emissions drop the prepared cache. Restore and reset
already construct a fresh native plan, so neither carries cached arrays.

Native state and `SnapshotKeys` use the compact state pool; scratch and output
use the engine pool. Keep this cache in the engine domain. The existing output
reservation funds new raw arrays before construction; successful preparation
splits their exact array charge into cache ownership. Reused raw arrays keep
their existing lease and their charge is deducted from the next output
reservation. Metadata is reserved before allocation. Installed cache entries,
including metadata, have an internal 8 MiB cap; a prepared replacement can retain
the old and new versions until commit.

Enable caching only for an engine pool reporting `MemoryLimit::Infinite`.
Finite and unknown pools retain the original path: an optional resident cache
must not reduce credit for mandatory input/state allocations or change which
budget error occurs first. Metadata refusal and insufficient transferable output
credit use the original array-building path without propagating a cache error.
This does not change compact-state budgets, checkpoint encoding, fingerprints,
or any public configuration. Engine accounting follows the existing runtime
ownership boundary; it does not promise accounting until the last external
Arrow buffer reference dies.

## Source verification

The clean-chunk identity test first failed at the expected aggregate-array
`Arc::ptr_eq` assertion, then passed with transactional reuse. The finite/unknown
pool control separately failed at three engine consumer registrations versus
the original two; sharing the existing scratch registration restored that
original protocol. Failed emissions preserve the committed row/byte quota
ledger; the existing test-only work counter continues to count attempted work.

All 16 focused C1 tests passed. The final source also passed 13 directly
affected key-output, seeded-aggregate, compact-budget, checkpoint-codec,
decimal, and ordered-LAST controls. One seed fixture initially retained the new
cache owner beyond its zero-reservation assertion; explicitly dropping that
owner with the outputs made the unchanged assertion pass on its focused rerun.
No production changes followed the combined run.

Scoped core library/test Clippy passed with warnings denied. Rust formatting,
the complexity ratchet, changed-function CCN at most eight, whitespace, and the
unchanged project schema were checked. Full regression/coverage CI, independent
specialist review, and paired performance acceptance remain outstanding. No
remote operation or performance measurement ran for this candidate.

## Focused harness proposal

Extend the existing `crates/calc-flow/benches/stream_sql_aggregate.rs` with two
explicitly selected cases; preserve its existing default inventory. Use the
entry-test fixture unchanged: 10,000 closed groups, two UTF-8 keys (including
the null key), Int64 SUM, and 20 appends of 1,000 existing keys. Sparse appends
touch slots 9000–9999; the dense control touches `(i * 37) % 10000`.

Each invocation warms its own plan, consumes and validates all 10,000 rows of
every append snapshot against an independent model, and retains the original
snapshot to check immutability. Report process time separately from warmup and
validation, and retain full invocation wall time and reservation observations.
Setup, warmup, correctness, workers, sealing, and cleanup all count toward the
600-second measurement budget; estimate builds separately. Compare main
`3ca4c019b46d75a58a18c1df6111ddc2f8262da4` against the reviewed candidate using
one identical sealed harness and the maintained `release_pairs` two-round,
ten-AB/BA-pair rule. Freeze the inner sample count and invocation report format
with the harness review before execution; do not reduce existing scheduled SQL
validator requirements. Finite/custom-pool acceleration is outside this
candidate's claim. No performance measurement has run for this implementation.
