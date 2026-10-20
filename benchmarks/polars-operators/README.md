# Polars operator coverage: DAL-394

The addition covers cumulative average, 64/256-row argmax, 64-row distinct
count, exact-time cross-section mean and ten-second tumbling sum. Both Polars
references now support every scenario in the existing matrix; no newly
requested cell requires an unsupported marker. The existing 1M interval-join
cap is unchanged. This task changes benchmark adapters and evidence only.

## Measurement plan

Measure only the 84 added cells: six operators, seven original row scales
(10 through 10M), and the existing 1T/32T references. Baseline catalog:
`1085c651` (main at task start). Candidate: the clean implementation commit
recorded in `results.json`. These are external-reference coverage measurements,
with no baseline timing or speedup/regression verdict.

Use the maintained collector: two rounds, ten samples per round, one warmup
and a fresh process per round, with a fixed 1,800-second total collection
budget including wheel installation, process startup, fixtures, oracles,
warmup, validation and cleanup. No native build is needed: reuse the sealed
DAL-389 release wheel only after checking that the candidate's complete native
source-input hash and Cargo lock hash match the original build. Preserve its
original revision and wheel/native hashes separately from the new harness
revision and hash. No previous timing observations are reused.

## Semantics and timing

Every engine receives the same immutable Arrow fixture: up to 64 entities,
exact-eighth prices, UTC microsecond event times, unique sequence values and
64,000-row Arrow chunks. All new values are Float64. Every warmup and sample
is checked against the maintained independent NumPy/table oracles at
`rtol=atol=1e-10`. Tests additionally exercise repeated maxima, repeated values
leaving the distinct window, temporal/sequence ordering, negative timestamps,
exact microsecond bin boundaries and input immutability.

Average uses per-symbol cumulative sum/count ordered by event time and sequence.
Argmax returns the row lag of the oldest tied maximum in a partial/full frame.
Per-symbol rolling maxima supply the lookup price; forward ASOF from the frame's
starting row selects the earliest row with that price, which must lie in the
frame. This avoids expanding all rolling-group members. Unique count uses
`n_unique` in a native integer-index partial/full 64-row frame. CS mean
groups exact event times, and window sum groups symbol and UTC timestamps
truncated to epoch-anchored half-open ten-second bins. See
[suite semantics](../../docs/benchmark-suite.md#inputs-correctness-and-timing-boundaries).

The timer includes `plan.collect(engine="streaming").to_arrow()`, including
lazy execution, ordering, rolling groups, aggregation and Arrow output.
Input conversion and lazy plan construction, fixtures, warmup, process/IPC
overhead and correctness checks stay outside individual samples. Polars may
execute unsupported streaming operations in its in-memory engine; the report
records the requested collection engine, not a claim that every operator has
bounded streaming memory. All calculations use Polars native expressions.
