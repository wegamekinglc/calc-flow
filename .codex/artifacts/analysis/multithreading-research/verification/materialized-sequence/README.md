# Materialized Arrow sequence stream input evidence

> Opt-in/additive Python API change; native extension unchanged (built from
> `main` @ `22f4b9a5`). Windows 11, Python 3.13, release native,
> `TOKIO_WORKER_THREADS=4`. One million rows per case.

## Hypothesis

A sequence of already materialized Arrow tables is currently only accepted as
an async iterable, so every batch crosses the Python source bridge. Treating a
non-empty list/tuple as one materialized input (schema-checked, concatenated
with shared chunk buffers) lets the existing native Arrow source replay it
without a Python source call per batch. From the fixed-cost profile, removing
that hop should dominate at 1,024-row batches and fade as batches grow.

## Method

`target/stream_mode_probe.py`: `table_input -> select(price*2+1) -> stream`
with 1,024/8,192/65,536-row batches totalling one million rows. Each process
runs one mode, one warmup, then three measured runs; result is the median.
Five AB/BA pairs per case execute the two modes in alternating order in fresh
processes. The 65,536-row case received ten additional pairs (15 total).

## Results (paired ratio async_iterable / materialized, >1 means materialized faster)

| Rows per batch | Batches | async p50 | materialized p50 | Paired speedup | Range |
|----------------|---------|-----------|------------------|----------------|-------|
| 1,024 | 1,024 | 0.3290 s | 0.0538 s | **6.64×** | 5.98–7.39 |
| 8,192 | 128 | 0.0672 s | 0.0367 s | **1.83×** | 1.65–2.20 |
| 65,536 | 16 | 0.0407 s | 0.0373 s | 1.07× (15 pairs) | 0.89–1.36 |

The fixed-cost profile measured ≈300 us/batch of Python-bridge overhead; at
1,024 rows that was most of the 447 us per-batch cost, so removing it explains
the 6.6× result. At 65,536 rows the hop is a small fraction of per-batch work,
so the two inputs are at parity: no confirmed regression across 15 pairs
(median 1.07×, range includes both directions).

## Correctness

- New Python tests compare the sequence output with the async-iterable output,
  assert the same number of record-boundary batches, and assert no
  `_IterableSource.next` calls for the sequence path (monkeypatched counter).
- Empty sequences, mismatched schemas and non-Arrow items are rejected;
  snapshot SQL keeps its single-complete-table rule.
- `pa.concat_tables` reuses chunk buffers (verified data-buffer addresses), so
  the sequence does not copy caller Arrow data.
- Existing materialized/iterable stream tests remain unchanged.

## Merge decision

**Merge.** The change is additive (new accepted input form), Python-only, keeps
native source and record boundaries, and shows a real 6.6× small-batch benefit
with parity at large batches. No format, fingerprint, recovery or public native
signature changes.
