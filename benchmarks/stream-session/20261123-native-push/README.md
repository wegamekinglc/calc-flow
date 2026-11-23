# Native single-row session push

Both releases met the absolute target of a median at most 10 µs and a
P95 at most 25 µs on this host.

| Rust push latency (µs)       | Baseline | Candidate |
|------------------------------|----------|-----------|
| Pooled single-push P50       | 5.6915   | 5.1845    |
| Pooled single-push P95       | 12.3702  | 11.9143   |
| Median of 20 process medians | 5.6905   | 5.20525   |

The median of process medians fell 8.53%. The paired median changes
were −9.78% and −7.91%. Their exact 97.85% intervals were
[−11.95%, −2.12%] and [−11.43%, −4.72%]. Both exclude zero.
Classification under the registered 5% rule is
**no confirmed regression**. The upper endpoints both exceed −5%,
so this experiment does not establish a greater-than-5% improvement.

The experiment completed in 7.391 seconds of the 300-second budget,
including 0.95 seconds of untimed release-worker correctness checks.
Separate release builds took 16 minutes 36 seconds and 17 minutes
26 seconds. All 44 fresh processes completed their oracles, including
four excluded warmups. The host ran WSL2 on an Intel Core i9-13900HX.
Every worker ran on CPU 4 with one Tokio thread. Sibling CPU 5 reported
0% use at preflight and during acquisition. Both resource snapshots
passed.

This comparison exercises the public Rust `StreamSession::push` call on
an already warm dual-SMA session. Each call submits one row for the same
symbol with an explicit watermark, and returns an owned `SessionOutput`.
The clock excludes envelope construction and result validation. Each
process starts with 4,096 history rows and then pushes 64 individual
appends.

The candidate combines [#674](https://github.com/wegamekinglc/calc-flow/pull/674)
and [#675](https://github.com/wegamekinglc/calc-flow/pull/675) on one
baseline. With explicit watermarks, #674 removes construction of the
input endpoint path for each port validation. Its ordered-timestamp scan
does not execute. #675 changes how the queue computes the visible size
of eligible fixed-width Arrow columns. Strings still use Arrow's generic
slice-size calculation. The result covers this combined Rust path. The
individual PRs carry their own work-count evidence.

The offline reference has columns `event_time` (UTC microseconds),
`sequence` (0 to 4,159), `symbol` (S000), `price`, and `dual_sma_spread`.
Each window admits one observation, so all 4,160 prices yield a non-null
spread. A separate scalar check used the arithmetic mean over the last
up to 5 and 20 prices on every row. Both release workers verify the full
Arrow schema, the four passthrough columns, spread values and null mask,
unchanged original input and an empty finish. A deliberate null
substitution at the last row failed validation in the baseline worker.

See the [registration](protocol.md), [reproduction commands](reproduce.md),
[raw per-process timings](workers.jsonl), and [build, host and statistical
evidence](result.json). The reproduction commands recover the three
checksum-verified source files used for the recorded timings.

The public Python study uses the same product trees, Arrow fixtures and
64-push lifecycle under ordinary GC. [Python steady-session results](../20261123-public-steady-push/README.md)
