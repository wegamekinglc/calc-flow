# Rust rolling SMA20

One single-threaded StreamingRunner, 10,000,000 rows in ten one-million-row Arrow inputs, 64 interleaved UTF-8 entities and one native watermark per batch, followed by EOF. Each fresh worker validates every emitted sequence and the raw timestamp, symbol, price and SMA20 result after stopping the operator clock. Each worker validates the complete five-field schema, all 1,216 warm-up nulls and all 9,998,784 valid means. The acquisition limit includes these checks, teardown and all four excluded warmups.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `093724a219416a17427b2e6b060de46f28a665e9`.

Across the 20 formal pairs, P50 moved from 382.178 to 469.151 ms (+22.76%). P95 moved from 402.857 to 501.931 ms.

- Round 1 median paired change +23.02%, exact interval [+18.79%, +30.75%] (0.97852 coverage).
- Round 2 median paired change +22.95%, exact interval [+19.57%, +26.59%] (0.97852 coverage).

The maintained two-round verdict was `regression`. Elapsed acquisition time was 32.05 s on logical CPU 2, with 29 resource intervals including preflight.

## Main against candidate

Baseline `4e4ce4d75137ffca2f089c30efd29553ace92ff5`, candidate `093724a219416a17427b2e6b060de46f28a665e9`.

Across the 20 formal pairs, P50 moved from 459.179 to 458.191 ms (-0.22%). P95 moved from 472.059 to 478.345 ms.

- Round 1 median paired change +0.04%, exact interval [-2.61%, +2.12%] (0.97852 coverage).
- Round 2 median paired change -0.34%, exact interval [-1.89%, +2.59%] (0.97852 coverage).

The maintained two-round verdict was `no-confirmed-regression`. Elapsed acquisition time was 32.30 s on logical CPU 0, with 30 resource intervals including preflight.

Python public execution uses its own 10M-row single-thread experiment. These Rust data quantify the ordinary Rust runner at 10M rows, 64 entities and one core.
