# Rust rolling SMA20

One single-threaded StreamingRunner, 10,000,000 rows in ten one-million-row Arrow inputs, 64 interleaved UTF-8 entities and one native watermark per batch, followed by EOF. Each fresh worker validates every emitted sequence and the raw timestamp, symbol, price and SMA20 result after stopping the operator clock. Each worker validates the complete five-field schema, all 1,216 warm-up nulls and all 9,998,784 valid means. The acquisition limit includes these checks, teardown and all four excluded warmups.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `4e4ce4d75137ffca2f089c30efd29553ace92ff5`.

Across the 20 formal pairs, P50 moved from 355.465 to 433.642 ms (+21.99%). P95 moved from 369.629 to 460.379 ms.

- Round 1 median paired change +20.92%, exact interval [+18.78%, +30.31%] (0.97852 coverage).
- Round 2 median paired change +21.97%, exact interval [+17.85%, +26.81%] (0.97852 coverage).

The maintained two-round verdict was `regression`. Elapsed acquisition time was 29.51 s on logical CPU 4, with 27 resource intervals including preflight.

## Main against candidate

Baseline `6916514e06dcf8361f44574e08874781cbc26c1d`, candidate `4e4ce4d75137ffca2f089c30efd29553ace92ff5`.

Across the 20 formal pairs, P50 moved from 453.048 to 423.273 ms (-6.57%). P95 moved from 468.879 to 451.002 ms.

- Round 1 median paired change -7.22%, exact interval [-9.50%, -0.45%] (0.97852 coverage).
- Round 2 median paired change -5.90%, exact interval [-9.49%, -0.84%] (0.97852 coverage).

The maintained two-round verdict was `no-confirmed-regression`. Elapsed acquisition time was 30.25 s on logical CPU 4, with 28 resource intervals including preflight.

Python public execution uses its own 10M-row single-thread experiment. These Rust data quantify the ordinary Rust runner at 10M rows, 64 entities and one core.
