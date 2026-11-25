# Rust rolling duration mean

One single-threaded StreamingRunner, 10,000,000 rows in ten one-million-row Arrow inputs, 64 interleaved UTF-8 entities and one native watermark per batch, followed by EOF. Each fresh worker validates every emitted sequence and the raw timestamp, symbol, price and duration mean result after stopping the operator clock. Each worker validates the complete five-field schema, all 10,000,000 non-null means and the exact number of lower-edge exclusions and double evictions. The acquisition limit includes these checks, teardown and all four excluded warmups.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `e1d9ccd3fd72cb20668e06891cb49e1fbfd4d75a`.

Across the 20 formal pairs, P50 moved from 428.919 to 452.651 ms (+5.53%). P95 moved from 447.974 to 469.129 ms.

- Round 1 median paired change +4.59%, exact interval [+2.43%, +5.76%] (0.97852 coverage).
- Round 2 median paired change +6.15%, exact interval [+3.74%, +8.27%] (0.97852 coverage).

The maintained two-round verdict was `inconclusive`. Elapsed acquisition time was 33.33 s on logical CPU 0, with 30 resource intervals including preflight.

Python public execution uses its own 10M-row single-thread experiment. These Rust data quantify the ordinary Rust runner at 10M rows, 64 entities and one core.
