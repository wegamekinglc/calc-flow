# Rust cumulative average

One single-threaded StreamingRunner, 1,000,000 rows in one Arrow input batch, 64 interleaved UTF-8 entities and a source watermark followed by EOF. Each fresh worker validates every emitted sequence and average after stopping the operator clock. The wall limit includes this check, teardown and all four excluded warmups.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `51f8b0cd5c2ac1e0577481f6fd4f7e4d290acfbe`.

Across the 20 formal pairs, P50 moved from 29.110 to 25.534 ms (-12.29%). P95 moved from 30.890 to 32.444 ms.

- Round 1 median paired change -11.79%, exact interval [-14.17%, -5.13%] (0.97852 coverage).
- Round 2 median paired change -12.88%, exact interval [-19.85%, +4.33%] (0.97852 coverage).

The maintained two-round verdict was `no-confirmed-regression`. Elapsed acquisition time was 10.26 s on logical CPU 2, with 4 resource intervals including preflight.

## Main against candidate

Baseline `72072c1dc6c0874b79ccd8ce78103c97caec623d`, candidate `51f8b0cd5c2ac1e0577481f6fd4f7e4d290acfbe`.

Across the 20 formal pairs, P50 moved from 90.745 to 25.041 ms (-72.40%). P95 moved from 92.124 to 26.690 ms.

- Round 1 median paired change -72.74%, exact interval [-73.19%, -70.73%] (0.97852 coverage).
- Round 2 median paired change -71.86%, exact interval [-73.20%, -70.59%] (0.97852 coverage).

The maintained two-round verdict was `improved`. Elapsed acquisition time was 7.41 s on logical CPU 2, with 5 resource intervals including preflight.

Python public execution uses its own 10M-row four-thread experiment. These Rust data quantify one workload at 1M rows, 64 entities and one core.
