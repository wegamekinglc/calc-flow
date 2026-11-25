# Rust rolling argmax256

One single-threaded StreamingRunner, 10,000,000 rows in ten one-million-row Arrow inputs, 64 interleaved UTF-8 entities and one native watermark per batch, followed by EOF. Each fresh worker validates every emitted sequence and the age of its oldest maximum in the 256-row window after stopping the operator clock. The wall limit includes this check, teardown and all four excluded warmups.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `5edc4cde06dadb6aa571bf5702b1a0573fabe5a7`.

Across the 20 formal pairs, P50 moved from 485.434 to 469.057 ms (-3.37%). P95 moved from 511.721 to 494.566 ms.

- Round 1 median paired change -3.23%, exact interval [-8.93%, -1.67%] (0.97852 coverage).
- Round 2 median paired change -3.12%, exact interval [-7.30%, -0.68%] (0.97852 coverage).

The maintained two-round verdict was `no-confirmed-regression`. Elapsed acquisition time was 36.66 s on logical CPU 2, with 29 resource intervals including preflight.

## Main against candidate

Baseline `596e28832c9829972a5311ef0a4c3d88f4885564`, candidate `5edc4cde06dadb6aa571bf5702b1a0573fabe5a7`.

Across the 20 formal pairs, P50 moved from 519.250 to 460.529 ms (-11.31%). P95 moved from 554.596 to 497.323 ms.

- Round 1 median paired change -12.75%, exact interval [-16.56%, -3.95%] (0.97852 coverage).
- Round 2 median paired change -12.23%, exact interval [-13.60%, -8.01%] (0.97852 coverage).

The maintained two-round verdict was `no-confirmed-regression`. Elapsed acquisition time was 33.01 s on logical CPU 0, with 30 resource intervals including preflight.

Python public execution uses its own 10M-row single-thread experiment. These Rust data quantify the ordinary Rust runner at 10M rows, 64 entities and one core.
