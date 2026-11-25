# Rust rolling SMA20

One single-threaded StreamingRunner, 10,000,000 rows in ten one-million-row Arrow inputs, 64 interleaved UTF-8 entities and one native watermark per batch, followed by EOF. Each fresh worker validates every emitted sequence and the raw timestamp, symbol, price and SMA20 result after stopping the operator clock. Each worker validates the complete five-field schema, all 1,216 warm-up nulls and all 9,998,784 valid means. The acquisition limit includes these checks, teardown and all four excluded warmups.

## Release against candidate

The acquisition stopped after 30.74 s. 42 completed workers passed the full ten-million-row schema, null-position and value checks. The two rounds recorded 10 and 9 of the required ten pairs. This acquisition has no two-round timing verdict.

The final 1.01 s interval recorded 306 major page faults, 0 swap pages, 28.85 GiB available, 5.36% host load and 14.50% sibling load. The maintained major page fault limit is fewer than 256 per interval.

## Main against candidate

Baseline `e87b138fc37aae4d21d8e498833dca5b9e3b99e1`, candidate `6916514e06dcf8361f44574e08874781cbc26c1d`.

Across the 20 formal pairs, P50 moved from 550.227 to 473.709 ms (-13.91%). P95 moved from 559.227 to 484.240 ms.

- Round 1 median paired change -13.39%, exact interval [-14.28%, -12.27%] (0.97852 coverage).
- Round 2 median paired change -13.88%, exact interval [-14.30%, -12.93%] (0.97852 coverage).

The maintained two-round verdict was `improved`. Elapsed acquisition time was 34.25 s on logical CPU 0, with 32 resource intervals including preflight.

Python public execution uses its own 10M-row single-thread experiment. These Rust data quantify the ordinary Rust runner at 10M rows, 64 entities and one core.
