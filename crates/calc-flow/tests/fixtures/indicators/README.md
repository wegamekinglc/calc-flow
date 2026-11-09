# Fixed indicator reference fixtures

Regenerate from the repository root with Python 3.9+ and the pinned Rust toolchain:

```bash
python crates/calc-flow/tests/fixtures/indicators/generate.py
```

The script uses only Python's standard library and runs `rustfmt` on `golden.rs`.
It imports neither production transitions nor TA-Lib/other indicator libraries.
The fixed periods are 1, 2, 3, 7 and 32. Each fixture has 64 complete rows (63
updates), with deterministic prices on a 1/16 grid and fixed high/low/close
offsets. `provenance.json` records the formulas, checked predicate and generated
file checksum; generation is deterministic. Every emitted float literal must
round-trip to the original float64 bits, including negative fractional values
and signed zero.

The primary reference performs individual Python float64 operations in the
specified incremental seed and Wilder order, with explicit nonnegative zero
normalization and RSI zero/flat cases. Its expected output bits are compared
exactly. The independent Fraction reference seeds with a sum divided by count,
then uses the exact rational Wilder numerator and `100*gain/(gain+loss)`.
Before emitting fixtures it verifies periods 1..32, at most 64 updates, the
bounded dyadic input grid, high>=low, and every positive exact gain/loss/TR mean
in `[2^-20,2^13]` (zero also allowed). The checked accuracy limits are RSI
absolute error `100*256*2^-52` and ATR relative error `256*2^-52`.

This supplementary accuracy bound applies only to these qualified fixtures.
Subnormal/underflow, extreme finite values, missing-segment resets and synthetic
large counters have separately named IEEE/hand-derived Rust regressions; they
are not filtered into the Fraction set. In particular, period-2 RSI on
`[0, 2^-1074, 0]` requires exactly 100 at the last position, whereas the exact
rational result is 50.

The former Target-only generator allowed an all-zero integer sum divided by
count to become float0.0, contaminating a later supposedly exact Fraction mean.
The checked-in generator keeps Fraction zeros/sum starts and asserts their type
at every update. Correcting that reference changes only two supplementary RSI
literals (period 2, rows 54 and 62) by one ULP each. All input bits, primary IEEE
bits and accuracy tolerances are preserved.
