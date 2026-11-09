# Fixed MACD reference fixtures

Run from the repository root with Python 3.9+:

```bash
python crates/calc-flow/tests/fixtures/macd/generate.py
```

This fixed standard-library reference imports no production code or indicator
library. It independently composes three first-valid seeded EWMA recurrences
with separate float64 subtraction, multiplication and addition. Null/NaN input
skips fast/slow, a carried ready line still updates signal, and a valid NaN line
is missing only for signal's update. Infinity and generated NaN remain valid.

The five predetermined cases cover missing/carry, independent minima larger
than spans, finite overflow/Inf/NaN, spans above 2^53, signed zero and defaults.
Inputs and results are stored as raw nullable float64 bits. Comparisons require
exact validity and finite/signed-zero bits; generated NaN payloads are compared
by classification. Raw state-codec bit preservation is a separate obligation.
The provenance file records cases, normalized parameters, row counts and the
generated file checksum. No approximate or TA-Lib-compatible seed is selected.
The generator emits formatted Rust directly without starting another process.
