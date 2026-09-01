# Symbolic exponential indicators contract

Status: frozen for SCE-16 implementation.

## Scope

SCE-16 adds one native temporal primitive, unadjusted exponentially weighted
moving average (EWMA), and two Python conveniences: `ts.ema` is an exact alias
for `ts.ewma`, while `ts.macd` is the row-local difference of two EWMA nodes.
MACD does not add a second native accumulator or project declaration.

## Declaration and algorithm

The canonical symbolic node is `ewma@1(value, span, min_periods=1)`. `span` and
`min_periods` are positive integers. Unlike row frames, `min_periods` has no
upper bound. The native rolling declaration is:

```json
{
  "kind": "ewma",
  "primitive_version": 1,
  "input": "price",
  "output": "price_ema",
  "span": 3,
  "min_periods": 1
}
```

For each entity, let `alpha = 2 / (span + 1)`. Null and NaN inputs are ignored:
they do not increment the valid count and the current accumulator is emitted if
it has reached `min_periods`. The first valid sample seeds the accumulator
exactly. Each later valid sample applies, in canonical row order:

```text
average = average + alpha * (sample - average)
```

Positive and negative infinity are valid IEEE inputs and use that same formula;
the resulting IEEE value, including NaN, is retained. Before the first valid
sample, and while the valid count is below `min_periods`, the output is Arrow
null. Output type is nullable `float64` for every numeric input type.

`ts.ema(value, span=s, min_periods=m)` returns the exact same declaration node
as `ts.ewma`. `ts.macd(value, fast_span=f, slow_span=s, min_periods=m)` requires
`f < s` and expands to `ts.ewma(..., span=f) - ts.ewma(..., span=s)`.

## State, sharing, and recovery

EWMA requires constant state per `(entity, input column, span)`: a valid-sample
count and the current IEEE binary64 value. Outputs with the same key share one
accumulator even when `min_periods` differs. Different spans never share state.

Rolling layout v1 remains the frozen retained-history format for existing
primitives. Any declaration containing EWMA must use layout v2. Layout v2 keeps
the v1 history and buffered-row fields and adds deterministic EWMA rows ordered
by entity key and compiled accumulator ordinal. Each row stores the ordinal,
valid count, and exact binary64 value. Segment metadata, snapshot metadata, and
inventory descriptors must all equal the declaration's layout version and
schema fingerprint. Restore validates the group kind, range, uniqueness,
nonzero count, entity encoding, deterministic order, and configuration hash
before installing any state.

Batch and stream execution use the same ordered fold. Results are invariant to
input batch segmentation, and a checkpoint/restart continuation is identical
to an uninterrupted run. Existing layout-v1 declarations and fingerprints do
not change.

## Reference provenance

Finance-style vectors are independently derived from
`alpha-miner/Finance-Python` commit
`3e33d3e70c3458b4c6dcf76b88df6148229b402c`. That project supplies the span
formula, first-sample seeding, ignored-NaN behavior, and MACD composition. Calc
Flow's Arrow-null output, canonical ordering, immutable declarations, shared
native state, and durable layout-v2 recovery remain Calc Flow contracts. Tests
never import or fetch the external project.

The local reference mapping is explicit:

- Finance-Python `testMACD` compares native MACD with the difference of two
  `XAverage` accumulators; Calc Flow asserts the same composition by node
  identity, project fingerprint, and independently calculated output vectors.
- Finance-Python `testEMAMACD` applies another `XAverage` to MACD; Calc Flow's
  finance vector covers `ts.ema(ts.macd(...))` across two native rolling
  stages.
- Finance-Python MACD deepcopy/pickle tests motivate Calc Flow's stronger
  immutable-declaration and durable checkpoint/restart tests. Calc Flow does
  not expose mutable or pickle-backed accumulator objects.
