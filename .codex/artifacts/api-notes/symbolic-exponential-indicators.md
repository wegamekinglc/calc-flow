# Symbolic exponential indicators API note

Status: frozen for SCE-16 implementation.

## Python

```python
ts.ewma(value, *, span: int, min_periods: int = 1) -> ColumnExpr
ts.ema(value, *, span: int, min_periods: int = 1) -> ColumnExpr
ts.macd(
    value,
    *,
    fast_span: int = 12,
    slow_span: int = 26,
    min_periods: int = 1,
) -> ColumnExpr
```

`ema` is a declaration-identity alias for `ewma`. `macd` is symbolic
composition and therefore benefits from ordinary common-subexpression and
rolling-state sharing. All arguments are keyword-only after `value`.

Stable validation paths are:

- `calc_flow.symbolic.ts.ewma.span`
- `calc_flow.symbolic.ts.ewma.min_periods`
- `calc_flow.symbolic.ts.macd.fast_span`
- `calc_flow.symbolic.ts.macd.slow_span`
- `calc_flow.symbolic.ts.macd.min_periods`

`fast_span >= slow_span` is an `invalid_literal` at the `fast_span` path.

## Rust/project v3

`RollingOutputSpec::Ewma` adds `primitive_version`, `input`, `output`, `span`,
and `min_periods`. `RollingSpec.state_layout_version` accepts v1 and v2, but an
EWMA output requires v2. The public constants retain v1 for existing rolling
lowering and expose v2 explicitly for exponential state.

No public Rust `Macd` type, project node, or runtime operator is added.
