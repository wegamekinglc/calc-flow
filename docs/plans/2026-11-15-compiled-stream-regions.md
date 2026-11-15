# Compiled stream regions: phase 0 and phase 1

## Decision

Calc Flow will compile eligible streaming graphs into synchronous regions over
owned native state. A region uses one scheduler and bounded region-owned links
instead of creating one Tokio task or generic channel wakeup per internal edge.
An envelope is accepted before its downstream dispatch. A full-link drain can
run before the current callback returns, matching the existing bounded
cooperative handoff rather than adding a zero-handoff inline call. Existing
source and sink tasks remain the real asynchronous boundaries. Unsupported
graphs retain the current `StreamingRunner` path.

This is an incremental migration, not a second semantic runtime. Logical node
identity, metrics, progress, backpressure, cancellation, error attribution, and
emit-to-commit ordering remain compatible. Later phases cover shared group-by
state, native joins, changelog output, checkpoint-compatible state pages, and
key-partitioned scheduling; those phases are outside this PR.

## Phase 0 evidence contract

Use current `main` as the product baseline. Reuse the existing direct adapters:

- `.codex/artifacts/arroyo-comparison/measure.py` executes the pinned Arroyo
  WebSocket/JSON integration path with correctness checks.
- `benchmarks/rolling_indicator_comparison.py` executes TA-Lib with per-entity
  slicing, native calls, and timestamp-major output reconstruction.

Do not introduce duplicate adapters. Record the actual current source/build
identity, dependency versions, worker counts, batch boundaries, output protocol,
checkpoint configuration, and machine identity. Direct operator and complete
append-to-Arrow scopes remain separate. TA-Lib comparisons must preload enough
history to produce valid outputs; all-warm-up null outputs are not valid-output
throughput evidence.

The existing Arroyo cases cover stateless transforms and tumbling windows, not
unbounded row-wise rolling SMA. They must not be presented as rolling parity;
that external gate remains unavailable until the same semantics are admitted.

Initial workloads use batch sizes 1, 4, 16, 64, and 99 for warm rolling, plus a
large rolling control. Each feature measurement uses the repository's paired
AB/BA collector, fixed samples, ten-minute default budget, and maintained +5%
regression rule. No additional cases, samples, or retries are added to obtain a
favorable verdict.

## Phase 1 local rolling region

The first admitted region is:

```text
one source -> native rolling mean/dual-SMA -> exact projection -> one Arrow sink
```

The initial implementation is checkpoint-disabled and does not restore state.
It has one physical region task, preserves both logical operator identities, and
uses bounded local admission between the logical operators. Each upstream output
is accepted before its downstream dispatch; full-link drains preserve the
current rolling callback/commit boundary and downstream error identity. Full
local-edge budgets, FIFO control order,
entry acknowledgement, terminal cleanup, and failure diagnostics remain intact.

The implementation reuses existing operator dispatch and lifecycle ownership.
It must not add unsafe code, detach background work, or introduce an alternative
numerical implementation.

## Acceptance gates

For each selected sub-100 workload, the target is:

- at least 2x current-main throughput;
- at least 80% of the selected Arroyo and TA-Lib adapter throughput;
- no confirmed p95 regression above 5%;
- no confirmed regression on the large rolling control.

The PR must also preserve values, output order, logical metrics, cancellation,
error attribution, backpressure, and terminal cleanup. Full CI and final
specialist review are required. A test-only or compile-only result does not
establish performance acceptance, and inconclusive measurements are reported as
such rather than as improvement or equivalence.

## Source references

The design uses the pinned [TA-Lib SMA loop](https://raw.githubusercontent.com/TA-Lib/ta-lib/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/input/sma/sma.c),
[TA-Lib stream-state generation](https://raw.githubusercontent.com/TA-Lib/ta-lib/58160778ecaab21e90eafd6e5462f6e833100af9/ta_codegen/generator/src/backends/c_stream.rs),
[Arroyo chaining optimizer](https://raw.githubusercontent.com/ArroyoSystems/arroyo/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-datastream/src/optimizers.rs),
and [Arroyo chained collector](https://raw.githubusercontent.com/ArroyoSystems/arroyo/eac7541f061c467c729549caabd6658d00ce9b23/crates/arroyo-operator/src/operator.rs).
These sources inform the design; they do not substitute for current-product
measurements.
