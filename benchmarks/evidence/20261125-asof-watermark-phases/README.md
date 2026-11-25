# ASOF watermark settlement

The release is calc-flow-python-v2026.11.14 at 7012b5b4. The comparison base is main at 72072c1d. Its complete Git tree matches 89cc36a8, the build recorded in the uninstrumented baseline receipts.

The Rust fixture loads 100,000 rows on each side, in batches of 64,000 and 36,000. A single watermark at 1,563 microseconds finalizes all rows for 64 integer keys, at zero tolerance. The single-thread collector checks every output row across ten columns and verifies the operator has evicted all right payloads. The Python workload uses a four-thread public stream with 1,000,000 rows per side, 64 string keys, zero tolerance and two projected columns. Within each comparison, both versions receive identical input and run in alternating order.

The phase clocks assign elapsed time in the operator task to matching/output planning, Arrow output, left commit, right eviction, sink emission, or residual work. Release includes some sizing work inside match-plan construction. The parallel release path charges time waiting for worker results to its calling phase. Medians of individual phases need not sum to the median of the handler.

## Native product measurements

Both uninstrumented comparisons use head 3c8d6cdd. Against main, p50 was 12.697 → 8.117 ms and p95 was 13.188 → 8.662 ms. Paired median changes in the two independent rounds were −35.87% and −35.98%, both improved. Against the November 14 release, p50 was 7.421 → 8.309 ms and p95 was 7.753 → 8.809 ms. Paired changes were +12.59% and +13.27%, both regressions. All 88 executions validated every output and eviction. The two comparisons used separate pinned cores, the same verified fixture, and fresh processes. The candidate build took 579.3 seconds separately. [Product raw observations, receipts, and verifier](asof-watermark-product-native.tar.gz) · [Per-round outcomes and secondary admission medians](native-product-summary.json)

## Diagnostic and public evidence

See the [native phase comparison](native-phase-summary.md) and [instrumented Rust raw data and replay instructions](native-readme.md). The public wheel phase comparison has its own row, process and timing provenance. [Python diagnostic](https://github.com/wegamekinglc/calc-flow/blob/91c75c4d921581f2687f1b5f90fba1bc53c21cfd/benchmarks/join-asof/20261125-asof-public-watermark-phases/README.md)
