# ASOF left retirement in the native watermark callback

The uninstrumented single-thread Rust operator accepted 100,000 rows per side, then settled a single watermark and materialized ten columns. Each comparison used four warmups and two rounds of ten alternating pairs. All 88 fresh executions validated the complete ordered output and state before and after the watermark. The exact source tree and compiled artifact receipts accompany each arm.

| Comparison                              | Baseline p50 / p95 | Candidate p50 / p95 | Paired medians       | Two-round verdict       |
|-----------------------------------------|--------------------|---------------------|----------------------|-------------------------|
| Preceding Arrow PR against this change  | 7.706 / 7.994 ms   | 7.023 / 7.408 ms    | −8.27% and −7.97%    | No confirmed regression |
| 2026.11.14 against this change          | 7.280 / 7.977 ms   | 7.413 / 7.737 ms    | +0.71% and −1.72%    | Inconclusive            |

For the preceding PR comparison, the two confidence intervals were [−14.10%, −5.31%] and [−11.69%, −4.52%]. For the direct release comparison they were [−3.04%, +8.05%] and [−4.12%, +4.51%]. The latter leaves the release parity question open. The start-to-watermark admission timers are secondary. The public Python 10-million-row job has its own independent comparison.

The branch starts at merged main `cdd73704`, with the exact tree of the preceding Arrow optimization `3c8d6cdd`. The candidate is `dce77f04b24cd62ae5c725ba33c0e335b2637ec9`. The release is `7012b5b409553df04e29e6d7f9a11abf919cbf93`. SHA-256 of the archive is `bbb937d126a5b42a5caa98b34186cc23930f5b5c4f2049fb203e413f369c79f7`. [Preregistered protocol](protocol.md) · [Machine-readable results](native-product-summary.json) · [Raw observations, build receipts, resource intervals and replay](asof-watermark-product-native.tar.gz)
