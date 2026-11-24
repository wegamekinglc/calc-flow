# Ordered ASOF history and contiguous Join input

ASOF compares main 983f57435639037f40fb8a9d30aa1f33567ccd16 with 6269bda3b1516902a88c5dd0c5beeb790720830d. Each side of the native admit-and-settle fixture has 100,000 rows across 64 ordered keys with zero tolerance. Every execution checks all output rows and retained plus evicted state. Two rounds of ten formal pairs and four warmups ran in separate processes pinned to one logical CPU.

| Native ASOF handler | Main      | Candidate |
|---------------------|-----------|-----------|
| p50                 | 32.820 ms | 28.261 ms |
| p95                 | 35.431 ms | 30.547 ms |

Both rounds met the maintained improvement rule. Paired changes were −13.34% [−18.13%, −9.20%] and −13.34% [−17.13%, −8.80%]. Three resource samples passed during the 5.228 s acquisition. The two optimized builds took 680.4 s and 574.6 s separately. The focused Rust work-count test recorded 8,064 → 2,352 right timestamp reads for 576 mixed dense and sparse probes.

[ASOF observations and build receipts](asof/) · [Pre-registered protocol](protocol.md) · [Oracle and interval replay](verify.py) · [Local and CI validation](asof/verification.txt)

For Join the public static workload multiplies each incoming price by its matched dimension factor. The compiled projection has a continuous input range for consecutive singleton matches. On main, the Rust work-count test read 128 coordinates and executed 24 Arrow takes for 64 outputs in eight budget-constrained chunks. At 271fc75c469f4b544d57237664b3d887a460e7bb the same test recorded 64 reads and eight takes. It validates the products, delivered sequence ownership, output order and budget. These counts measure native work. The independent wheel comparisons are recorded on the corresponding PR.

[Baseline work-count failure](join-work-counts/red.log) · [Candidate hot-path tests](join-work-counts/green-final.log) · [Fanout and clipping tests](join-work-counts/boundaries.log) · [Compiled projection tests](join-work-counts/projection.log)
