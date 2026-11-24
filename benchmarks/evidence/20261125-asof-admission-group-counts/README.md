# ASOF admission group counts

The native Rust fixture has 100,000 rows per side over 64 interleaved keys with zero tolerance. Its primary clock ends after the last of four input handler calls. Each run verifies 100,000 pending left and 100,000 retained right rows, zero emissions, then advances a common watermark and verifies all output rows and final state accounting. Builds use identical fixture and adapter bytes, Rust 1.94, optimized code and one pinned thread. Main and candidate use the same lock. The release comparison uses the release product's own lock.

| Baseline → candidate        | Admission p50      | Admission p95      | Paired median change by round | Primary result          |
|-----------------------------|--------------------|--------------------|-------------------------------|-------------------------|
| Main a35ad02a → 89cc36a8    | 16.288 → 15.410 ms | 16.711 → 16.289 ms | −5.60%, −4.58%                | No confirmed regression |
| Release 7012b5b4 → 89cc36a8 | 16.546 → 14.871 ms | 17.073 → 15.716 ms | −10.37%, −11.28%              | Improved                |

Both main-relative round intervals were below zero, at [−7.63%, −2.39%] and [−7.90%, −1.51%]. The second round's median improvement was below the 5% effect floor. Against the release the intervals were [−14.06%, −7.63%] and [−13.34%, −7.91%]. Each independent comparison completed four warmups and twenty formal pairs. Acquisition took 5.079 s and 5.037 s. The three native builds took 541.3 s (main), 858.4 s (release), and 460.6 s (candidate).

The secondary clock measures only the subsequent watermark handler including settlement, Arrow output and eviction. Its p50 was 12.013 → 12.359 ms relative to main and 6.726 → 11.984 ms relative to release. Its p95 was 12.931 → 13.250 ms and 7.310 → 12.504 ms respectively. These descriptive timings locate a remaining gap after Rust admission on this native fixture. The public Python million-row workload has its own clock, schemas and four-thread runtime.

[Registration](protocol.md) · [Native summary](native-summary.json) · [Main samples and identities](obs-admit-a35ad02a0b-89cc36a88d/) · [Release samples and identities](obs-admit-7012b5b409-89cc36a88d/) · [Adapter](stage_admit.rs)

The focused regression uses 1,024 accepted rows on each side. Fresh input now performs 1,024 histogram lookups and 1,024 counter increments per side, compared to 2,048 on the original main. Recovery counts its distinct live payload and identity columns. A mixed restored run accepted at a 6,183-byte admission budget had held 6,279 bytes. It now holds 6,071 bytes; 112 bytes cover temporary locators. Tests exercise the one-byte-short failure, sparse fresh keys, partial commit, and explicit historical capacity recovery. The final tree passed 34 columnar and 24 stream state tests and all-feature crate Clippy.

[Test logs and scope](tests/) · [Implementation, public measurements and CI](https://github.com/wegamekinglc/calc-flow/pull/730)

To verify the samples, run the repository's managed Python interpreter from this directory with the harness checkout on PYTHONPATH. The committed summary was generated with the following command.

```bash
python summarize.py obs-admit-7012b5b409-89cc36a88d obs-admit-a35ad02a0b-89cc36a88d
```
