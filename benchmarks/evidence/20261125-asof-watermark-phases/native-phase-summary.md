# Native ASOF watermark comparison

The uninstrumented Rust watermark handler rose from **7.020 ms to 12.288 ms at p50** between the November 14 release and main. P95 rose from 7.698 ms to 13.342 ms. Each arm had 20 formal samples and two excluded warmups on CPU 2. Both independent rounds classified the change as a regression. Every trial validated all 100,000 output rows, all matches and the retirement of all 100,000 right rows.

The largest measured component was the Arrow output stage. Its average increase of 4.772 ms accounted for 89.3% of the full handler’s average 5.343 ms increase. It included output charging, materialization, batch assembly and edge byte checks. In the release, sizing performed while building row references is accounted with matching. Its final output-buffer reservation is accounted with Arrow on the calling task. The phase boundaries are documented in protocol.md.

| Exclusive stage       | Release p50 (ms) | Main p50 (ms) | Round 1 paired change (ms) | Round 2 paired change (ms) |
| --------------------- | ---------------- | ------------- | -------------------------- | -------------------------- |
| Matching and finality | 3.435            | 2.376         | −0.957                     | −1.100                     |
| Arrow output          | 2.744            | 7.522         | +4.652                     | +4.721                     |
| Left commit           | 0.012            | 1.015         | +1.079                     | +0.950                     |
| Right eviction        | 0.875            | 1.199         | +0.381                     | +0.265                     |
| Emit                  | 0.0016           | 0.0013        | −0.0002                    | −0.0003                    |
| Residual              | 0.017            | 0.215         | +0.226                     | +0.174                     |

Left commit and right eviction together increased by 1.332 ms on average. Matching improved by 0.949 ms on average. Each formal release trial recorded exactly one successfully admitted gather-service ticket in Arrow output. The parent await includes the time to receive its result. Each formal main trial recorded zero tickets.

The instrumented outer handler medians were 7.198 ms and 12.476 ms, respectively 2.52% and 1.53% higher than their independently measured uninstrumented controls. The six exclusive phase intervals summed to the measured inner wall time exactly in every diagnostic observation. The outer callback additionally includes ingress observation, formatting and appending the trace, and returning to the benchmark. The two groups completed in 5.102 s and 5.040 s. All sampled resource intervals passed the maintained gate.

Main was 72072c1dc6c0874b79ccd8ce78103c97caec623d and the release was 7012b5b409553df04e29e6d7f9a11abf919cbf93. Both arms ran the same Rust fixture, including two inputs of 100,000 rows over 64 keys, zero tolerance and one watermark. The fixture uses fixed-width fields and validates pre-watermark and final state. The measured Arrow stage combines charging and materialization, so their separate contributions require separate evidence.
