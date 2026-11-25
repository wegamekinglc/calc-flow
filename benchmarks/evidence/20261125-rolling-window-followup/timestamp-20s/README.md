# Timestamp rolling mean over twenty seconds

The 11.14 product 7012b5b4 and the selected main e1d9ccd3 each processed ten million input rows, 64 interleaved keys and a clock that skips one second every seven entity ticks. Each 20-second window excludes its lower bound. Both entry points use a checkpoint-disabled StreamingRunner. Each measures delivery through the final rows flushed by EOF. Native Rust times first source delivery through the last sink row. The installed package times ready enqueue through full Arrow delivery and concatenation. Both validate the physical sequence and all ten million window values. Timings are milliseconds. Each completed cohort has two rounds of ten formal pairs and four excluded warmups.

| Entry  | 11.14 p50 | Candidate p50 | 11.14 p95 | Candidate p95 | Verdict      |
|--------|-----------|---------------|-----------|---------------|--------------|
| Rust   | 428.919   | 452.651       | 447.974   | 469.129       | inconclusive |
| Python | 414.299   | 425.506       | 433.306   | 455.683       | inconclusive |

[Native Rust results](rust/native-summary.md) · [Installed Python results](python/README.md)
