# SMA20 over 10 million rows

| Entry point                 | Baseline | Baseline p50 (ms) | Candidate p50 (ms) | Paired verdict          |
|-----------------------------|----------|-------------------|--------------------|-------------------------|
| Rust, checkpointing off     | 11.14    |           382.178 |            469.151 | regression              |
| Rust, checkpointing off     | parent   |           459.179 |            458.191 | no-confirmed-regression |
| Python, managed checkpoints | 11.14    |           339.563 |            417.011 | regression              |
| Python, managed checkpoints | parent   |           420.491 |            423.239 | inconclusive            |

The focused test reduces recurrent page borrows from 4,093 to four across 4,097 rows and four keys. The checkpoint-disabled native workload exercises this path. The two-round timing comparison shows no material throughput gain against the parent. The installed default benchmark constructs a managed StreamingRunner with the default checkpoint policy and a 24-hour epoch interval. Its checkpoint-enabled state excludes the changed fast path. Both entry points retain a release-relative regression.

[Native measurements](rust/native-summary.md) · [Installed package measurements](python/README.md)
