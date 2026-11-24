# Public 4-thread ASOF Join on one million rows

The scheduled case was `engines/1000000/calc-flow-stream/asof_join`. The baseline was `7012b5b409553df04e29e6d7f9a11abf919cbf93` and the candidate was `b076010b9df3ffe8065360274589f5d697675dc8`. Both release wheels were built independently with the same portable release command, compiler and Python dependency lock. Each native build used its registered Cargo.lock. See [wheel identities and per-file digests](wheel-identities.json).

Both inputs contained the same million ordered rows distributed over 64 symbols, with zero tolerance. Each published left sequence had to match the exact right price at the same timestamp. The clock measured ready enqueue through delivery and final Arrow concatenation. The full sorted sequence and value columns were verified against the independent oracle.

| Clock | Baseline  | Candidate  |
|-------|-----------|------------|
| P50   | 59.833 ms | 260.033 ms |
| P95   | 62.299 ms | 281.384 ms |

| Round | Paired median | Exact interval       |
|-------|---------------|----------------------|
| 1     | +338.95%      | [+312.91%, +360.18%] |
| 2     | +334.57%      | [+308.60%, +351.46%] |

The maintained verdict was **regression**. Both rounds completed ten alternating AB/BA pairs. All four excluded warmups and 40 timed executions validated the exact million-row output. [Statistics, resource counters and elapsed time](result.json)

This independent comparison asks whether the candidate still exceeds the release product's latency gate. Each endpoint uses its own exact Cargo.lock. Three native transitive versions changed (form_urlencoded, url and libz-sys), and the core added a dependency edge to the already-locked smallvec. The Python lock, workload, compiler, flags and timing boundary agree. The original main-to-candidate comparison isolates the current code change.

The measurement host ran Linux under WSL2. Four distinct exposed physical core groups were selected at the three-second preflight. The worker retained four Tokio threads and four Polars threads. Each one-second interval checked every selected core's SMT siblings, host utilization, memory, swap and major faults using the same predicate as the maintained resource guard. Windows hypervisor scheduling is an external source of timing noise.

The public acquisition consumed 20.995 seconds of its preregistered 120-second wall limit including setup and cleanup. Release builds were accounted separately. The source identities, toolchain, commands, relative timelines, worker output oracles and all IPC records are in the [original evidence archive](raw-evidence.tar.gz). The [receipt](receipt.json) verifies every member byte for byte.

To recompute the paired verdict, extract the archive at the baseline checkout and run the following command from the checkout root after installing the benchmark dependencies.

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
row = json.loads(Path('observation/samples.json').read_text())
print(comparison(row))
PY
```

The original two Rust clocks used 10.190 seconds. The four public acquisitions used 8.467, 32.612, 18.593 and 20.995 seconds respectively, including the stopped static main comparison. Charging the full 60 seconds reserved for shared qualification and cleanup yields 150.857 seconds against the joint 600-second budget.
