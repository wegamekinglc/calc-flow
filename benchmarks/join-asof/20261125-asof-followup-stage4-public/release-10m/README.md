# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93` and candidate `3c8d6cddc1b69c71b42d2df04683664519c531e7` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Each side sends ten data batches of one million rows, interleaved with the scheduled watermarks and the sink-progress wait after each chunk. The registered candidate changes the native ASOF path. The archived binding records its exact scope and merge base. The measured path also includes state ingestion, Arrow output, queueing and delivery.

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The main comparison tests the individual change with matching Cargo.lock.

| Clock | Baseline   | Candidate   |
|-------|------------|-------------|
| P50   | 622.776 ms | 1651.002 ms |
| P95   | 641.964 ms | 1738.431 ms |

| Round | Paired median | Exact interval       |
|-------|---------------|----------------------|
| 1     | +165.10%      | [+159.79%, +178.83%] |
| 2     | +167.54%      | [+159.88%, +170.87%] |

The maintained verdict was **regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full ten million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 92 intervals and took 95.947 seconds within the controller and 96.02 seconds including interpreter startup and exit, against the 300-second limit. [Result and resources](result.json)

This separately registered ten-million-row release comparison reserved up to 300 seconds, including startup and cleanup. It reused the same three independently qualified installations as the one-million-row comparisons. Shared qualification of all three installations took 7.219 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
