# Public 10-million-row cumulative average

The scheduled managed four-thread average consumed ten batches of one million rows, with 64 keys and the original watermarks. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93` and candidate `51f8b0cd5c2ac1e0577481f6fd4f7e4d290acfbe` used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. The candidate specializes the per-key cumulative average in the Rust rolling kernel. [Wheel identities](wheel-identities.json)

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The main comparison tests the individual change with matching Cargo.lock.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 289.900 ms | 242.813 ms |
| P95   | 318.636 ms | 268.751 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | -15.29%       | [-21.43%, -13.30%] |
| 2     | -17.12%       | [-19.31%, -13.48%] |

The maintained verdict was **improved**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million ordered sequence identities and finite cumulative-average outputs outside timing against independent per-key NumPy cumsum/count at rtol=1e-10, atol=1e-10. The largest observed absolute error was 2.7e-12.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 55 intervals and took 59.558 seconds within the controller and 59.63 seconds including interpreter startup and exit, against the 240-second limit. [Result and resources](result.json)

Each independent comparison reserved up to 240 seconds; shared installation and qualification were bounded by 30 seconds. Shared qualification of all three installations took 8.091 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
