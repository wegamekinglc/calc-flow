# Public ten-million-row rolling SMA20

The scheduled managed single-thread SMA20 workload consumed ten batches of one million rows with 64 keys. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline 7012b5b409553df04e29e6d7f9a11abf919cbf93 and candidate 4e4ce4d75137ffca2f089c30efd29553ace92ff5 used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. The candidate reuses per-key entity identities during rolling slab processing. [Wheel identities](wheel-identities.json)

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The main comparison uses matching Cargo.lock.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 307.608 ms | 381.785 ms |
| P95   | 319.741 ms | 389.056 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | +23.78%       | [+22.80%, +27.08%] |
| 2     | +24.56%       | [+20.54%, +25.97%] |

The maintained verdict was **regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million ordered sequence identities and rolling means outside timing against the independent per-key NumPy oracle. The leading nineteen rows per key require missing results; each run contains 9,998,784 finite outputs and 1,216 warmup values. The largest observed absolute error was 8.527e-14.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker used one Tokio thread and one physical core, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 71 intervals and took 76.273 seconds within the controller and 76.34 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

Each independent comparison reserved 180 seconds including installation, identity qualification, each worker startup, fixture construction, full correctness verification, sampling and shutdown. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
