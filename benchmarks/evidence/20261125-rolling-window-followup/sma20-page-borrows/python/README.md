# Public ten-million-row rolling SMA20

The scheduled managed single-thread SMA20 workload consumed ten batches of one million rows with 64 keys. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline 4e4ce4d75137ffca2f089c30efd29553ace92ff5 and candidate 093724a219416a17427b2e6b060de46f28a665e9 used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. This managed workload uses the standard checkpoint-enabled runner. It exercises the unchanged path and serves as a product-level control. [Wheel identities](wheel-identities.json)

The baseline is the exact registered main 4e4ce4d75137ffca2f089c30efd29553ace92ff5. The candidate's immediate parent is 1b0ca88d3b4162ac9f17b3e43324fd3c3a3ea7ac; the archive records the common ancestor and all intervening source files.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 420.491 ms | 423.239 ms |
| P95   | 430.233 ms | 447.636 ms |

| Round | Paired median | Exact interval   |
|-------|---------------|------------------|
| 1     | -0.10%        | [-1.98%, +9.54%] |
| 2     | +0.63%        | [-1.99%, +4.05%] |

The maintained verdict was **inconclusive**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million sequence identities and rolling means after sorting by sequence outside timing against the independent per-key NumPy oracle. The leading nineteen rows per key require missing results; each run contains 9,998,784 finite outputs and 1,216 warmup values. The largest observed absolute error was 2.842e-14.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker used one Tokio thread and one physical core, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 79 intervals and took 84.896 seconds within the controller and 84.98 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

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

[Whole-product comparison against 11.14](release-baseline/README.md)
