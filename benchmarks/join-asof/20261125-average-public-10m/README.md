# Public 10-million-row cumulative average

The scheduled managed four-thread average consumed ten batches of one million rows, with 64 keys and the original watermarks. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline `72072c1dc6c0874b79ccd8ce78103c97caec623d` and candidate `51f8b0cd5c2ac1e0577481f6fd4f7e4d290acfbe` used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. The candidate specializes the per-key cumulative average in the Rust rolling kernel. [Wheel identities](wheel-identities.json)

The main baseline wheel was built from commit 89cc36a88d7905c955ecdf7584b185c262b31e2b, whose full Git tree, release source digest and Cargo.lock match the main merge commit. The archived manifests report the genuine build commit.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 910.502 ms | 246.706 ms |
| P95   | 980.619 ms | 259.896 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | -73.06%       | [-74.31%, -72.04%] |
| 2     | -73.03%       | [-73.59%, -72.33%] |

The maintained verdict was **improved**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million ordered sequence identities and finite cumulative-average outputs outside timing against independent per-key NumPy cumsum/count at rtol=1e-10, atol=1e-10. The largest observed absolute error was 2.7e-12.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 71 intervals and took 76.084 seconds within the controller and 76.18 seconds including interpreter startup and exit, against the 240-second limit. [Result and resources](result.json)

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

[Whole-product comparison against 11.14](release-baseline/README.md)
