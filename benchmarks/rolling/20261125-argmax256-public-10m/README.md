# Public ten-million-row rolling argmax

The scheduled managed single-thread argmax256 workload consumed ten batches of one million rows with 64 keys. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline 596e28832c9829972a5311ef0a4c3d88f4885564 and candidate 5edc4cde06dadb6aa571bf5702b1a0573fabe5a7 used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. The candidate specializes grouped Float64 row-window argmax in the Rust rolling kernel. [Wheel identities](wheel-identities.json)

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 449.004 ms | 400.742 ms |
| P95   | 501.851 ms | 450.133 ms |

| Round | Paired median | Exact interval    |
|-------|---------------|-------------------|
| 1     | -11.50%       | [-14.18%, -6.11%] |
| 2     | -10.93%       | [-17.60%, -0.64%] |

The maintained verdict was **no-confirmed-regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million ordered sequence identities and finite rolling argmax ages outside timing against the independent per-key oracle, with zero numerical error required. The largest observed absolute error was 0.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker used one Tokio thread and one physical core, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 150 intervals and took 156.437 seconds within the controller and 156.50 seconds including interpreter startup and exit, against the 190-second limit. [Result and resources](result.json)

Each independent comparison reserved up to 190 seconds; shared installation and qualification were bounded by 20 seconds. Shared qualification of all three installations took 10.220 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

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
