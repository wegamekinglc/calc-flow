# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins a million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `72072c1dc6c0874b79ccd8ce78103c97caec623d` and candidate `3c8d6cddc1b69c71b42d2df04683664519c531e7` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The registered candidate changes the native ASOF path. The archived binding records its exact scope and merge base. The measured path also includes state ingestion, Arrow output, queueing and delivery.

The main baseline wheel was built from commit 89cc36a88d7905c955ecdf7584b185c262b31e2b, whose full Git tree, release source digest and Cargo.lock match the main merge commit. The archived manifests report the genuine build commit.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 209.028 ms | 192.522 ms |
| P95   | 279.274 ms | 240.625 ms |

| Round | Paired median | Exact interval    |
|-------|---------------|-------------------|
| 1     | -10.21%       | [-18.50%, -8.41%] |
| 2     | -3.94%        | [-16.35%, +0.92%] |

The maintained verdict was **no-confirmed-regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 19 intervals and took 23.027 seconds within the controller and 23.11 seconds including interpreter startup and exit, against the 120-second limit. [Result and resources](result.json)

The feature reserved up to 120 seconds for each public comparison and 60 seconds for shared installation, qualification and cleanup, under the joint 600-second Rust and Python limit. Shared qualification of all three installations took 7.219 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```

[One-million-row release comparison](release-baseline/README.md) · [Ten-million-row release comparison](release-10m/README.md)
