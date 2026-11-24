# Public 4-thread static dimension Join

The scheduled workload joins a sealed 64-row dimension with a million fact rows. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `983f57435639037f40fb8a9d30aa1f33567ccd16` and candidate `271fc75c469f4b544d57237664b3d887a460e7bb` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

This fixture preloads and seals the 64-row right dimension before the timer starts. It exercises the product projection that reads a contiguous fact-side value range directly and copies sequence identifiers into the returned Arrow buffer. It also includes probing, multiplication, queueing and delivery.

| Clock | Baseline  | Candidate |
|-------|-----------|-----------|
| P50   | 58.443 ms | 55.461 ms |
| P95   | 62.904 ms | 60.162 ms |

| Round | Paired median | Exact interval    |
|-------|---------------|-------------------|
| 1     | -4.66%        | [-7.25%, +0.62%]  |
| 2     | -5.06%        | [-12.35%, -4.02%] |

The maintained verdict was **no-confirmed-regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 13 intervals and took 17.488 seconds against the 120-second limit. [Result and resources](result.json)

The feature reserved up to 120 seconds for each public comparison, 90 seconds for the native case and 30 seconds for shared setup and cleanup. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```

[Whole-product comparison against the 11.14 release](release-baseline/README.md)
