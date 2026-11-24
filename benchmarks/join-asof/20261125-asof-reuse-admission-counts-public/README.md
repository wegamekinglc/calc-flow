# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins a million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `a35ad02a0b065a91505b94304abb6c67e3ac5644` and candidate `89cc36a88d7905c955ecdf7584b185c262b31e2b` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The change reuses counts already computed for each key during admission and keeps allocated index capacity within its projected budget. The intermediate head was superseded before any warmups or samples. The measured path also includes state ingestion, Arrow output, queueing and delivery.

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 198.310 ms | 196.155 ms |
| P95   | 223.844 ms | 206.550 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | -4.54%        | [-10.92%, +11.37%] |
| 2     | -5.10%        | [-12.87%, -0.12%]  |

The maintained verdict was **inconclusive**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 18 intervals and took 22.843 seconds within the controller and 22.92 seconds including interpreter startup and exit, against the 120-second limit. [Result and resources](result.json)

The feature reserved up to 120 seconds for each public comparison and 60 seconds for shared installation, qualification and cleanup, under the joint 600-second Rust and Python limit. Shared qualification of all three installations took 9.889 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

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
