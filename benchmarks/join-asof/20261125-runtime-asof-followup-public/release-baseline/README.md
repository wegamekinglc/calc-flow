# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins a million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93` and candidate `6269bda3b1516902a88c5dd0c5beeb790720830d` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The change reuses the end of the visible right-hand history for repeated probes of an eligible key within one finalized selection. The measured path also includes state ingestion, Arrow output, queueing and delivery.

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The main comparison tests the individual change with matching Cargo.lock.

| Clock | Baseline  | Candidate  |
|-------|-----------|------------|
| P50   | 64.662 ms | 223.960 ms |
| P95   | 99.806 ms | 318.208 ms |

| Round | Paired median | Exact interval       |
|-------|---------------|----------------------|
| 1     | +212.62%      | [+199.48%, +314.02%] |
| 2     | +243.10%      | [+217.62%, +297.42%] |

The maintained verdict was **regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 16 intervals and took 20.224 seconds against the 120-second limit. [Result and resources](result.json)

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
