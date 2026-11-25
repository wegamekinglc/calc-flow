# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93` and candidate `f8780771277e4dc78ed0efe8b9c04f31f7485d41` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate specializes index installation for ordered admission runs. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The preceding-product comparison has matching Cargo.lock.

Each execution also checked physical output sequence against the full original sequence before the maintained sorted oracle. After EOF, both sides had accepted and ended with all ten million rows; ten million outputs matched, ten million right rows were evicted, the left and right state was empty and all three limit-failure counters were zero. Each complete native status is in its original worker response.

| Clock | Baseline   | Candidate   |
|-------|------------|-------------|
| P50   | 621.647 ms | 1447.715 ms |
| P95   | 630.612 ms | 1486.138 ms |

| Round | Paired median | Exact interval       |
|-------|---------------|----------------------|
| 1     | +134.26%      | [+129.62%, +135.92%] |
| 2     | +134.18%      | [+122.43%, +140.52%] |

The maintained verdict was **regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full ten million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 87 intervals and took 92.670 seconds within the controller and 92.75 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

Each public comparison reserved 180 seconds including product qualification, installation, runtime and worker startup, fixture preparation, all oracle checks and cleanup. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
