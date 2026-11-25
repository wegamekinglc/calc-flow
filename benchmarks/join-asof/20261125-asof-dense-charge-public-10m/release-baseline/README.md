# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93` and candidate `ade0e864e7eaeeef5f55993cc1d91950efc52fdb` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate specializes projected resident charge accounting for fresh dense ASOF admission. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

This comparison tests the complete candidate product against the unmodified 11.14 release. Each native product uses its original Cargo.lock. Versions of form_urlencoded, url and libz-sys changed, and the core added a direct edge to the already-locked smallvec. The preceding-product comparison has matching Cargo.lock.

Each execution also checked physical output sequence against the full original sequence before the maintained sorted oracle. After EOF, both sides had accepted and ended with all ten million rows; ten million outputs matched, ten million right rows were evicted, the left and right state was empty and all three limit-failure counters were zero. Each complete native status is in its original worker response.

| Clock | Baseline   | Candidate   |
|-------|------------|-------------|
| P50   | 632.356 ms | 969.467 ms  |
| P95   | 655.504 ms | 1073.550 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | +55.76%       | [+49.00%, +62.57%] |
| 2     | +51.99%       | [+48.93%, +66.65%] |

The maintained verdict was **regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full ten million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 82 intervals and took 88.226 seconds within the controller and 88.29 seconds including interpreter startup and exit, against the 165-second limit. [Result and resources](result.json)

Each public comparison reserved 165 seconds including product qualification, installation, runtime and worker startup, fixture preparation, all oracle checks and cleanup. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
