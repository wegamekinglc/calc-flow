# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `dce77f04b24cd62ae5c725ba33c0e335b2637ec9` and candidate `f8780771277e4dc78ed0efe8b9c04f31f7485d41` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate specializes index installation for ordered admission runs. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

The preceding-product comparison intentionally uses the uninstrumented #733 wheel. Its recorded full tree is 880b62b310019aff0d36fa8eabd7bb4815a6bbd0. The new candidate's product parent is 8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae. The archive records their merge base and source differences.

Each execution also checked physical output sequence against the full original sequence before the maintained sorted oracle. After EOF, both sides had accepted and ended with all ten million rows; ten million outputs matched, ten million right rows were evicted, the left and right state was empty and all three limit-failure counters were zero. Each complete native status is in its original worker response.

| Clock | Baseline    | Candidate   |
|-------|-------------|-------------|
| P50   | 1651.559 ms | 1455.298 ms |
| P95   | 1764.807 ms | 1513.303 ms |

| Round | Paired median | Exact interval     |
|-------|---------------|--------------------|
| 1     | -13.47%       | [-16.15%, -10.48%] |
| 2     | -11.12%       | [-13.23%, -9.71%]  |

The maintained verdict was **improved**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full ten million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 114 intervals and took 121.447 seconds within the controller and 121.74 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

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

[Ten-million-row comparison against the 11.14 release](release-baseline/README.md)
