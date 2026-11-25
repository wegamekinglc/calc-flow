# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `84b731859d5d56dd911bdb84a870e87fa88f4665` and candidate `25e9595379e4cdaaa30329d7600497db0794018f` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate specializes contiguous ASOF payload compaction. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

Both comparison arms named as the candidate below really ran 25e95953. The later PR head 9c300939 extracted a Boolean predicate. The exact source diff from the measured head is preserved in the raw archive. No timing or binary-identity comparison of the later head was performed here.

The preceding-product comparison intentionally uses the uninstrumented #739 wheel. Its recorded full tree is 919f7cf5f800b839faaec2584e57e04f9cb8df88. The new candidate's product parent is 75b0f18d2c602c12ba02bd5da099d7a0df5f2b57. The archive records their merge base and source differences.

Each execution also checked physical output sequence against the full original sequence before the maintained sorted oracle. After EOF, both sides had accepted and ended with all ten million rows; ten million outputs matched, ten million right rows were evicted, the left and right state was empty and all three limit-failure counters were zero. Each complete native status is in its original worker response.

| Clock | Baseline    | Candidate   |
|-------|-------------|-------------|
| P50   | 1130.930 ms | 1048.618 ms |
| P95   | 1230.753 ms | 1093.964 ms |

| Round | Paired median | Exact interval    |
|-------|---------------|-------------------|
| 1     | -9.44%        | [-14.90%, -7.62%] |
| 2     | -6.07%        | [-7.01%, -3.50%]  |

The maintained verdict was **no-confirmed-regression**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated the full ten million sequence identities and finite values outside timing. Maximum absolute error was zero.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 90 intervals and took 101.017 seconds within the controller and 101.40 seconds including interpreter startup and exit, against the 165-second limit. [Result and resources](result.json)

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

[Ten-million-row comparison against the 11.14 release](release-baseline/README.md)
