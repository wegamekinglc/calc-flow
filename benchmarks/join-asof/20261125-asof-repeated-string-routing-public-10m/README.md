# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `f8780771277e4dc78ed0efe8b9c04f31f7485d41` and candidate `84b731859d5d56dd911bdb84a870e87fa88f4665` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate reuses group routing when adjacent string-key morsels have equal contents. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

The preceding-product comparison intentionally uses the uninstrumented #737 wheel. Its recorded full tree is 5e197fffc00a17225dcc8fb8b17fea07a4de7208. The new candidate's product parent is 7fdc0dd99da897af40059f0ab5918153ac4f4f14. The archive records their merge base and source differences.

The acquisition stopped after 4 completed warmups and 21 formal single-arm samples. The stop interval recorded 365 major faults, 598 swapped pages and 1.0% sibling utilization. No paired verdict was established. The original archive retains any worker-local responses completed during cancellation.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 61 intervals and took 72.260 seconds within the controller and 72.41 seconds including interpreter startup and exit, against the 165-second limit. [Result and resources](result.json)

Each public comparison reserved 165 seconds including product qualification, installation, runtime and worker startup, fixture preparation, all oracle checks and cleanup. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

[Ten-million-row comparison against the 11.14 release](release-baseline/README.md)
