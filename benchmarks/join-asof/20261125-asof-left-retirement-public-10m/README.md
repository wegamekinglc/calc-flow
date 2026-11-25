# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins ten million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `cdd73704325223804789128e699a1f7d96e2410d` and candidate `dce77f04b24cd62ae5c725ba33c0e335b2637ec9` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The candidate retires a completely delivered left prefix as a unit. Both sides use ten one-million-row input batches and the maintained interleaved watermarks and sink-progress waits. The measured path also includes state ingestion, Arrow output, queueing and delivery.

The main baseline wheel was built from commit 3c8d6cddc1b69c71b42d2df04683664519c531e7, whose full Git tree, release source digest and Cargo.lock match the main merge commit. The archived manifests report the genuine build commit.

The acquisition stopped after 2 completed warmups and 0 formal single-arm samples. The stop interval recorded 22 major faults, 1 swapped pages and 64.6% sibling utilization. No paired verdict was established. The original archive retains any worker-local responses completed during cancellation.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 11 intervals and took 18.156 seconds within the controller and 18.25 seconds including interpreter startup and exit, against the 240-second limit. [Result and resources](result.json)

The stage reserved 240 seconds per public comparison, 40 seconds per native comparison and 40 seconds for shared overhead. Shared qualification of all three installations took 19.286 seconds in total. Full runtime and worker startup remained in each monitored acquisition. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

[Ten-million-row comparison against the 11.14 release](release-baseline/README.md)
