# Public ten-million-row rolling SMA20

The scheduled managed single-thread SMA20 workload consumed ten batches of one million rows with 64 keys. The timed interval begins with ready source enqueue and ends with complete Arrow delivery and concatenation. Baseline e87b138fc37aae4d21d8e498833dca5b9e3b99e1 and candidate 6916514e06dcf8361f44574e08874781cbc26c1d used independent portable release wheels, the same compiler, interpreter, locked Python dependencies and thread settings. The candidate routes distinct UTF-8 keys during multi-entity rolling slab admission. [Wheel identities](wheel-identities.json)

The baseline is the exact registered main e87b138fc37aae4d21d8e498833dca5b9e3b99e1. The candidate's immediate parent is 8b4fb0b5b809b04a5b3241f8b0e48fe57597b3ae; the four intervening files are evidence reports. The archive records their exact paths.

The acquisition stopped after 2 completed warmups and 19 formal single-arm samples. The stop interval recorded 340 major faults, 0 swapped pages and 11.0% sibling utilization. No paired verdict was established. The original archive retains any worker-local responses completed during cancellation.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker used one Tokio thread and one physical core, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 41 intervals and took 48.978 seconds within the controller and 49.10 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

Each independent comparison reserved 180 seconds including installation, identity qualification, each worker startup, fixture construction, full correctness verification, sampling and shutdown. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

[Whole-product comparison against 11.14](release-baseline/README.md)
