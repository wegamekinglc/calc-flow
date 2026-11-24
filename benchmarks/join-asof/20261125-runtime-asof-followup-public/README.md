# Public 4-thread zero-tolerance ASOF Join

The scheduled workload joins a million timestamped rows on each side across 64 symbols, with exact-time matches. The timing window covers ready enqueue through delivery and final Arrow concatenation. The elapsed values below are wall-clock measurements. Baseline `983f57435639037f40fb8a9d30aa1f33567ccd16` and candidate `6269bda3b1516902a88c5dd0c5beeb790720830d` used independently built portable release wheels, the same Rust compiler, Python interpreter, Python dependencies and four-thread settings. [Wheels and build identities](wheel-identities.json)

Left and right input events enter the managed stream during the timed interval. The change reuses the end of the visible right-hand history for repeated probes of an eligible key within one finalized selection. The measured path also includes state ingestion, Arrow output, queueing and delivery.

The acquisition stopped after 0 completed warmups and 0 formal single-arm samples. The stop interval recorded 348 major faults, 0 swapped pages and 0.0% sibling utilization. No paired verdict was established.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker retained four CPU cores from different groups, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 2 intervals and took 8.312 seconds against the 120-second limit. [Result and resources](result.json)

The feature reserved up to 120 seconds for each public comparison, 90 seconds for the native case and 30 seconds for shared setup and cleanup. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

[Whole-product comparison against the 11.14 release](release-baseline/README.md)
