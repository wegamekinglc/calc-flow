# Public 4-thread static Join on one million rows

The scheduled case was `engines/1000000/calc-flow-stream/join`. The baseline was `92230f68c9024a053f2e7c74185c3a98f5003917` and the candidate was `0b6f2971ac2bedd31fe515f099bd2342ec5e8d34`. Both release wheels were built independently with the same portable release command, compiler and Python dependency lock. Each native build used its registered Cargo.lock. See [wheel identities and per-file digests](wheel-identities.json).

The right side was a sealed 64-row dimension. The clock started when the ready one-million-row fact input was enqueued and ended after delivery and Arrow concatenation. Each execution validated the complete sequence and price-product columns against the independent oracle, including all finite values.

The registered resource or completion gate stopped this attempt after 8.467 seconds. 0 warmups and 0 formal single-arm samples completed. The last unsuitable interval recorded 323 host major faults, 0 swapped pages and 0.0% sibling activity. The paired gate has no verdict. [Stop details and resource counters](result.json)

The original partial samples, qualification records and preregistration remain in the [evidence archive](raw-evidence.tar.gz).

[Independent comparison against the 11.14 release](release-baseline/README.md)

The original two Rust clocks used 10.190 seconds. The four public acquisitions used 8.467, 32.612, 18.593 and 20.995 seconds respectively, including the stopped static main comparison. Charging the full 60 seconds reserved for shared qualification and cleanup yields 150.857 seconds against the joint 600-second budget.
