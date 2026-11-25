# Rolling distinct count at 10 million rows

Single worker, 64 interleaved string keys, 64-row Float64 distinct-count window. Release 7012b5b4 and pre-PR parent 6f29e53028b6bbad7e20abba8da36dc933065c88 were compared separately against cb8adc7b638b0876c1d9ff248ec54e57c3568056. All timings are in milliseconds. Rust uses the native runner with checkpointing off and validates physical sequence, source fields, every UInt64 output, successful job completion and the final row count. Its clock starts at the first source poll and stops when the sink receives row 10,000,000, including the final group flushed at stream end. The independently installed Python products use the default managed stream and validate all rows and values after sorting by sequence. The Python clock covers ready enqueue through Arrow table concatenation and stops before submitting EOF. Each completed group contains four excluded warmups and two rounds of ten adjacent paired observations.

| Surface | Baseline | Baseline p50 | Candidate p50 | Baseline p95 | Candidate p95 | Paired verdict |
|---------|----------|--------------|---------------|--------------|---------------|----------------|
| Rust    | 11.14    | 903.085      | 763.183       | 914.181      | 791.066       | improved       |
| Python  | 11.14    | 850.305      | 722.119       | 880.043      | 749.603       | improved       |
| Rust    | parent   | 902.703      | 759.480       | 913.272      | 777.782       | improved       |
| Python  | parent   | 843.689      | 716.276       | 867.242      | 751.985       | improved       |

[Raw resource and correctness evidence, controllers, source identities and replay](unique64-10m-raw.tar.gz), SHA-256 4826dab8ba0ff721caa0bd5370fde3df8b438970ef06c5618311ec338aab3d9e.
