# Native Rust ASOF admission

100,000 rows per side, 64 keys, zero tolerance and a single Tokio thread. Admission covers four process_data calls. Output validation follows the subsequent watermark for every row. Each group includes 20 formal adjacent pairs, four excluded warmups, all worker startups, fixtures, correctness checks and teardown.

## Release against candidate

Baseline `7012b5b409553df04e29e6d7f9a11abf919cbf93`, candidate `f8780771277e4dc78ed0efe8b9c04f31f7485d41`.

Admission P50 moved from 17.204 to 13.195 ms (-23.30%). P95 moved from 17.793 to 13.790 ms.

- Round 1 median paired change -23.94%, interval [-25.67%, -21.69%].
- Round 2 median paired change -23.50%, interval [-27.16%, -19.53%].

The maintained verdict for admission was `improved`. Watermark P50 moved from 7.055 to 6.789 ms and P95 from 8.119 to 8.016 ms. Acquisition took 6.30 s on logical CPU 2 with 3 resource intervals.

## Prior against candidate

Baseline `dce77f04b24cd62ae5c725ba33c0e335b2637ec9`, candidate `f8780771277e4dc78ed0efe8b9c04f31f7485d41`.

Admission P50 moved from 15.622 to 13.367 ms (-14.43%). P95 moved from 16.097 to 14.575 ms.

- Round 1 median paired change -12.23%, interval [-18.53%, -8.83%].
- Round 2 median paired change -14.61%, interval [-19.45%, -6.62%].

The maintained verdict for admission was `improved`. Watermark P50 moved from 7.282 to 7.220 ms and P95 from 7.909 to 8.010 ms. Acquisition took 5.28 s on logical CPU 2 with 3 resource intervals.

