# Issue 431: Join chunk state and batch selection

Baseline: main `da004efe6c14ac90249dd8c33144ff6c6a30c1d7`, including #430.
The task uses `feature/dal341-join-chunk-state`; no shared identity or ASOF
ordering contract changes.

## Implementation

Online state stores columnar metadata in 1,024-row chunks, with shared payload
sources and stable chunk/row coordinates. Expiration stores sorted incoming
runs and one heap head per run. Removing a row releases its source reference;
empty chunks retire without relocating other rows or repairing dense indices.
Logical row/byte gauges remain incremental, including complete expiration.

Native probing retains count/limit/fill and stores opposite references plus an
incoming identity range or incoming runs. Fragment reservations cover both
selection arrays; fragment backing dies before its reservation. SQL fallback
builds retained key columns with batch take/interleave, without row-batch views.

Output uses take/interleave for single/multiple sources, including legacy and
nested payloads. Encoded children are compacted in whole columns and view data
is detached. Kernel errors propagate; there is no per-row slice/concat fallback.

Chunk metadata and expiration preparation precede output. Abandoned preparation
releases new reservations; state publication still follows all successful
emissions. The original unfunded legacy admission route remains available only
when optional acceleration refuses legacy input. Recovered funded expiration
state keeps mandatory admission. V1 fees, input immutability, physical row IDs,
ordering and checkpoint layouts 1/2/3 are unchanged. `StoredRow` remains a
transient admission/wire adapter so existing validation/error priority and wire
readers are preserved; it is absent from online retained storage.

## Verification scope

Select Join unit controls, SQL/property and state recovery integration tests,
scoped all-feature Clippy, formatting, complexity and schema/whitespace checks.
Full workspace regression, cross-platform and total coverage remain CI gates.

Allocation/RSS acquisition is limited to the maintained materialization
harness's four correctness cases, with a ten-minute collection budget, builds
separate. Development-profile allocation and RSS observations are diagnostics;
this work makes no release timing or R16 performance acceptance claim.
