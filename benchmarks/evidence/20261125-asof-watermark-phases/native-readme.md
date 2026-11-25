# ASOF watermark phase evidence

The diagnostic hooks live in main.patch and release.patch and in their two detached source worktrees. The predeclared cases, time budget and exact measurement boundaries are in protocol.md. The independent retained clock group emits no files from inside the operator. For the traced wheel or native program set CALC_FLOW_ASOF_WATERMARK_TRACE to a new absolute JSONL path per process. Each operator finalization appends one record, including empty progress and EOF callbacks.

Build the native arms from both diagnostic commits with build-native.sh in the same scratch checkout and Cargo target used for the prior admission study. It copies the exact existing fixture and records Cargo receipts, dependency locks and SHA-256 digests. Controls come from the previous verified receipts in the sibling asof-admission-count-nov25 directory. The uninstrumented main reference and current main commit share complete Git tree cec8e19a4b73137b659dbdf616e349a6b58351c3.

Run observe.py calibration, followed by observe.py diagnostic on a quiet machine with no active wheel or native build, one group at a time. Original observation rows, per-process trace files, receipts, resource sampling and per-stage round results are saved in the two new obs-* directories. Each group is one-shot. For an offline replay, invoke observe.py replay obs-calibration, observe.py replay obs-diagnostic, then comparison.py. The verifier uses the maintained scripts in the source-main checkout. Place the containing repository's managed virtual environment on PATH; Python 3.11+ and psutil are required. No network is needed for replay.

The gather vector reports accepted service tickets in match, Arrow, left commit, right eviction, emit, and residual order. A ticket measures admission to the runtime service, and its elapsed wait is already contained in the phase. Worker execution times are never added on top. In Python a multi-source invocation can produce several progress notifications; first validate all callback deltas and full output rows across the entire trial before aggregating. Compare end-to-end timings using ordinary wheels separately.

comparison.json reports arithmetic-mean differences as an additive attribution including the gap outside the trace; median and p95 levels and paired median changes answer separate latency questions. Raw observations and original trace files remain authoritative.

Archive replay can use PYTHONPATH pointing at a checkout of 72072c1d for the maintained statistics and resource helpers. The serialized Git patches suffice to reproduce the instrumented source from each exact upstream revision without shipping a source worktree.

The two small Git bundles retain the exact temporary diagnostic commit identities and require only their published parent revisions. In a clone containing those parents, run git bundle verify on each file, then git fetch <bundle-path> HEAD. Build from the resulting exact commit IDs without needing to reconstruct local commit metadata.

To verify the archived run, extract the tarball, enter evidence-native, and run the following with a Python environment containing psutil. Set CHECKOUT to a clone at 72072c1d.

```sh
sha256sum -c manifest.sha256
PYTHONPATH="$CHECKOUT" python -B comparison.py
sha256sum -c manifest.sha256
```

Use native-phase-summary.md for the human-readable table and comparison.json for all percentiles, round-level confidence intervals and additive attribution. Product and diagnostic binaries are identified by SHA-256 in the receipts. Native binaries are excluded from the source archive. See protocol.md and build-native.sh for the original controlled build and sampling procedure.
