## A4 Functional Preflight Evidence Review

**Verdict: Approve — existing functional preflight evidence only.**

This review covers the 26 frozen public left-only diagnostic cases and their
52 existing fresh native executions. The recorded callback durations remain
preflight data: there are **zero formal paired measurement observations**.
This approval does not establish a performance gain, FR16's full ASOF target,
managed durable restart, required CI success, or merge readiness.

### Exact Reviewed Inputs

- Build-only entry SHA256:
  `73a86e218e705ad8a6545e6667a225063c97d7be16341011006bd43fe45d6730`.
- Verified comparator inputs SHA256:
  `1d712a512e57fdd4d01a88ef9fffe75cbf2ed5eae1d7e7c76337bed97891b980`.
- Catalog SHA256:
  `5cd571a9821e367c2732d265e87b9206847246ca0346062b9d9c2007d501e364`.
- Completion SHA256:
  `d0462acd7e8e660d40a1d0ec12fa0b4e30ee9640419315dcb3c76a0ae9547fcb`.
- Root completion proof SHA256:
  `3bbd2bc6d4c5af025d16956070c8240aa904f47754979b870cb081107c740020`.
- Parent wrapper SHA256:
  `5f335247d78302eccab7a77ce372fe58c2140e0413718ab942b2836e556cfea6`.

The raw directory is
`target/issue363-asof-columnar-admission-perf/preflight-v1` at the repository
root. Reviewer proof files are under `target/issue363-a4-preflight-review`.
Each file in the reviewed raw directory is bound by relative path, full SHA256,
size and inode in the reviewer inventory. The proof covers **3,192 files**,
3,935,077,133 logical bytes and 1,058,723,267 bytes across 1,592 distinct
inodes. These are file-size inventories; they are not allocated disk blocks or
resident-memory measurements.

Baseline source export remains
`764843e634ae1a1da7a5b010095c3017349a25f4`, production
`9983cfd0c3d096da7a60c3a96ae03b785f5ae955`. Candidate source export remains
`fd73c1a91bb9d1b017c9de471a99789f867c3830`, production
`e3e3e11f7982ae7144d2b760b4b77bdddc20c060`. The comparator binaries, core
closures and build settings retain the identities approved in the
[build review](stream-asof-columnar-admission-build.md). Current stat identities
for all sealed tool inputs, both binaries, each closure manifest, 1,074 closure
artifacts and 98 closure directories match the captured identities. This review
reuses the prior full closure hash verification; it does not rebuild or claim
continuous cryptographic monitoring.

### Execution and Review Results

- All 26 `command-*.json` coordinator receipts have actual integer exit code 0,
  matching case identities and `--mode preflight`. Their stdout/stderr logs are
  retained and empty. Case start/finish times show sequential coordination.
- All 52 native `exit.json` receipts have actual integer exit code 0 and no
  recorded error. PIDs match `command.json`, raw `stdout.json`, accepted records
  and final case results. All 52 PIDs are distinct; all native PIDs and all 26
  recorded coordinator PIDs are currently absent from `/proc`.
- Native command binary identity and reported executable match the approved
  comparator for that side. Release debug assertions are disabled; the actual
  five-variable thread environment matches both the command and verified inputs.
  Actual worker timestamps are contained in the relevant coordinator interval,
  and baseline completion precedes candidate launch.
- Every case has exactly one completed preflight attempt, no warmups, no rounds,
  no performance statistics and accepted status. Every `accepted.json` record
  equals its final report and pair record; the raw worker stdout contains that
  same sample. Requests match the original case and owned evidence directory.
- Root observed parent session 89181 complete with exit 0. The completion file
  records all 26 cases and 17.365945594006917 seconds for the complete functional
  wrapper. This elapsed time is not an operator performance result.
- Reviewer command
  `python target/issue363-a4-preflight-review/verify_preflight.py > target/issue363-a4-preflight-review/verification.log 2>&1`
  completed with exit 0. It uses only stdlib file/parser/hash operations and
  executes only the reviewed pure catalog constructors and stdlib oracle.
  It does not import a native runtime or launch a child process.
- An additional stdlib check of the root completion proof against all actual
  case reports completed with exit 0; its result is preserved as
  `completion-proof-match.json`.

The first two reviewer helper attempts incorrectly compared public status
directly with persisted metrics, then omitted output-watermark normalization.
Both assertion logs are retained as
`verification-initial-checkpoint-normalization.log` and
`verification-initial-output-watermark-normalization.log`. Existing
`StreamAsofJoinOperator::capture_metadata` clears ingress watermark/idle/ended
fields and the output watermark in persisted metrics. After applying that
existing normalization in the reviewer helper, the corresponding comparisons
passed. These were reviewer helper errors; no native case was rerun and no
product failure is inferred. Root separately records its corrected initial
stdlib list-versus-string status parser in its completion proof.

### Functional Coverage

The original catalog is unchanged and equals the reviewed `catalog()` result:
eight row/worker threshold cases (including zero), three primary scales
(10k/100k/1M), nine key/sequence type cases, two input controls and four lifecycle
controls. Every standalone case JSON also matches that catalog entry.

For all **22 sample cases**, both versions complete the prebuilt public
`process_data` callbacks with the right input empty, all requested left rows
accepted and pending, no early output, and no unexpected late/duplicate/refusal
classification. The 1M case uses the actual 16 callbacks, **15 × 64,000 + 40,000**.
Fixed public state limits, full output schema and 64 continuation rows are
unchanged across versions.

The unchanged, build-bound Rust probe executes an independent ordinal oracle
over every output record. `check_record` compares the exact 12-column schema,
all six left arrays against generated expected values, and all six right arrays
against correctly typed NULL arrays. It checks complete row order rather than
a prefix or checksum alone. Signed/unsigned sequence extrema, Unicode wide
keys, nullable payload tags, large UTF8 and reversed incoming record order are
covered by their original catalog cases. Successful native exit and completed
Arrow archives corroborate the executed Rust assertions. This reviewer does
not import Arrow to decode the files independently.

Each sample has both live and nonterminal restored trajectories. The probe
captures admitted state, checks the prescribed prefix cuts, accepts 64 new
rows after those cuts, drains all rows at end and captures terminal state.
Final public logical rows, bytes, pending rows and identity-only rows are zero;
emitted and unmatched counts equal input rows plus 64.

The reviewer revalidated the approved `compare_pair` against all 22 actual
archive pairs. Admitted/primary/restored statuses, snapshot inline metadata,
charges, segment inventory, full segment bytes, output schema bytes, output
record ordering and output chunk inventory match across exact versions. All
**476 snapshot manifests** were checked against actual full segment sizes and
SHA256. Additional direct 1 MiB chunk comparisons checked **328 distinct inode
pairs**, including live-versus-restored outputs and snapshots within each
version. Archive segments are read-only regular files; any shared inode remains
inside one fresh worker's evidence directory. No hardlink sharing crosses
workers, cases or comparator versions.

For the **four lifecycle controls**, the actual two-sided outcomes and archived
control-state bytes match:

- Pre-cancel returns `Cancelled`, accepts no rows and emits nothing.
- Row refusal returns `AsofStateLimitExceeded`, with one state-limit failure.
- Byte refusal returns `AsofWorkspaceLimitExceeded`, with one workspace-limit
  failure.
- Pending cancel actually observes a first `Poll::Pending`, then returns
  `Cancelled` with empty public retained/admitted/output inventory.

All four outcomes explicitly mark `timing_observation: false`. Error paths do
not become callback timing samples. These observations do not identify a
private worker phase or prove exact reservation-pool refunds.

### Resource and Launch Gates

The actual 10k → 100k → 1M chain binds accepted reports with the same complete
case shape and verified comparator inputs. The 100k and 1M resource receipts
reference the precise previous-scale report and its current file identity.
The reviewer recomputed each estimate from the larger observed comparator HWM,
the actual row ratio and factor 1.25, and checked its 70% available-RAM limit.
Each comparator's actual launch guard rechecks the exact predecessor chain,
current sealed identities and its separately observed available-RAM limit
before native process creation, as ordered in the reviewed coordinator.

Every full-worker HWM remains below 70% of the recorded available RAM. The
maximum individual observed full-worker HWM is **460,083,200 bytes**; it agrees
with the root proof. All **228 recorded resource snapshots** report zero worker
swap. Samples retain `/proc` observations and affinity snapshots, explicitly
marked as incomplete execution traces. RSS, public logical state and thread
names do not prove private allocation balance, constructor thread attribution,
or a general maximum under concurrent workloads.

### Blocking Issues and Boundaries

No blocker was found in these existing functional preflight results. No source,
sealed tool, comparator, fixture or other agent's artifact was modified.
Rust builds and other product suites were not run by the reviewer; Python,
Studio and coverage gates are outside this handoff. No GitHub review, CI claim
or merge action was performed.

The cold public callback timer includes validation, identity proof, funding,
payload work, registration, gather dispatch/constructor/install and retirement
where applicable. These cases remain **left-only admission diagnostics**.
The frozen original A3 11-case full E2E fixture requires its separate execution
and evidence review. Operator snapshot restoration here does not exercise
managed v3 durable restart or application delivery. Private pool-refund and
precise worker-phase evidence remains in source tests, not inferred from these
public preflight outcomes.

Approval permits root to consider a separately granted, quiet, protocol-bound
measurement run. It does not grant execution itself. The planned two rounds of
10 ABBA pairs, all raw attempts, paired statistics and full E2E evidence are
still required before a performance conclusion. FR16's 1M/full-ASOF ≤100ms goal
remains unverified.

All reviewer-owned processes have completed: **0 running**. No native import,
build, probe execution, functional rerun or new measurement was performed by
this reviewer.
