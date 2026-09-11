"""Independent ready-runner CPU/RSS instrument; never a timing regression gate.

Run with the exact release package first on PYTHONPATH. Optional perf control
FIFOs enable samples only after runner readiness and disable them immediately
after Arrow output is materialized. Setup, correctness and shutdown are outside
that interval. RSS snapshots are process-wide observations, not allocator counts
or measurements of rolling state alone.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import platform
import re
import select
import sys
import time
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path


def prove_release(manifest: dict, native: Path) -> dict:
    """Reject stale/dev/unknown binaries before producing profile evidence."""
    with native.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (
        manifest.get("profile") != "release"
        or manifest.get("tracked_source_clean") is not True
        or re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_sha", "")) is None
        or manifest.get("native_sha256") != digest
    ):
        raise ValueError(
            "profiling requires a clean exact-source matching release binary"
        )
    return {"native": str(native), "native_sha256": digest, "build": manifest}


def read_memory_status(text: str) -> dict[str, int]:
    fields = {}
    for line in text.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, amount, unit = line.split()
            if unit != "kB":
                raise ValueError("unexpected /proc memory unit")
            fields[name] = int(amount) * 1024
    if fields.keys() != {"VmRSS:", "VmHWM:"}:
        raise ValueError("Linux VmRSS and VmHWM are required for this instrument")
    return {
        "rss_bytes": fields["VmRSS:"],
        "lifetime_peak_rss_bytes": fields["VmHWM:"],
    }


def _memory(phase: str, *, job=None, **facts) -> dict:
    import pyarrow as pa

    return {
        "phase": phase,
        "monotonic_ns": time.perf_counter_ns(),
        **read_memory_status(Path("/proc/self/status").read_text()),
        "python_arrow_pool_bytes": pa.total_allocated_bytes(),
        "status": None if job is None else job.status(),
        **facts,
    }


def _identities(args, manifest: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    identities = {
        "machine": {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "TOKIO_WORKER_THREADS",
                    "POLARS_MAX_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "power_mode": "not observable on this WSL instrument",
        },
        "dependency": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pyarrow": version("pyarrow"),
            "cargo_lock_sha256": manifest["cargo_lock_sha256"],
            "rustc": manifest["rustc_verbose"],
            "features": manifest["features"],
            "profile": manifest["profile"],
            "symbol_mode": manifest.get("symbol_mode", "shipped-strip"),
        },
        "workload": {
            "rows": args.rows,
            "scenario": args.scenario,
            "source_files": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "benchmarks/native_memory_profile.py",
                    "benchmarks/engine_comparison.py",
                    "benchmarks/engine_stream.py",
                    "benchmarks/warm_stream.py",
                    "benchmarks/rolling_indicator_comparison.py",
                    "scripts/benchmark_suite/catalog.py",
                )
            },
        },
    }
    return {
        kind: {
            "identity": identity,
            "sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        for kind, identity in identities.items()
    }


class PerfControl:
    """Own perf's acknowledged FIFO control descriptors outside timed work."""

    def __init__(self, control: Path | None, ack: Path | None) -> None:
        if (control is None) != (ack is None):
            raise ValueError("perf control and acknowledgement FIFOs are a pair")
        self.control = None if control is None else os.open(control, os.O_RDWR)
        self.ack = None if ack is None else os.open(ack, os.O_RDWR)

    def _command(self, command: bytes) -> None:
        if self.control is None:
            return
        os.write(self.control, command + b"\n")
        if not select.select([self.ack], [], [], 10)[0]:
            raise TimeoutError("perf did not acknowledge its sampling boundary")
        acknowledgement = os.read(self.ack, 4096)
        if acknowledgement not in (b"ack\n", b"ack\n\0"):
            raise RuntimeError(
                f"invalid perf control acknowledgement: {acknowledgement!r}"
            )

    async def set_enabled(self, enabled: bool) -> None:
        if self.control is not None:
            await asyncio.to_thread(self._command, b"enable" if enabled else b"disable")

    def close(self) -> None:
        for descriptor in (self.control, self.ack):
            if descriptor is not None:
                os.close(descriptor)


def _require_ready(source, sink) -> None:
    if not source.opened.is_set() or not sink.opened.is_set() or sink.rows:
        raise RuntimeError("profile must start on a ready runner with empty state")


async def _run_one(args, sample: int, control: PerfControl) -> dict:
    import pyarrow as pa

    from benchmarks.engine_comparison import EngineCase
    from benchmarks.engine_stream import _CollectSink, _ReadySource, stream_plan
    from calc_flow import (
        EdgeBudget,
        ManagedCheckpointRuntime,
        SinkBinding,
        SourceBinding,
        SourceProvidedWatermarks,
        StreamingRunner,
        StreamRuntimeConfig,
    )
    from scripts.benchmark_suite.catalog import BATCH_ROWS

    phases = [_memory("before_input")]
    # EngineCase owns the unchanged full-window workload and strict oracle.
    case = EngineCase(
        {"rows": args.rows, "scenario": args.scenario, "backend": "calc-flow-stream"},
        args.state_root / f"sample-{sample}",
    )
    # This instrument already owns its async loop; the synchronous adapter's
    # unused loop must never be driven recursively by EngineCase.close().
    case.loop.close()
    case.loop = None
    input_bytes = case.data.table.nbytes
    phases.append(_memory("input_and_oracle_prepared", input_logical_bytes=input_bytes))
    source, sink = _ReadySource(), _CollectSink(args.rows)
    job = await StreamingRunner(
        stream_plan(args.scenario),
        {"input": SourceBinding(source, watermark_policy=SourceProvidedWatermarks())},
        {"output": [SinkBinding.ordinary("profile", sink)]},
        ManagedCheckpointRuntime(args.state_root / f"sample-{sample}"),
        config=StreamRuntimeConfig(
            checkpoint_interval=timedelta(hours=24),
            edge_budget=EdgeBudget(max_rows=BATCH_ROWS, max_bytes=64 << 20),
        ),
    ).start_async()
    try:
        await asyncio.wait_for(source.ready.wait(), timeout=30)
        _require_ready(source, sink)
        gc.collect()
        phases.append(_memory("runner_ready", job=job))
        await control.set_enabled(True)
        try:
            started = time.perf_counter_ns()
            for event in case.events[:-1]:
                await source.push(event)
            await asyncio.wait_for(sink.complete.wait(), timeout=600)
            output = pa.concat_tables(sink.tables)
            finished = time.perf_counter_ns()
        finally:
            await control.set_enabled(False)
        phases.append(
            _memory("timed_output", job=job, output_logical_bytes=output.nbytes)
        )
        correctness = case.validate(output)
        if sink.rows != args.rows:
            raise RuntimeError("profile output rows differ from the workload")
        # Neither logical nbytes nor PyArrow's pool includes all Rust allocations.
        # Releasing these references leaves a live job, its rolling history and
        # allocator caches; RSS here must not be called retained state bytes.
        del output, event
        sink.tables.clear()
        case.close()
        del case
        gc.collect()
        await asyncio.sleep(0)
        phases.append(_memory("caller_input_and_output_released_live_job", job=job))
        await source.push(None)
        outcome = await asyncio.wait_for(job.wait_async(), timeout=600)
        if outcome.state != "completed" or sink.rows != args.rows:
            raise RuntimeError(
                f"profile failed during terminal verification: {outcome}"
            )
        phases.append(_memory("terminal_job", job=job))
        return {
            "sample": sample,
            "seconds": (finished - started) / 1e9,
            "timed_start_ns": started,
            "timed_end_ns": finished,
            "correctness": correctness,
            "phases": phases,
        }
    finally:
        await job.cancel_async()


async def run(args) -> dict:
    import calc_flow._native as native

    manifest = json.loads(args.build.read_text())
    proof = prove_release(manifest, Path(native.__file__))
    control = PerfControl(args.perf_control, args.perf_ack)
    try:
        samples = [
            await _run_one(args, sample, control) for sample in range(args.samples)
        ]
    finally:
        control.close()
    return {
        "contract": "calc-flow.native-cpu-rss-profile/1",
        "provenance": proof,
        "identities": _identities(args, manifest),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version,
        "python_executable": sys.executable,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": args.rows,
        "scenario": args.scenario,
        "instrument": "perf-cpu-clock-user-dwarf"
        if args.perf_control
        else "phase-rss-only",
        "environment_note": args.environment_note,
        "boundary": (
            "ready empty-state runner through enqueue/tasks/channels/rolling/"
            "watermarks/Arrow materialization"
        ),
        "limitations": [
            "CPU samples include perf enable/disable acknowledgement margins "
            "outside the recorded timer.",
            "perf samples user CPU only; elapsed send waits and kernel CPU "
            "are not CPU attribution.",
            "RSS is process-wide, includes imports/input/oracle/output/allocator "
            "caches, and is not live allocation bytes.",
            "VmHWM is a process-lifetime peak and can include setup or correctness "
            "work; it is not the timed-region peak.",
            "Phase RSS is discrete and does not prove a transient peak was observed.",
            "Python Arrow pool bytes exclude buffers allocated by Rust and NumPy.",
            "Timed-output status can precede final callback publication; "
            "terminal_job includes EOF/shutdown outside timing.",
            "This profiled run is diagnostic only; ordinary release timing "
            "requires the independent paired gate.",
        ],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--scenario", choices=("sma20", "dual_sma"), default="sma20")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--perf-control", type=Path)
    parser.add_argument("--perf-ack", type=Path)
    parser.add_argument("--environment-note", required=True)
    args = parser.parse_args()
    if args.rows <= 0 or args.samples <= 0:
        parser.error("rows and samples must be positive")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
