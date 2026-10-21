"""Bounded, correctness-gated live WebSocket comparison; see REPORT.md."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import signal
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

BASE = datetime(2026, 1, 1, tzinfo=UTC)
CHUNK = 512
CF_REF = "dab98a559314bd46d866ed7a6489abcdd3810afe"
ARROYO_REF = "4ef15f6336a596651f10cf27c1ee29c29bf81577"


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(rows: int) -> tuple[list[bytes], list[dict[str, int]]]:
    records = [
        {
            "id": i,
            "k": i % 16,
            "ts": (BASE + timedelta(seconds=2 * (i // CHUNK))).isoformat(),
            "x": (i * 17) % 101 - 50,
        }
        for i in range(rows)
    ]
    marker = {
        "id": -1,
        "k": -1,
        "ts": (BASE + timedelta(seconds=2 * (rows // CHUNK + 1))).isoformat(),
        "x": 0,
    }
    chunks = [records[start : start + CHUNK] for start in range(0, rows, CHUNK)]
    frames = [
        (
            "\n".join(json.dumps(row, separators=(",", ":")) for row in chunk) + "\n"
        ).encode()
        for chunk in [*chunks, [marker] * CHUNK]
    ]
    return frames, records


def oracle(case: str, records: list[dict]) -> list[dict[str, int]]:
    if case == "stateless":
        return [
            {"id": row["id"], "y": row["x"] * 2 + 1} for row in records if row["k"] < 8
        ]
    grouped: dict[tuple[int, int], tuple[int, int]] = {}
    for row in records:
        group = (row["id"] // (2 * CHUNK), row["k"])
        total, count = grouped.get(group, (0, 0))
        grouped[group] = (total + row["x"], count + 1)
    return [
        {
            "ws": int(BASE.timestamp() * 1_000_000) + w * 4_000_000,
            "k": key,
            "total": total,
            "cnt": count,
        }
        for (w, key), (total, count) in sorted(grouped.items())
    ]


def sql(case: str, endpoint: str) -> str:
    query = (
        "SELECT id, x * 2 + 1 AS y FROM events WHERE id >= 0 AND k < 8"
        if case == "stateless"
        else """SELECT CAST(EXTRACT(EPOCH FROM w.start) * 1000000 AS BIGINT) AS ws,
        k, total, cnt FROM (SELECT TUMBLE(INTERVAL '4 seconds') AS w,
        k, SUM(x) AS total, COUNT(*) AS cnt FROM events WHERE id >= 0 GROUP BY 1, 2)"""
    )
    return f"""CREATE TABLE events (
      id BIGINT NOT NULL, k BIGINT NOT NULL, ts TIMESTAMP NOT NULL, x BIGINT NOT NULL,
      WATERMARK FOR ts AS (ts - INTERVAL '1 microsecond')
    ) WITH (connector = 'websocket', endpoint = '{endpoint}', format = 'json',
      framing = 'newline');
    {query};
    """


async def cf_worker(case: str, endpoint: str) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.json as paj

    import calc_flow as cf
    import calc_flow._native as native

    native_hash = digest(Path(native.__file__))
    if os.environ.get("COMPARISON_NATIVE_SHA256", native_hash) != native_hash:
        raise RuntimeError("loaded native binary differs from the controller seal")
    print(
        json.dumps({"native_sha256": native_hash, "module": native.__file__}),
        file=sys.stderr,
        flush=True,
    )

    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("k", pa.int64()),
            ("ts", pa.timestamp("us", tz="UTC")),
            ("x", pa.int64()),
        ]
    )
    source = cf.table_input("events", schema=schema)
    accepted = source.filter(source["id"] >= 0)
    if case == "stateless":
        selected = accepted.filter(accepted["k"] < 8)
        output = selected.select("id", y=selected["x"] * 2 + 1)
    else:
        output = cf.window.tumbling(
            accepted,
            event_time="ts",
            size_micros=4_000_000,
            group_by=("k",),
            aggregates=(
                cf.window.sum("x", output="total"),
                cf.window.count("id", output="cnt"),
            ),
        )

    async def batches():
        async with connect(endpoint, compression=None, max_size=4 * 1024 * 1024) as ws:
            async for frame in ws:
                data = frame.encode() if isinstance(frame, str) else frame
                table = paj.read_json(
                    io.BytesIO(data),
                    read_options=paj.ReadOptions(use_threads=False),
                    parse_options=paj.ParseOptions(explicit_schema=schema),
                )
                yield table
                if case == "window":
                    yield cf.Watermark(
                        table["ts"][0].as_py() - timedelta(microseconds=1)
                    )

    def write(table):
        if case == "window":
            table = pa.table(
                {
                    "ws": pc.cast(table["window_start"], pa.int64()),
                    "k": table["k"],
                    "total": table["total"],
                    "cnt": table["cnt"],
                }
            )
        payload = "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in table.to_pylist()
        )
        sys.stdout.write(payload)
        sys.stdout.flush()

    async with output.stream(
        batches(),
        config=cf.StreamRuntimeConfig(checkpointing=False),
        watermarks=cf.SourceProvidedWatermarks(),
    ) as results:
        async for table in results:
            await asyncio.to_thread(write, table)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)],
        "p99": ordered[max(0, int(len(ordered) * 0.99 + 0.999999) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def prepare_worker(args, engine: str, case: str, directory: Path, server):
    endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    query_path = directory / "query.sql"
    query_path.write_text(sql(case, endpoint))
    command = (
        [
            str(args.arroyo),
            "run",
            "-p",
            "1",
            "--state-dir",
            str(directory / "state"),
            str(query_path),
        ]
        if engine == "arroyo"
        else [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            case,
            "--endpoint",
            endpoint,
        ]
    )
    command = ["taskset", "-c", args.cpus, *command]
    environment = {
        **os.environ,
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
        "ARROYO__DISABLE_TELEMETRY": "true",
        "ARROYO__DEFAULT_CHECKPOINT_INTERVAL": "1h",
        "ARROYO__PIPELINE__SOURCE_BATCH_SIZE": str(CHUNK),
        "ARROYO__PIPELINE__SOURCE_BATCH_LINGER": "100ms",
        "ARROYO__PIPELINE__ALLOWED_RESTARTS": "0",
        "ARROYO__ADMIN__HTTP_PORT": "0",
        "ARROYO__ADMIN__BIND_ADDRESS": "127.0.0.1",
        "ARROYO__API__BIND_ADDRESS": "127.0.0.1",
        "ARROYO__CONTROLLER__BIND_ADDRESS": "127.0.0.1",
        "ARROYO__WORKER__BIND_ADDRESS": "127.0.0.1",
    }
    save(
        directory / "command.json",
        {
            "argv": command,
            "cwd": str(Path.cwd()),
            "settings": {
                k: v
                for k, v in environment.items()
                if k.startswith(
                    (
                        "ARROYO__",
                        "TOKIO_",
                        "OMP_",
                        "OPENBLAS_",
                        "ARROW_",
                        "RUST_LOG",
                    )
                )
            },
        },
    )
    return command, environment


def output_observers(directory: Path, expected_count: int):
    arrivals: list[tuple[float, dict]] = []
    memory: list[dict] = []
    done = asyncio.Event()

    async def read(process):
        with (directory / "stdout.jsonl").open("wb") as raw:
            while line := await process.stdout.readline():
                now = time.perf_counter()
                raw.write(line)
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(row, dict) and (("id" in row) or ("ws" in row)):
                    arrivals.append((now, row))
                    if len(arrivals) >= expected_count:
                        done.set()

    async def sample_memory(process):
        parent = psutil.Process(process.pid)
        while process.returncode is None:
            try:
                tree = [parent, *parent.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                memory.append({"at": time.perf_counter(), "rss_bytes": rss})
            except psutil.Error:
                pass
            await asyncio.sleep(0.01)

    return arrivals, memory, done, read, sample_memory


async def await_output(connected, release, done, send_done, memory):
    await asyncio.wait_for(connected.wait(), 25)
    ready = time.perf_counter()
    ready_rss = memory[-1]["rss_bytes"] if memory else None
    release.set()
    await asyncio.wait_for(done.wait(), 15)
    await asyncio.wait_for(send_done.wait(), 5)
    # Observe duplicates while the input remains open.
    await asyncio.sleep(0.1)
    return ready, ready_rss


def output_metrics(case: str, frames, expected, sent, arrivals):
    actual = [row for _, row in arrivals]
    key = (
        (lambda row: row["id"])
        if case == "stateless"
        else (lambda row: (row["ws"], row["k"]))
    )
    if sorted(actual, key=key) != sorted(expected, key=key):
        raise ValueError(f"wrong output: {len(actual)} rows, expected {len(expected)}")
    latencies = []
    finalization = []
    for at, row in arrivals:
        index = (
            (row["id"] // CHUNK)
            if case == "stateless"
            else int((row["ws"] - BASE.timestamp() * 1_000_000) // 2_000_000)
        )
        # Window index is its first frame. Each window spans two frames.
        contribution = index if case == "stateless" else index + 1
        latencies.append((at - sent[contribution]) * 1000)
        if case == "window":
            progress = min(index + 3, len(frames) - 1)
            finalization.append((at - sent[progress]) * 1000)
    elapsed = arrivals[-1][0] - sent[0]
    return actual, elapsed, latencies, finalization


async def stop_process(process) -> None:
    # Stop this exact child only. The open source is released after stop.
    if process.returncode is None:
        process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), 12)
        except TimeoutError:
            process.kill()
            await process.wait()


async def acquisition(
    args, engine: str, case: str, gap: float, directory: Path
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    frames, records = fixture(args.rows)
    expected = oracle(case, records)
    save(directory / "expected.json", expected)
    sent: list[float] = []
    arrivals, memory, done, read, sample_memory = output_observers(
        directory, len(expected)
    )
    connected = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    send_done = asyncio.Event()
    started = time.perf_counter()

    async def source(ws):
        connected.set()
        await release.wait()
        for frame in frames:
            sent.append(time.perf_counter())
            await ws.send(frame)
            if gap:
                await asyncio.sleep(gap)
            else:
                await asyncio.sleep(0)
        send_done.set()
        # No EOF until every expected output was observed and checked.
        await closed.wait()

    async with serve(source, "127.0.0.1", 0, compression=None) as server:
        command, environment = prepare_worker(args, engine, case, directory, server)
        with (directory / "stderr.log").open("wb") as errors:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=errors,
                env=environment,
                limit=4 * 1024 * 1024,
            )

            reader = asyncio.create_task(read(process))
            sampler = asyncio.create_task(sample_memory(process))
            failure = None
            result = {}
            try:
                ready, ready_rss = await await_output(
                    connected, release, done, send_done, memory
                )
                actual, elapsed, latencies, finalization = output_metrics(
                    case, frames, expected, sent, arrivals
                )
                result = {
                    "engine": engine,
                    "case": case,
                    "gap_s": gap,
                    "useful_input_rows": args.rows,
                    "wire_input_rows": args.rows + CHUNK,
                    "output_rows": len(actual),
                    "correctness": "exact_all_rows",
                    "output_before_eof": True,
                    "elapsed_s": elapsed,
                    "rows_per_second": args.rows / elapsed,
                    "latency_ms": quantiles(latencies),
                    "finalization_latency_ms": quantiles(finalization)
                    if finalization
                    else None,
                    "startup_to_connection_s": ready - started,
                    "rss_peak_bytes": max(row["rss_bytes"] for row in memory),
                    "rss_ready_bytes": ready_rss,
                    "input_sha256": hashlib.sha256(b"".join(frames)).hexdigest(),
                }
            except Exception as error:
                failure = error
                result = {
                    "engine": engine,
                    "case": case,
                    "gap_s": gap,
                    "error": repr(error),
                    "received_rows": len(arrivals),
                    "sent_frames": len(sent),
                }
            finally:
                await stop_process(process)
                closed.set()
                await reader
                await sampler
                result["exit_code"] = process.returncode
                save(directory / "sample.json", result)
                save(directory / "memory.json", memory)
                save(
                    directory / "arrivals.json",
                    [{"at": at, "row": row} for at, row in arrivals],
                )
                save(directory / "send_times.json", sent)
            if failure:
                raise RuntimeError(f"{directory.name}: {failure!r}") from failure
            return result


async def controller(args) -> None:
    import calc_flow._native as native

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    identity = {
        "calc_flow_commit": CF_REF,
        "arroyo_commit": ARROYO_REF,
        "arroyo_binary_sha256": digest(args.arroyo),
        "harness_sha256": digest(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "cpus": args.cpus,
        "rows": args.rows,
        "pairs": args.pairs,
        "budget_s": args.budget,
        "contract": "cross-engine-live-integration-v1-informational",
        "created_utc": datetime.now(UTC).isoformat(),
        "native_sha256": digest(Path(native.__file__)),
        "native_path": native.__file__,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("calc-flow-python", "pyarrow", "numpy", "psutil", "websockets")
        },
    }
    os.environ["COMPARISON_NATIVE_SHA256"] = identity["native_sha256"]
    save(root / "identity.json", identity)
    samples = []
    cases = [(case, gap) for case in args.cases for gap in args.gaps]
    for case, gap in cases:
        for pair in range(args.pairs):
            order = (
                ("calc_flow", "arroyo") if pair % 2 == 0 else ("arroyo", "calc_flow")
            )
            for engine in (value for value in order if value in args.engines):
                if time.perf_counter() - started > args.budget:
                    raise TimeoutError("declared acquisition budget exhausted")
                name = f"{case}-gap{gap:g}-pair{pair}-{engine}"
                print(name, flush=True)
                result = await acquisition(args, engine, case, gap, root / name)
                samples.append({"pair": pair, **result})
                save(root / "samples.json", samples)
                print(json.dumps(result), flush=True)
    summary = []
    for case, gap in cases:
        for engine in ("calc_flow", "arroyo"):
            if engine not in args.engines:
                continue
            selected = [
                s
                for s in samples
                if s["engine"] == engine and s["case"] == case and s["gap_s"] == gap
            ]
            summary.append(
                {
                    "case": case,
                    "gap_s": gap,
                    "engine": engine,
                    "throughput": quantiles([s["rows_per_second"] for s in selected]),
                    "latency_p50_ms": quantiles(
                        [s["latency_ms"]["p50"] for s in selected]
                    ),
                    "latency_p95_ms": quantiles(
                        [s["latency_ms"]["p95"] for s in selected]
                    ),
                    "rss_peak_mib": quantiles(
                        [s["rss_peak_bytes"] / 2**20 for s in selected]
                    ),
                    "startup_s": quantiles(
                        [s["startup_to_connection_s"] for s in selected]
                    ),
                }
            )
    save(root / "summary.json", summary)
    save(
        root / "completion.json",
        {"elapsed_s": time.perf_counter() - started, "samples": len(samples)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["stateless", "window"])
    parser.add_argument("--endpoint")
    parser.add_argument("--arroyo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--cpus", default="4-7")
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--cases", nargs="+", default=["stateless", "window"])
    parser.add_argument("--gaps", nargs="+", type=float, default=[0.0, 0.005])
    parser.add_argument("--engines", nargs="+", default=["calc_flow", "arroyo"])
    args = parser.parse_args()
    if args.worker:
        asyncio.run(cf_worker(args.worker, args.endpoint))
    else:
        if args.rows <= 0 or args.rows % (CHUNK * 2):
            parser.error("--rows must be a positive multiple of 1024")
        args.arroyo = args.arroyo.resolve()
        asyncio.run(controller(args))


if __name__ == "__main__":
    main()
