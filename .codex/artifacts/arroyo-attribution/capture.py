"""Foreground process ownership and shared live ingress/output boundary."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from common import ROWS, Transport, check, environment, legacy, prepared, save
from websockets.asyncio.server import serve


@asynccontextmanager
async def observed(command, settings, directory, count):
    arrivals, memory, done, read, sample = legacy.output_observers(directory, count)
    save(
        directory / "command.json",
        {"argv": command, "cwd": str(Path.cwd()), "settings": settings},
    )
    with (directory / "stderr.log").open("wb") as errors:
        process = await asyncio.create_subprocess_exec(
            *command,
            env={**os.environ, **settings},
            stdout=asyncio.subprocess.PIPE,
            stderr=errors,
            limit=4 * 1024 * 1024,
        )
        reader = asyncio.create_task(read(process))
        sampler = asyncio.create_task(sample(process))
        try:
            yield process, arrivals, memory, done
        finally:
            if process.pid == int(os.environ["DAL417_DAEMON_PID"]):
                raise RuntimeError("refusing to stop the workspace daemon")
            await legacy.stop_process(process)
            await reader
            await sampler
            save(directory / "memory.json", memory)
            save(
                directory / "arrivals.json",
                [{"at": at, "row": row} for at, row in arrivals],
            )
            save(directory / "exit.json", {"exit_code": process.returncode})


def command_for(args, case, arm, directory, transport):
    endpoint, profile = transport.endpoint, transport.profile
    settings = environment()
    settings["ARROYO__PIPELINE__SOURCE_BATCH_SIZE"] = str(case.batch_rows)
    if arm == "arroyo":
        query = directory / "query.sql"
        query.write_text(legacy.sql("window" if case.window else "stateless", endpoint))
        command = [
            str(args.arroyo),
            "run",
            "-p",
            "1",
            "--state-dir",
            str(directory / "state"),
            str(query),
        ]
    else:
        command = [
            sys.executable,
            str(Path(__file__).with_name("run.py")),
            "worker",
            "--case",
            case.name,
            "--arm",
            arm,
            "--directory",
            str(directory),
            "--endpoint",
            endpoint,
            "--native-sha",
            args.native_sha,
        ]
        if profile:
            command.append("--profile")
    return ["taskset", "-c", "4-7", *command], settings


def live_metrics(case, sent, arrivals, expected):
    check([row for _, row in arrivals], expected)
    latencies = []
    progress = []
    base_us = int(legacy.BASE.timestamp() * 1_000_000)
    for at, row in arrivals:
        if not case.window:
            index = row["id"] // case.batch_rows
            latencies.append((at - sent[index]) * 1000)
        else:
            index = (row["ws"] - base_us) // 2_000_000
            latencies.append((at - sent[index + 1]) * 1000)
            progress.append((at - sent[min(index + 3, 64)]) * 1000)
    elapsed = arrivals[-1][0] - sent[0]
    return {
        "elapsed_s": elapsed,
        "rows_per_second": ROWS / elapsed,
        "latency_ms": legacy.quantiles(latencies),
        "progress_latency_ms": legacy.quantiles(progress) if progress else None,
        "output_rows": len(arrivals),
        "correctness": "exact_all_rows",
        "output_before_eof": True,
    }


async def wait_ready(connected, process):
    waiter = asyncio.create_task(connected.wait())
    exited = asyncio.create_task(process.wait())
    done, pending = await asyncio.wait(
        (waiter, exited), timeout=25, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if waiter not in done:
        raise RuntimeError(f"source did not connect; child exit={process.returncode}")


async def live_sample(args, case, arm, directory, profile):
    frames, _, expected = prepared(case)
    save(directory / "expected.json", expected)
    connected, release, closed, sent_all = (asyncio.Event() for _ in range(4))
    sent = []

    async def source(ws):
        connected.set()
        await release.wait()
        for frame in frames:
            sent.append(time.perf_counter())
            await ws.send(frame)
            await asyncio.sleep(0)
        sent_all.set()
        await closed.wait()

    async with serve(source, "127.0.0.1", 0, compression=None) as server:
        endpoint = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        command, settings = command_for(
            args, case, arm, directory, Transport(endpoint, profile)
        )
        try:
            async with observed(command, settings, directory, len(expected)) as record:
                process, arrivals, memory, done = record
                await wait_ready(connected, process)
                if profile and arm == "streaming":
                    await wait_profile_ready(directory)
                release.set()
                await asyncio.wait_for(done.wait(), 20)
                await asyncio.wait_for(sent_all.wait(), 5)
                await asyncio.sleep(0.1)
                result = live_metrics(case, sent, arrivals, expected)
                result["rss_peak_bytes"] = max(row["rss_bytes"] for row in memory)
        finally:
            release.set()
            closed.set()
            save(directory / "send_times.json", sent)
    result["input_sha256"] = hashlib.sha256(b"".join(frames)).hexdigest()
    return result


async def wait_profile_ready(directory):
    deadline = time.perf_counter() + 5
    while not (directory / "profile-ready").exists():
        if time.perf_counter() >= deadline:
            raise TimeoutError("callback profiler did not become ready")
        await asyncio.sleep(0.005)


async def arrow_sample(args, case, arm, directory, profile):
    import json

    command, settings = command_for(args, case, arm, directory, Transport("", profile))
    async with observed(command, settings, directory, 1) as record:
        process, _, memory, _ = record
        await asyncio.wait_for(process.wait(), 25)
        if process.returncode != 0:
            raise RuntimeError(f"Arrow worker failed: {process.returncode}")
        result = json.loads((directory / "worker-result.json").read_text())
        result["rss_peak_bytes"] = max(row["rss_bytes"] for row in memory)
    frames, _, _ = prepared(case)
    result["input_sha256"] = hashlib.sha256(b"".join(frames[:-1])).hexdigest()
    return result
