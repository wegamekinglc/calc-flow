"""Bounded real-engine network/cleanup probe; does not collect benchmark samples."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import signal
import socket
import sys
from pathlib import Path

import psutil

from scripts.risingwave_isolation import network_isolation
from scripts.toolkit import sha256_file, write_json

MODES = ("normal", "exception", "worker-kill", "launcher-kill")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def listeners() -> list[dict]:
    return [
        {"address": c.laddr.ip, "port": c.laddr.port, "pid": c.pid}
        for c in psutil.net_connections(kind="inet")
        if c.status == psutil.CONN_LISTEN
    ]


async def wait_file(path: Path, process=None) -> None:
    async with asyncio.timeout(45):
        while not path.exists():
            if process is not None:
                require(
                    process.returncode is None, "probe exited before ready; inspect log"
                )
            await asyncio.sleep(0.025)


async def dashboard_response() -> str:
    reader, writer = await asyncio.open_connection("127.0.0.1", 5691)
    try:
        writer.write(
            b"GET /api/version HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        return (await asyncio.wait_for(reader.readline(), 5)).decode().strip()
    finally:
        writer.close()
        await writer.wait_closed()


async def worker(args) -> None:
    from scripts.risingwave_comparison import RW_COMMIT, RisingWave, process_usage

    isolation = network_isolation()
    os.sched_setaffinity(0, {0, 1, 2, 3})
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    (args.output / "risingwave.toml").write_text(
        "[server]\ntelemetry_enabled = false\n"
    )
    rw = RisingWave(args, args.output)
    try:
        await rw.start()
        version = await rw.execute("SELECT version()")
        require(RW_COMMIT[:10] in str(version), "wrong RisingWave binary")
        ready = {
            "isolation": isolation,
            "version": version,
            "listeners": listeners(),
            "dashboard_http": await dashboard_response(),
            "resource_probe": {
                "python": process_usage(os.getpid()),
                "risingwave": process_usage(rw.process.pid),
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "address_space_limit": list(resource.getrlimit(resource.RLIMIT_AS)),
        }
        require(
            any(c["port"] == 5691 for c in ready["listeners"]),
            "dashboard not listening",
        )
        require("200" in ready["dashboard_http"], "dashboard route did not respond")
        write_json(args.output / "ready.json", ready)
        await wait_file(args.output / "release")
        if args.worker == "worker-kill":
            os.kill(os.getpid(), signal.SIGKILL)
        require(args.worker != "exception", "injected post-readiness exception")
    finally:
        await rw.stop()
        write_json(
            args.output / "worker-cleanup.json",
            {"risingwave_returncode": rw.process.returncode},
        )


def host_addresses() -> list[str]:
    return sorted(
        {
            a.address.split("%")[0]
            for addresses in psutil.net_if_addrs().values()
            for a in addresses
            if a.family in (socket.AF_INET, socket.AF_INET6)
        }
    )


async def probe_host(address: str) -> dict:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(address, 5691), 0.5)
    except (OSError, TimeoutError) as error:
        return {
            "address": address,
            "port": 5691,
            "reachable": False,
            "error": type(error).__name__,
        }
    writer.close()
    await writer.wait_closed()
    return {"address": address, "port": 5691, "reachable": True}


def owned_processes(pid: int) -> list[dict]:
    return [
        {"pid": p.pid, "created": p.create_time(), "name": p.name()}
        for p in [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
    ]


def still_alive(record: dict) -> bool:
    try:
        process = psutil.Process(record["pid"])
        return (
            process.create_time() == record["created"]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


async def confirm_cleanup(processes: list[dict]) -> None:
    async with asyncio.timeout(5):
        while any(still_alive(p) for p in processes):
            await asyncio.sleep(0.025)


async def settle(process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


async def one_probe(args, mode: str) -> dict:
    output = args.output / mode
    output.mkdir()
    with (output / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "benchmarks/risingwave/isolate.sh",
            sys.executable,
            "-m",
            "scripts.verify_risingwave_isolation",
            "--worker",
            mode,
            "--risingwave",
            str(args.risingwave),
            "--python312-lib",
            str(args.python312_lib),
            "--output",
            str(output),
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        processes = []
        try:
            await wait_file(output / "ready.json", process)
            processes = owned_processes(process.pid)
            result = await observe_host(output, processes)
            if mode == "launcher-kill":
                process.kill()
            else:
                (output / "release").touch()
            await asyncio.wait_for(process.wait(), 30)
            await confirm_cleanup(processes)
            require(
                (process.returncode == 0) == (mode == "normal"),
                "wrong probe exit status",
            )
            return {
                **result,
                "mode": mode,
                "exit_code": process.returncode,
                "live_children_after": [],
            }
        finally:
            await settle(process)
            await confirm_cleanup(processes)


async def observe_host(output: Path, processes: list[dict]) -> dict:
    ready = json.loads((output / "ready.json").read_text())
    attempts = await asyncio.gather(*(probe_host(a) for a in host_addresses()))
    host_dashboard = [c for c in listeners() if c["port"] == 5691]
    require(not host_dashboard, "dashboard leaked into host listener table")
    require(not any(a["reachable"] for a in attempts), "dashboard reachable from host")
    return {
        "worker": ready,
        "host_connect_attempts": attempts,
        "host_dashboard_listeners": host_dashboard,
        "owned_processes": processes,
    }


async def verify(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    require(
        not any(c["port"] == 5691 for c in listeners()),
        "host dashboard port already occupied",
    )
    results = []
    async with asyncio.timeout(180):
        for mode in MODES:
            results.append(await one_probe(args, mode))
    write_json(
        args.output / "verification.json",
        {
            "baseline": "aca2a5e97592ca6e44e3a23b06659084a89b105c",
            "risingwave_sha256": sha256_file(args.risingwave),
            "harness_sha256": sha256_file(Path("scripts/risingwave_comparison.py")),
            "wrapper_sha256": sha256_file(Path("benchmarks/risingwave/isolate.sh")),
            "performance_samples_collected": 0,
            "results": results,
        },
    )
    print(
        json.dumps({"probes_passed": len(results), "performance_samples_collected": 0})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--risingwave", type=Path, required=True)
    parser.add_argument("--python312-lib", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(worker(args) if args.worker else verify(args))


if __name__ == "__main__":
    main()
