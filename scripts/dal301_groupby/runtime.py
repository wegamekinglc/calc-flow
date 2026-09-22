from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

from scripts.benchmark_suite.process import ROOT, Worker, child_environment
from scripts.toolkit import sha256_file, write_json


class AuditWorker(Worker):
    @classmethod
    async def start(cls, site: Path, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        log = (root / "stderr.log").open("wb")
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "scripts.dal301_groupby",
                "worker",
                "--output",
                str(root),
                cwd=ROOT,
                env=child_environment(site),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
            )
        except BaseException:
            log.close()
            raise
        return cls(process, log)

    async def request(self, **message) -> dict:
        record = {"request": message, "time_ns": time.time_ns()}
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=900)
            with Path(self.log.name).with_name("stdout.log").open("ab") as stream:
                stream.write(line)
            result = json.loads(line)
            if not isinstance(result, dict) or "error" in result:
                raise ValueError(f"invalid worker result: {result}")
            record["response"] = result
            if message["operation"] == "prepare":
                self.input_hashes = result["input_hashes"]
            return result
        except BaseException as error:
            record["error"] = repr(error)
            raise
        finally:
            with Path(self.log.name).with_suffix(".ipc.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    async def close(self) -> None:
        path = Path(self.log.name).with_name("exit.json")
        try:
            await super().close()
        finally:
            write_json(
                path, {"pid": self.process.pid, "exit_code": self.process.returncode}
            )
        if self.process.returncode != 0:
            raise RuntimeError(f"worker exited {self.process.returncode}")


def input_hashes(active, root: Path) -> dict:
    import pyarrow as pa

    result = {}
    for name in ("table", "dimension"):
        table = getattr(active.data, name)
        path = root / f"{name}.arrow"
        with (
            pa.OSFile(str(path), "wb") as stream,
            pa.ipc.new_stream(stream, table.schema) as writer,
        ):
            writer.write_table(table)
        result[name] = sha256_file(path)
    return result


def worker(root: Path) -> None:
    from scripts.benchmark_suite.worker import dispatch
    from scripts.dal301_groupby.contract import cases

    active = None
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message["operation"] == "prepare" and message["case"] not in cases():
                raise ValueError("worker accepts only the fixed group_by cases")
            response, active = dispatch(message, active, root)
            if message["operation"] == "prepare":
                response = {**response, "input_hashes": input_hashes(active, root)}
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        if active is not None:
            active.close()


def resource_sample() -> dict:
    import psutil

    builds = [
        p.info
        for p in psutil.process_iter(["pid", "name"])
        if p.info["name"] in ("rustc", "cargo", "maturin", "cc1", "clang")
    ]
    return {
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "swap": psutil.swap_memory()._asdict(),
        "builds": builds,
        "load": os.getloadavg(),
    }


async def monitor(path: Path, stop: asyncio.Event) -> None:
    with path.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            stream.write(json.dumps(resource_sample()) + "\n")
            stream.flush()
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=0.25)
