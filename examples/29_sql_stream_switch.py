"""Run the same SQL declaration as one batch or a cumulative stream."""

from __future__ import annotations

import argparse
import asyncio

import pyarrow as pa

import calc_flow as cf


async def stream_snapshots(
    program: cf.Program, batches: tuple[pa.Table, ...]
) -> list[dict[str, list[int]]]:
    async def source():
        for batch in batches:
            yield batch

    snapshots = []
    async with program.execute({"events": source()}) as results:
        async for output in results:
            if output.name != "totals":
                raise RuntimeError(f"unexpected output {output.name!r}")
            snapshots.append(output.table.to_pydict())
    return snapshots


def make_program(engine: str) -> cf.Program:
    program = cf.Program("sum-across-modes", engine=engine)
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    return program.output("totals", source.sql("SELECT SUM(x) AS total FROM input"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=("sql", "streaming", "both"), default="both"
    )
    selected = parser.parse_args().engine

    batches = (pa.table({"x": [1]}), pa.table({"x": [2, 3]}))
    expected = {
        "sql": [{"total": [6]}],
        "streaming": [{"total": [1]}, {"total": [6]}],
    }

    for engine in ("sql", "streaming") if selected == "both" else (selected,):
        program = make_program(engine)
        if engine == "sql":
            table = program.execute({"events": pa.concat_tables(batches)})["totals"]
            actual = [table.to_pydict()]
        else:
            actual = asyncio.run(stream_snapshots(program, batches))
        if actual != expected[engine]:
            raise RuntimeError(f"unexpected {engine} result: {actual}")
        print(f"{engine}: {actual}")


if __name__ == "__main__":
    main()
