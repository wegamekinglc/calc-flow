"""Round-trip a project through JSON, YAML, and an async file store."""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import Batch, FileProjectStore, PipelineBuilder, ProjectDocument, Runtime
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)


async def run(directory: str) -> None:
    builder = PipelineBuilder("saved-totals").expression("calculate", "total = a + b")
    original = builder.project
    document = ProjectDocument.model_validate(original)
    canonical = document.canonical_json()

    from_json = import_project_json(export_project_json(document))
    from_yaml = import_project_yaml(export_project_yaml(document))
    if from_json.canonical_json() != canonical:
        raise RuntimeError("unexpected JSON round-trip result")
    if from_yaml.canonical_json() != canonical:
        raise RuntimeError("unexpected YAML round-trip result")

    store = FileProjectStore(directory)
    await store.create(from_yaml)
    loaded = await store.get(document.root["id"])
    if loaded.canonical_json() != canonical:
        raise RuntimeError("unexpected stored project content")
    if [item.root["id"] for item in await store.list()] != [document.root["id"]]:
        raise RuntimeError("unexpected project-store inventory")

    plan = Runtime().compile_batch_project(loaded.canonical_json())
    result = await plan.execute_async(
        {"input": Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))}
    )
    totals = result.outputs["output"].to_pyarrow()["total"].to_pylist()
    if totals != [3, 7]:
        raise RuntimeError(f"unexpected reloaded totals: {totals}")
    if builder.project != original:
        raise RuntimeError("unexpected mutation of the original builder")
    print("reloaded totals:", totals)
    print("JSON, YAML, and file-store round trips agree")


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-project-example-") as directory:
        asyncio.run(run(directory))


if __name__ == "__main__":
    main()
