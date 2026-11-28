"""Round-trip a project through JSON, YAML, and an async file store."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import pyarrow as pa

import calc_flow as cf
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)


def run(directory: str) -> None:
    data = pa.table({"a": [1, 3], "b": [2, 4]})
    t = cf.table_input("numbers", schema=data.schema)
    program = cf.Program(
        "saved-totals",
        mode="batch",
        outputs={"totals": t.select(total=t["a"] + t["b"])},
    )
    document = program.to_project()
    canonical = document.canonical_json()

    from_json = import_project_json(export_project_json(document))
    from_yaml = import_project_yaml(export_project_yaml(document))
    if from_json.canonical_json() != canonical:
        raise RuntimeError("unexpected JSON round-trip result")
    if from_yaml.canonical_json() != canonical:
        raise RuntimeError("unexpected YAML round-trip result")

    store = cf.FileProjectStore(directory)
    store.create_blocking(from_yaml)
    loaded = store.get_blocking(document.root["id"])
    if loaded.canonical_json() != canonical:
        raise RuntimeError("unexpected stored project content")
    if [item.root["id"] for item in store.list_blocking()] != [document.root["id"]]:
        raise RuntimeError("unexpected project-store inventory")

    # Reloaded projects expose native bindings, not Python program aliases.
    plan = cf.Runtime().compile_batch_project(loaded.canonical_json())
    result = plan.execute({"input": cf.Batch.from_pyarrow(data)})
    totals = result.outputs["output"].to_pyarrow()["total"].to_pylist()
    if totals != [3, 7]:
        raise RuntimeError(f"unexpected reloaded totals: {totals}")
    if program.to_project().canonical_json() != canonical:
        raise RuntimeError("unexpected mutation of the original program")
    print("reloaded totals:", totals)
    print("JSON, YAML, and file-store round trips agree")


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-project-example-") as directory:
        run(directory)


if __name__ == "__main__":
    main()
