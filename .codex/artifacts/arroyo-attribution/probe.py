"""Capture engine/node identity and finite-window semantic limitations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import CF_REF, case_named, check, declaration, save, schema
from engines import arrow_tables, finite_plan
from profiling import Profile


def window_probe(root):
    import calc_flow as cf

    result = {}
    try:
        declaration("sql", True)
    except ValueError as error:
        result["native_window_sql_engine_rejection"] = str(error)
    else:
        raise RuntimeError("expected native event window to require streaming")
    tables, _ = arrow_tables(case_named("arrow-full"))
    source = cf.table_input("events", schema=schema())
    output = source.sql(
        "SELECT CAST(FLOOR(EXTRACT(epoch FROM ts) / 4) * 4000000 AS BIGINT) AS ws, "
        "k, SUM(x) AS total, COUNT(id) AS cnt FROM input "
        "WHERE id >= 0 GROUP BY 1, 2"
    )
    program = cf.Program(
        "terminal-window-values", engine="sql", outputs={"result": output}
    )
    actual = program.collect({"events": tables[0]})["result"].to_pylist()
    from common import prepared

    _, _, expected = prepared(case_named("window-live-512"))
    check(actual, expected)
    partial = program.collect({"events": tables[0].slice(0, 512)})["result"].to_pylist()
    result.update(
        {
            "finite_full_table_matches_final_values": True,
            "finite_first_frame_count_values": sorted({row["cnt"] for row in partial}),
            "final_count_values": sorted({row["cnt"] for row in actual}),
            "finite_query": output._node.op.name,
            "temporal_equivalence": False,
            "reason": (
                "full table requires all data; per-frame aggregate is partial, "
                "no watermark finality"
            ),
        }
    )
    save(root / "finite-window-final-values.json", actual)
    return result


def main():
    import calc_flow as cf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    programs = {engine: declaration(engine) for engine in ("sql", "streaming")}
    for engine, program in programs.items():
        save(
            args.output / f"{engine}-expression-project.json",
            program.to_project().model_dump(),
        )
    if programs["sql"].fingerprint != programs["streaming"].fingerprint:
        raise RuntimeError("engine control does not preserve the declaration graph")
    runtime = cf.Runtime()
    profile = Profile(args.output, False)
    plan, inputs, output = finite_plan(programs["sql"], runtime, profile)
    tables, _ = arrow_tables(case_named("arrow-full"))
    batch = cf.Batch.from_pyarrow(tables[0])
    actual = plan.execute({name: batch for name in inputs})
    save(args.output / "datafusion-metrics.json", actual.datafusion_metrics)
    save(args.output / "node-timings.json", actual.node_timings)
    source = cf.table_input("events", schema=schema())
    sql_node = source.sql(
        "SELECT id, x * 2 + 1 AS y FROM input WHERE id >= 0 AND k < 8"
    )
    save(
        args.output / "streaming-sql-node-project.json",
        cf.Program(
            "sql-node-distinction", engine="streaming", outputs={"result": sql_node}
        )
        .to_project()
        .model_dump(),
    )
    save(
        args.output / "semantics.json",
        {
            "source_ref": CF_REF,
            "expression_fingerprint": programs["sql"].fingerprint,
            "same_expression_graph_across_engines": True,
            "compiled_input_bindings": inputs,
            "compiled_output_binding": output,
            "window": window_probe(args.output),
            "elapsed_s": time.perf_counter() - start,
        },
    )
    print("Engine graph, physical plan and finite-window semantic probe passed")


if __name__ == "__main__":
    main()
