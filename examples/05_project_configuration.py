"""Persist, validate, compile, and execute a data-only project config."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    DataSourceConfig,
    EdgeConfig,
    FileProjectStore,
    InputFormat,
    NodeConfig,
    PipelineConfig,
    ProjectConfig,
    compile_project,
    validate_project,
)


def project_config() -> ProjectConfig:
    return ProjectConfig(
        id="sales-demo",
        name="Sales demo",
        description="A portable DataFusion project configuration.",
        pipeline=PipelineConfig(
            id="main",
            name="Configured sales pipeline",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="gross = quantity * unit_price",
                ),
                NodeConfig(
                    id="select_large",
                    kind="expression",
                    select=("order_id", "gross"),
                    filter_expression="gross >= 20",
                    position={"x": 280, "y": 0},
                ),
            ),
            edges=(EdgeConfig(source_node="calculate", target_node="select_large"),),
        ),
        data_sources=(
            DataSourceConfig(
                id="sample-orders",
                input_name="input",
                format=InputFormat.INLINE_JSON,
                data=[
                    {"order_id": "A-100", "quantity": 3, "unit_price": 10},
                    {"order_id": "A-101", "quantity": 1, "unit_price": 12},
                ],
                source_id="example",
            ),
        ),
    )


def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-projects-") as directory:
        store = FileProjectStore(directory)
        store.create(project_config())
        project = store.get("sales-demo")

        report = validate_project(project)
        if not report.valid:
            raise RuntimeError(report.errors)

        sample = project.data_sources[0].data
        if not isinstance(sample, list):
            raise TypeError("the inline sample must contain a list of records")
        run = compile_project(project).execute(
            {"input": Batch.table(pa.Table.from_pylist(sample))}
        )

        print("fingerprint:", report.fingerprint)
        print("stored projects:", [item.id for item in store.list()])
        print("result:", run.output.table_payload.to_pylist())


if __name__ == "__main__":
    main()
