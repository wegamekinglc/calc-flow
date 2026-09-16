"""Project-only late-output capability probes and native diagnostic projection."""

from __future__ import annotations

import json
import re
from typing import Protocol

from calc_flow import CalcFlowError


class ValidatingRuntime(Protocol):
    def validation_report(self, project_json: str) -> dict[str, object]: ...


def late_nodes(project: dict) -> list[tuple[int, dict]]:
    nodes = project.get("graph", {}).get("nodes", [])
    return [(index, node) for index, node in enumerate(nodes) if _is_late_node(node)]


def _is_late_node(node: dict) -> bool:
    operator = node.get("operator", {})
    return operator.get("kind") in {"rolling", "cross_section"} and (
        operator.get("spec", {}).get("late_policy", {}).get("kind") == "side_output"
    )


def project_late_issues(project: dict, report: dict) -> dict:
    if not late_nodes(project):
        return report
    return {
        **report,
        "issues": [_late_issue(project, issue) for issue in report["issues"]],
    }


def _late_issue(project: dict, issue: dict) -> dict:
    # Both regexes parse native stream_compile error text; a native wording
    # change silently degrades Studio paths/codes, so keep them in sync with
    # the calc-flow compile diagnostics.
    if issue["code"] != "stream_compile":
        return issue
    message = issue["message"]
    match = re.search(r"(?:invalid: )([\w.\[\]]+) \[([a-z_]+)\]:", message)
    if match:
        path, code = match.groups()
        if code == "sink_output_mismatch":
            code = "missing_binding"
        return {**issue, "path": path, "code": code}
    if "temporal_output_unavailable:" in message:
        return {
            **issue,
            "path": _temporal_path(project, message),
            "code": "temporal_output_unavailable",
        }
    return issue


def _temporal_path(project: dict, message: str) -> str:
    match = re.search(r"graph.edges\[(\d+)\]: temporal_output_unavailable:", message)
    if match and int(match[1]) < len(project["graph"].get("edges", [])):
        return f"graph.edges[{match[1]}]"
    return "graph.edges"


def _probe_project(kind: str) -> str:
    fields = [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "key", "data_type": "int64", "nullable": False},
        {"name": "x", "data_type": "float64", "nullable": False},
    ]
    spec = {
        "configuration_version": 1,
        "state_layout_version": 1,
        "event_time": "ts",
        "sequence_by": ["ts"],
        "allowed_lateness_micros": 0,
        "late_policy": {
            "kind": "side_output",
            "metrics_version": 1,
            "schema_version": 1,
        },
    }
    if kind == "rolling":
        spec.update(
            partition_by=["key"],
            value_policy="stateful_numeric_v1",
            outputs=[
                {
                    "kind": "lag",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "previous",
                    "periods": 1,
                }
            ],
        )
    else:
        spec.update(
            entity_by=["key"],
            grouping={"kind": "exact_time"},
            value_policy="nan_exclude_preserve_v1",
            outputs=[
                {
                    "kind": "rank",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "rank",
                    "direction": "ascending",
                    "tie_method": "average",
                    "null_placement": "exclude",
                    "min_samples": 1,
                }
            ],
        )
    connector = {"provider": "calc-flow-connectors", "name": "file", "version": "2.0.0"}
    return json.dumps(
        {
            "format_version": 3,
            "id": "late_capability",
            "name": "late capability",
            "runtime": {"mode": "stream", "options": {}},
            "graph": {
                "name": "late-capability",
                "nodes": [
                    {
                        "id": "probe",
                        "operator": {"kind": kind, "spec": spec},
                        "input_ports": [
                            {
                                "name": "input",
                                "kind": "table",
                                "required": True,
                                "schema": fields,
                            }
                        ],
                    }
                ],
            },
            "sources": [
                {
                    "binding": "input",
                    "connector": connector,
                    "schema": fields,
                    "options": {
                        "path": "__late_capability__.parquet",
                        "format": "parquet",
                    },
                    "watermark": {
                        "policy": "bounded_out_of_orderness",
                        "column": "ts",
                        "delay_ms": 1,
                        "emit_interval_ms": 1,
                    },
                }
            ],
            "sinks": [
                {
                    "binding": name,
                    "connector": connector,
                    "options": {"path": "__late_capability__", "output": name},
                    "delivery": "at_least_once",
                }
                for name in ("output", "late")
            ],
        }
    )


def supported_late_operators(runtime: ValidatingRuntime) -> tuple[str, ...]:
    """Compile fixed data-only probes; validation never opens a Source or Sink."""
    supported = []
    for kind in ("cross_section", "rolling"):
        try:
            report = runtime.validation_report(_probe_project(kind))
        except (CalcFlowError, ValueError):
            continue
        if report.get("valid") is True and report.get("fingerprint"):
            supported.append(kind)
    return tuple(supported)
