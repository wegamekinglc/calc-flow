from __future__ import annotations

import json

import pytest

import calc_flow
from calc_flow._native import registered_connectors
from calc_flow.capabilities import (
    ConnectorCapabilities,
    ConnectorCapability,
    connector_capabilities,
)


class TestNativeConnectorEnumeration:
    def test_enumerates_the_file_connector(self) -> None:
        connectors = registered_connectors()
        names = [c["name"] for c in connectors]
        assert "file" in names, f"the file connector is compiled in: {names}"

    def test_file_connector_carries_the_frozen_capabilities(self) -> None:
        connectors = registered_connectors()
        file_c = next(c for c in connectors if c["name"] == "file")
        assert file_c["provider"] == "calc-flow-connectors"
        assert file_c["kind"] == "both"
        caps = file_c["capabilities"]
        assert caps["delivery"] == "at_least_once"
        assert caps["replay"] == "replayable_exact"
        assert caps["watermark"] == "generated_only"
        assert caps["transaction"] == "pre_commit_commit"
        assert caps["snapshot"] is True
        assert caps["polling"] is False
        assert caps["cdc"] is False
        assert caps["lookup"] is False

    def test_only_lists_compiled_in_connectors(self) -> None:
        """The enumeration must not announce unreachable connectors."""
        connectors = registered_connectors()
        names = {c["name"] for c in connectors}
        # The default wheel carries only the file transport; the other
        # transports compile behind their features and must not appear.
        assert names == {"file"}, f"only compiled-in connectors appear: {names}"

    def test_options_schema_is_data_only_json(self) -> None:
        connectors = registered_connectors()
        file_c = next(c for c in connectors if c["name"] == "file")
        schema = json.loads(file_c["options_schema"])
        assert isinstance(schema, dict)
        assert "path" in schema


class TestConnectorCapabilities:
    def test_builds_from_native_data(self) -> None:
        native = registered_connectors()
        parsed = connector_capabilities(native)
        assert len(parsed) >= 1
        file_cap = next(c for c in parsed if c.name == "file")
        assert file_cap.provider == "calc-flow-connectors"
        assert file_cap.kind == "both"
        assert file_cap.capabilities.snapshot is True

    def test_sorts_deterministically(self) -> None:
        native = registered_connectors()
        caps1 = connector_capabilities(native)
        caps2 = connector_capabilities(native)
        assert caps1 == caps2

    def test_rejects_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="source, sink, or both"):
            ConnectorCapability(
                provider="p",
                name="n",
                version="1",
                kind="bogus",
                capabilities=ConnectorCapabilities(
                    delivery="at_least_once",
                    replay="unreplayable",
                    watermark="generated_only",
                    transaction="none",
                    snapshot=False,
                    polling=False,
                    cdc=False,
                    lookup=False,
                ),
                formats=(),
                options_schema={},
            )

    def test_rejects_empty_provider(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            ConnectorCapability(
                provider="",
                name="n",
                version="1",
                kind="source",
                capabilities=ConnectorCapabilities(
                    delivery="at_least_once",
                    replay="unreplayable",
                    watermark="generated_only",
                    transaction="none",
                    snapshot=False,
                    polling=False,
                    cdc=False,
                    lookup=False,
                ),
                formats=(),
                options_schema={},
            )


class TestProjectV3Surface:
    def test_v3_types_are_re_exported(self) -> None:
        assert hasattr(calc_flow, "ConnectorCapability")
        assert hasattr(calc_flow, "ConnectorCapabilities")
        assert hasattr(calc_flow, "connector_capabilities")

    def test_v3_schema_constant(self) -> None:
        """The Python surface must declare v3 in the project format list."""
        from calc_flow.capabilities import runtime_capabilities

        caps = runtime_capabilities(
            session_id="test",
            revision=1,
            package_version="4.0.0",
            registrations=[],
        )
        assert caps.project_format_versions == (3,)
        assert tuple(operator.kind for operator in caps.operators) == (
            "cross_section",
            "expression",
            "rolling",
            "sql",
            "stream_join",
        )
        assert tuple(connector.name for connector in caps.connectors) == ("file",)
