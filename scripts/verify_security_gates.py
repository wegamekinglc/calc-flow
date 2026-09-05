"""M7-02 security and supply-chain gate runner.

Validates the threat-model coverage checklist and invokes the audit
tools (cargo audit, cargo deny, npm audit) against the workspace with
all connector features enabled. Documents per-threat evidence linking
to the named tests that enforce each boundary.

Usage:
    uv run python scripts/verify_security_gates.py --checklist-only
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404 -- fixed, module-owned audit commands only
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ThreatEntry:
    """One threat-model boundary and its enforcement evidence."""

    threat: str
    boundary: str
    evidence: str


THREAT_MODEL: tuple[ThreatEntry, ...] = (
    ThreatEntry(
        threat="secret-value-in-config",
        boundary=(
            "Connector options and project documents carry only SecretRef values; "
            "no field can hold a credential"
        ),
        evidence=(
            "config project-v3 secret-ref tests; connector from_options tests across "
            "postgresql/clickhouse/http/kafka"
        ),
    ),
    ThreatEntry(
        threat="credential-leak-in-error",
        boundary=(
            "URL errors are truncated; secret resolution failures never carry the value"
        ),
        evidence=(
            "http url_redaction_truncates_errors; "
            "postgresql connection_url_only_from_secrets; "
            "clickhouse url_only_from_secrets"
        ),
    ),
    ThreatEntry(
        threat="path-traversal",
        boundary=(
            "File source paths reject .. components; identifiers are lowercase-only"
        ),
        evidence=(
            "file_connector discovery_fails_closed; pg/ch identifiers_reject_injection"
        ),
    ),
    ThreatEntry(
        threat="symlink-traversal",
        boundary="File discovery rejects symlinked entries and symlinked roots",
        evidence="file_connector symlinked_entries_fail_closed (unix)",
    ),
    ThreatEntry(
        threat="decompression-bomb",
        boundary=(
            "Parquet row groups are bounded; decode expansion is checked "
            "against DecodeBounds"
        ),
        evidence=(
            "parquet row-group bound tests; DecodeBounds row/byte checks in all codecs"
        ),
    ),
    ThreatEntry(
        threat="oversized-message",
        boundary=(
            "HTTP response body and WebSocket frames have configurable byte limits"
        ),
        evidence=(
            "http max_response_bytes; websocket max_frame_bytes; "
            "JSON line max_batch_bytes"
        ),
    ),
    ThreatEntry(
        threat="malicious-schema",
        boundary=(
            "Project v3 schema is generated and every layer denies unknown "
            "fields; type matrix rejects unknown types"
        ),
        evidence="config_v3 unknown-field tests; database_types matrix rejection tests",
    ),
    ThreatEntry(
        threat="deep-connector-option",
        boundary=(
            "Connector options are bounded JSON; project transport validates JSON depth"
        ),
        evidence="models _copy_json_value max depth 32; connector option validation",
    ),
    ThreatEntry(
        threat="sql-identifier-injection",
        boundary="Table and column names pass through lowercase identifier validation",
        evidence="pg identifiers_reject_injection; ch identifiers_reject_injection",
    ),
    ThreatEntry(
        threat="sql-query-injection",
        boundary=(
            "Cursor predicates use typed PostgreSQL parameters and ClickHouse "
            "query parameters"
        ),
        evidence=(
            "postgresql composite_cursor_uses_lexicographic_row_comparison; "
            "clickhouse "
            "cursor_values_are_query_parameters_and_snapshot_bound_is_composite"
        ),
    ),
    ThreatEntry(
        threat="replication-slot-abuse",
        boundary=(
            "CDC slot create, require-existing, and explicit inactive-slot "
            "replacement policies fail closed on lost slots, recycled WAL, "
            "active replacement, and schema mismatch"
        ),
        evidence=(
            "postgresql_cdc exported_snapshot_hands_off_to_pgoutput_without_a_gap; "
            "postgresql_cdc preflight_slot and drop_inactive_slot"
        ),
    ),
    ThreatEntry(
        threat="wal-retention-growth",
        boundary=(
            "The replication client reports only the manifest-durable LSN and "
            "preflight rejects "
            "a slot ahead of durable state or a restart LSN whose WAL was recycled"
        ),
        evidence=(
            "postgresql_cdc cursor_ack_is_monotonic_and_slot_bound; "
            "postgresql_cdc preflight_slot"
        ),
    ),
    ThreatEntry(
        threat="database-ledger-forgery",
        boundary=(
            "Epoch ledger commits are in the same transaction as data; "
            "recovery validates identity"
        ),
        evidence="postgresql ledger ON CONFLICT; sink recovery evidence validation",
    ),
    ThreatEntry(
        threat="kafka-ledger-forgery",
        boundary=(
            "Transactional IDs are derived from pipeline/output identity; "
            "recovery accepts only matching checksummed state segments and "
            "markers from a dedicated compacted ledger"
        ),
        evidence=(
            "kafka sink_config_derives_transactional_identity; "
            "kafka kafka_roundtrip_and_transactional_exactly_once"
        ),
    ),
    ThreatEntry(
        threat="clickhouse-dedup-token-forgery",
        boundary=(
            "Dedup tokens derive from pipeline/output/epoch identity; not user-supplied"
        ),
        evidence="clickhouse dedup_tokens_are_stable_and_distinct_per_epoch",
    ),
    ThreatEntry(
        threat="tls-disabled-by-default",
        boundary=(
            "TLS verification defaults to on; insecure mode is explicit with a warning"
        ),
        evidence=(
            "http_config_parses_and_defaults_tls_on; websocket insecure flag tests"
        ),
    ),
    ThreatEntry(
        threat="state-cleanup-symlink",
        boundary="Checkpoint/state cleanup paths operate on managed directories only",
        evidence="checkpoint store tests; state segment path handling",
    ),
    ThreatEntry(
        threat="format-decoder-fuzz",
        boundary=(
            "Property tests cover project, checkpoint, format, and "
            "state metadata decoders"
        ),
        evidence="crates/calc-flow/tests/properties.rs; window_properties.rs",
    ),
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """One repository-relative source file and named enforcing test."""

    path: str
    symbol: str


THREAT_EVIDENCE: dict[str, tuple[EvidenceRef, ...]] = {
    "secret-value-in-config": (
        EvidenceRef(
            "crates/calc-flow/tests/project_connector_compile.rs",
            "stream_compile_validates_required_secret_slots_without_opening_factory",
        ),
    ),
    "credential-leak-in-error": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/http_websocket_connectors.rs",
            "url_redaction_truncates_errors",
        ),
    ),
    "path-traversal": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/file_connector.rs",
            "discovery_fails_closed",
        ),
    ),
    "symlink-traversal": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/file_connector.rs",
            "symlinked_entries_fail_closed",
        ),
    ),
    "decompression-bomb": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/file_connector.rs",
            "parquet_row_group_bound_names_the_row_group",
        ),
    ),
    "oversized-message": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/http_websocket_connectors.rs",
            "websocket_drop_oldest_is_bounded_and_observable",
        ),
    ),
    "malicious-schema": (
        EvidenceRef(
            "crates/calc-flow/tests/config.rs",
            "project_rejects_v1_and_unknown_fields_at_every_nested_level",
        ),
    ),
    "deep-connector-option": (
        EvidenceRef(
            "crates/calc-flow/tests/project_store.rs",
            "project_json_import_enforces_the_full_document_depth_limit",
        ),
    ),
    "sql-identifier-injection": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/postgresql_connector.rs",
            "identifiers_reject_injection",
        ),
    ),
    "sql-query-injection": (
        EvidenceRef(
            "crates/calc-flow-connectors/src/postgresql.rs",
            "composite_cursor_uses_lexicographic_row_comparison",
        ),
    ),
    "replication-slot-abuse": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/postgresql_cdc.rs",
            "exported_snapshot_hands_off_to_pgoutput_without_a_gap",
        ),
    ),
    "wal-retention-growth": (
        EvidenceRef(
            "crates/calc-flow-connectors/src/postgresql_cdc.rs",
            "cursor_ack_is_monotonic_and_slot_bound",
        ),
    ),
    "database-ledger-forgery": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/postgresql_connector.rs",
            "snapshot_reads_and_transactional_sink_commits",
        ),
    ),
    "kafka-ledger-forgery": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/kafka_connector.rs",
            "transactional_sink_recovery_validates_identity_evidence",
        ),
    ),
    "clickhouse-dedup-token-forgery": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/clickhouse_connector.rs",
            "dedup_tokens_are_stable_and_distinct_per_epoch",
        ),
    ),
    "tls-disabled-by-default": (
        EvidenceRef(
            "crates/calc-flow-connectors/tests/http_websocket_connectors.rs",
            "http_config_parses_and_defaults_tls_on",
        ),
    ),
    "state-cleanup-symlink": (
        EvidenceRef(
            "crates/calc-flow/tests/local_state.rs",
            "collection_stops_on_a_symbolic_link_without_deleting_valid_state",
        ),
    ),
    "format-decoder-fuzz": (
        EvidenceRef(
            "crates/calc-flow/tests/properties.rs",
            "canonical_json_round_trips_recursively_sorted_values",
        ),
    ),
}


# Evidence validation intentionally accumulates every stale mapping and symbol
# so CI reports the whole deterministic security checklist in one run.
def validate_evidence() -> tuple[str, ...]:
    """Returns stable validation failures for stale or missing evidence."""
    # #lizard forgives
    failures: list[str] = []
    threats = {entry.threat for entry in THREAT_MODEL}
    mapped = set(THREAT_EVIDENCE)
    for missing in sorted(threats - mapped):
        failures.append(f"{missing}: no machine-checked evidence")
    for unknown in sorted(mapped - threats):
        failures.append(f"{unknown}: evidence has no threat entry")
    for threat in sorted(threats & mapped):
        references = THREAT_EVIDENCE[threat]
        if not references:
            failures.append(f"{threat}: evidence list is empty")
            continue
        for reference in references:
            path = Path(reference.path)
            if path.is_absolute() or ".." in path.parts:
                failures.append(f"{threat}: unsafe evidence path {reference.path}")
                continue
            resolved = REPOSITORY_ROOT / path
            try:
                source = resolved.read_text(encoding="utf-8")
            except OSError:
                failures.append(f"{threat}: missing evidence file {reference.path}")
                continue
            if reference.symbol not in source:
                failures.append(
                    f"{threat}: missing symbol {reference.symbol} in {reference.path}"
                )
    return tuple(failures)


@dataclass(frozen=True, slots=True)
class AuditCommand:
    """One release audit command and its explicit working directory."""

    name: str
    argv: tuple[str, ...]
    cwd: Path


AUDIT_COMMANDS: tuple[AuditCommand, ...] = (
    AuditCommand(
        name="cargo audit",
        argv=(
            "cargo",
            "audit",
            "--ignore",
            "RUSTSEC-2026-0176",
            "--ignore",
            "RUSTSEC-2026-0177",
            "--ignore",
            "RUSTSEC-2026-0235",
        ),
        cwd=REPOSITORY_ROOT,
    ),
    AuditCommand(
        name="cargo deny",
        argv=("cargo", "deny", "--locked", "check"),
        cwd=REPOSITORY_ROOT,
    ),
    AuditCommand(
        name="npm audit",
        argv=("npm", "audit", "--omit=dev"),
        cwd=REPOSITORY_ROOT / "web-ui",
    ),
)


def print_checklist() -> None:
    """Prints the threat-model coverage checklist."""
    print("M7-02 Threat-Model Coverage Checklist")
    print("=" * 60)
    for entry in THREAT_MODEL:
        print(f"\n  Threat: {entry.threat}")
        print(f"  Boundary: {entry.boundary}")
        print(f"  Evidence: {entry.evidence}")
        for reference in THREAT_EVIDENCE[entry.threat]:
            print(f"  Checked: {reference.path}::{reference.symbol}")
    print(f"\n  Total: {len(THREAT_MODEL)} threats covered")
    print("\nAudit Commands (run in CI):")
    for command in AUDIT_COMMANDS:
        print(f"  {command.name}: {' '.join(command.argv)}")


def run_audits() -> int:
    """Runs every declared audit and returns the first failing status."""
    for command in AUDIT_COMMANDS:
        print(f"\nRUN: {command.name}")
        try:
            # AUDIT_COMMANDS is an immutable, module-owned allowlist; no request,
            # environment, or project value can alter the executable or argv.
            subprocess.run(  # nosec B603  # nosemgrep
                list(command.argv), cwd=command.cwd, check=True
            )
        except subprocess.CalledProcessError as error:
            return error.returncode or 1
        except OSError as error:
            print(f"FAILED TO START: {command.name}: {error}")
            return 127
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checklist-only",
        action="store_true",
        help="Print the threat-model checklist without running audits",
    )
    options = parser.parse_args()

    failures = validate_evidence()
    if failures:
        print("M7-02 evidence validation failed:")
        for failure in failures:
            print(f"  {failure}")
        return 2

    if options.checklist_only:
        print_checklist()
        return 0

    print_checklist()
    return run_audits()


if __name__ == "__main__":
    raise SystemExit(main())
