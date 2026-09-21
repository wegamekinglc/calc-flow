"""Fail closed on missing, corrupt, or incompatible comparison identities."""

from __future__ import annotations

import re

from scripts.toolkit import fingerprint_json

IDENTITIES = ("machine", "dependency", "workload")


def validate_identity(metadata: dict) -> None:
    for name in IDENTITIES:
        raw = metadata.get(f"{name}_identity")
        digest = metadata.get(f"{name}_fingerprint")
        if (
            not isinstance(raw, dict)
            or not raw
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or fingerprint_json(raw) != digest
        ):
            raise ValueError(
                f"incomparable {name} identity: missing or corrupt SHA-256"
            )

    workload = metadata["workload_identity"]
    for field in (
        "scenario",
        "scope",
        "backend",
        "scale",
        "input_rows",
        "output_rows",
        "table_rows",
        "array_elements",
        "matrix_dimension",
    ):
        if field in metadata and metadata[field] != workload.get(field):
            raise ValueError(
                f"incomparable workload identity: {field} disagrees with descriptor"
            )


def compare_identity(left: dict, right: dict) -> None:
    validate_identity(left)
    validate_identity(right)
    for name in IDENTITIES:
        if left[f"{name}_fingerprint"] != right[f"{name}_fingerprint"]:
            raise ValueError(f"incomparable {name} identity: fingerprint mismatch")
