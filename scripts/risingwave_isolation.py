"""Fail-closed network preflight for the Linux RisingWave comparison."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def _namespaces() -> dict:
    result = {}
    for kind in ("net", "pid"):
        parent = os.environ.get(f"CALC_FLOW_RW_PARENT_{kind.upper()}NS")
        current = os.readlink(f"/proc/self/ns/{kind}")
        if not parent or current == parent:
            raise RuntimeError(
                f"RisingWave isolation requires a private {kind} namespace"
            )
        result[kind] = {"parent": parent, "current": current}
    return result


def _capabilities() -> dict:
    capabilities = {
        key: value.strip()
        for line in Path("/proc/self/status").read_text().splitlines()
        for key, _, value in [line.partition(":")]
        if key.startswith("Cap")
    }
    required = {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}
    if capabilities.keys() != required or any(
        int(value, 16) for value in capabilities.values()
    ):
        raise RuntimeError("RisingWave isolation requires all capabilities dropped")
    return capabilities


def network_isolation() -> dict:
    """Reject host networking, extra interfaces, or retained network privileges."""
    namespaces = _namespaces()
    interfaces = [name for _, name in socket.if_nameindex()]
    if interfaces != ["lo"]:
        raise RuntimeError("RisingWave isolation requires only the loopback interface")
    capabilities = _capabilities()
    if Path("/proc/1/comm").read_text().strip() != "bwrap":
        raise RuntimeError("RisingWave isolation requires the bubblewrap PID reaper")
    return {
        "contract": "private-loopback-pid-v1",
        "namespaces": namespaces,
        "interfaces": interfaces,
        "capabilities": capabilities,
        "pid_reaper": "bwrap",
    }
