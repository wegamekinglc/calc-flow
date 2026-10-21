"""Preflight regression tests; no engine, network namespace, or native required."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.risingwave_isolation import network_isolation

ZERO_CAPABILITIES = "\n".join(
    f"{key}:\t00000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
)


class NetworkIsolationTests(unittest.TestCase):
    def test_missing_namespace_context_is_rejected(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("os.readlink", return_value="net:[100]"),
            self.assertRaisesRegex(RuntimeError, "private net"),
        ):
            network_isolation()

    def _snapshot(self, interfaces, status=ZERO_CAPABILITIES, reaper="bwrap"):
        with (
            patch.dict(
                "os.environ",
                {
                    "CALC_FLOW_RW_PARENT_NETNS": "net:[1]",
                    "CALC_FLOW_RW_PARENT_PIDNS": "pid:[1]",
                },
            ),
            patch("os.readlink", side_effect=("net:[2]", "pid:[2]")),
            patch("socket.if_nameindex", return_value=interfaces),
            patch("pathlib.Path.read_text", side_effect=(status, reaper)),
        ):
            return network_isolation()

    def test_extra_interface_is_rejected_even_with_namespace_marker(self):
        with self.assertRaisesRegex(RuntimeError, "loopback"):
            self._snapshot([(1, "lo"), (2, "eth0")])

    def test_retained_capability_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "capabilities"):
            self._snapshot(
                [(1, "lo")],
                ZERO_CAPABILITIES.replace("CapEff:\t00000000", "CapEff:\t00001000"),
            )

    def test_missing_pid_reaper_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "reaper"):
            self._snapshot([(1, "lo")], reaper="python")

    def test_valid_namespace_snapshot_is_recorded(self):
        snapshot = self._snapshot([(1, "lo")])
        self.assertEqual(snapshot["interfaces"], ["lo"])
        self.assertEqual(snapshot["contract"], "private-loopback-pid-v1")


if __name__ == "__main__":
    unittest.main()
