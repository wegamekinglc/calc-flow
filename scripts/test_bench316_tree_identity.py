from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.tree_identity_diagnostic import (
    activate_tree,
    identity_plan,
    maps_match_native,
    run,
    tree_fingerprint,
)


class TreeIdentityTests(unittest.TestCase):
    def test_identity_plan_balances_original_and_copy(self):
        self.assertEqual(
            identity_plan(),
            ("original_1", "copy_1", "copy_2", "original_2"),
        )

    def test_identical_copy_changes_inodes_and_round_trips_at_same_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "install" / "B1" / "site"
            active.mkdir(parents=True)
            native = active / "calc_flow" / "_native.abi3.so"
            native.parent.mkdir()
            native.write_bytes(b"same-native")
            (active / "metadata.txt").write_text("same-metadata")
            copy = root / "parked" / "copy"
            from shutil import copytree

            copytree(active, copy)
            original_fingerprint = tree_fingerprint(active)
            copied_fingerprint = tree_fingerprint(copy)
            self.assertEqual(original_fingerprint["files"], copied_fingerprint["files"])
            self.assertNotEqual(
                original_fingerprint["native"]["inode"],
                copied_fingerprint["native"]["inode"],
            )

            parked_original = root / "parked" / "original"
            activate_tree(active, parked_original, copy)
            self.assertEqual(
                tree_fingerprint(active)["native"]["inode"],
                copied_fingerprint["native"]["inode"],
            )
            activate_tree(active, copy, parked_original)
            self.assertEqual(
                tree_fingerprint(active)["native"]["inode"],
                original_fingerprint["native"]["inode"],
            )
            self.assertEqual(active.resolve(), root / "install" / "B1" / "site")

    def test_maps_identity_requires_native_inode_and_path(self):
        site = Path("/runner/install/B1/site")
        maps = [
            "7f00-7f10 r-xp 00000000 08:01 123 "
            "/runner/install/B1/site/calc_flow/_native.abi3.so",
            "7f10-7f20 r--p 00000000 08:01 999 "
            "/runner/install/A0/site/calc_flow/_native.abi3.so",
        ]
        self.assertTrue(maps_match_native(maps, site, 123))
        self.assertFalse(maps_match_native(maps, site, 999))
        self.assertFalse(maps_match_native(maps, site.parent, 123))

    def test_maps_identity_resolves_relative_install_path(self):
        site = Path("target/bench316-results/install/B1/site")
        native = site.resolve() / "calc_flow" / "_native.abi3.so"
        maps = [f"7f00-7f10 r-xp 00000000 08:01 123 {native}"]

        self.assertTrue(maps_match_native(maps, site, 123))


class TreeIdentityRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_biased_probe_keeps_paths_and_returns_original_inode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            observations = []

            async def fake_install(release, install_root):
                site = install_root / "site"
                native = site / "calc_flow" / "_native.abi3.so"
                native.parent.mkdir(parents=True)
                native.write_bytes(release["identity"].encode())
                record = site / "wheel.dist-info" / "RECORD"
                record.parent.mkdir()
                record.write_text(release["identity"])
                return site

            async def fake_measure(_case, _kind, sites, _releases, result_root):
                observations.append(
                    (
                        result_root.name,
                        str(sites["candidate"]),
                        (sites["candidate"] / "calc_flow" / "_native.abi3.so")
                        .stat()
                        .st_ino,
                    )
                )
                changes = (
                    [13.0, 14.0] if result_root.name == "ab_probe_0" else [0.0, 0.0]
                )
                return {
                    "status": "ok",
                    "result": {"round_changes": changes},
                    "evidence": [],
                }

            releases = iter(
                (
                    {
                        "identity": "A",
                        "wheel_sha256": "a" * 64,
                        "native_sha256": "same",
                    },
                    {
                        "identity": "B",
                        "wheel_sha256": "b" * 64,
                        "native_sha256": "same",
                    },
                )
            )
            with (
                patch(
                    "scripts.benchmark_suite.tree_identity_diagnostic.load_release",
                    side_effect=lambda _: next(releases),
                ),
                patch(
                    "scripts.benchmark_suite.tree_identity_diagnostic.install",
                    side_effect=fake_install,
                ),
                patch(
                    "scripts.benchmark_suite.tree_identity_diagnostic.shard_cases",
                    return_value=[{"id": "engines/100000/calc-flow-stream/group_by"}],
                ),
                patch(
                    "scripts.benchmark_suite.tree_identity_diagnostic.get_shard",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.measure.measure_case",
                    side_effect=fake_measure,
                ),
            ):
                await run(root / "base.json", root / "head.json", root / "out")

            self.assertEqual(
                [observation[0] for observation in observations],
                ["ab_probe_0", "original_1", "copy_1", "copy_2", "original_2"],
            )
            self.assertEqual(len({observation[1] for observation in observations}), 1)
            original_inode = observations[1][2]
            copied_inode = observations[2][2]
            self.assertNotEqual(original_inode, copied_inode)
            self.assertEqual(observations[3][2], copied_inode)
            self.assertEqual(observations[4][2], original_inode)


if __name__ == "__main__":
    unittest.main()
