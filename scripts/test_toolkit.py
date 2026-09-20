"""Tests for the shared script toolkit helpers."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.toolkit import (
    FULL_SHA,
    canonical_json,
    command_output,
    fingerprint_json,
    git_output,
    linear_percentile,
    require_executable,
    sha256_file,
    wheel_native_sha256,
    write_json,
)


class HashingTests(unittest.TestCase):
    def test_sha256_file_matches_whole_file_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "payload.bin")
            path.write_bytes(b"calc-flow" * 4096)
            self.assertEqual(
                sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_wheel_native_sha256_hashes_the_single_native_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory, "calc_flow-5.0.0.whl")
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("calc_flow/__init__.py", "")
                archive.writestr("calc_flow/_native.abi3.so", b"native-bytes")
            self.assertEqual(
                wheel_native_sha256(wheel),
                hashlib.sha256(b"native-bytes").hexdigest(),
            )

    def test_wheel_native_sha256_requires_exactly_one_native_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for members in ((), ("calc_flow/_native_a.so", "calc_flow/_native_b.so")):
                wheel = Path(directory, f"wheel-{len(members)}.whl")
                with zipfile.ZipFile(wheel, "w") as archive:
                    for member in members:
                        archive.writestr(member, b"native-bytes")
                with (
                    self.subTest(members=members),
                    self.assertRaisesRegex(ValueError, "exactly one native module"),
                ):
                    wheel_native_sha256(wheel)


class FingerprintTests(unittest.TestCase):
    def test_canonical_json_sorts_keys_without_whitespace(self) -> None:
        value = {"b": 1, "a": [2, None]}
        self.assertEqual(canonical_json(value), '{"a":[2,null],"b":1}')

    def test_fingerprint_json_is_the_canonical_sha256(self) -> None:
        value = {"b": 1, "a": 2}
        expected = hashlib.sha256(b'{"a":2,"b":1}').hexdigest()
        self.assertEqual(fingerprint_json(value), expected)

    def test_fingerprint_json_keeps_non_ascii_by_default(self) -> None:
        value = {"name": "café"}
        self.assertEqual(
            fingerprint_json(value),
            hashlib.sha256('{"name":"café"}'.encode()).hexdigest(),
        )
        self.assertEqual(
            fingerprint_json(value, ensure_ascii=True),
            hashlib.sha256(b'{"name":"caf\\u00e9"}').hexdigest(),
        )


class PercentileTests(unittest.TestCase):
    def test_linear_percentile_interpolates_between_order_statistics(self) -> None:
        self.assertEqual(linear_percentile([4.0, 1.0, 3.0, 2.0], 0.25), 1.75)
        self.assertEqual(linear_percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertEqual(linear_percentile([10.0], 0.95), 10.0)

    def test_linear_percentile_matches_the_rust_harness_formula(self) -> None:
        samples = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
        ordered = sorted(samples)
        for fraction in (0.05, 0.25, 0.75, 0.95):
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = lower + 1 if position > lower else lower
            weight = position - lower
            expected = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
            with self.subTest(fraction=fraction):
                self.assertEqual(linear_percentile(samples, fraction), expected)


class CommandTests(unittest.TestCase):
    def test_command_output_requires_success_and_strips_stdout(self) -> None:
        completed = Mock(returncode=0, stdout="  result \n", stderr="")
        with patch("scripts.toolkit.subprocess.run", return_value=completed) as run:
            result = command_output(["git", "status"], cwd=Path("/repo"))
        self.assertEqual(result, "result")
        self.assertEqual(run.call_args.kwargs["check"], True)
        self.assertEqual(run.call_args.kwargs["shell"], False)

    def test_git_output_returns_stripped_stdout(self) -> None:
        completed = Mock(returncode=0, stdout="abc123\n", stderr="")
        with patch("scripts.toolkit.subprocess.run", return_value=completed):
            self.assertEqual(git_output(Path("/repo"), "rev-parse", "HEAD"), "abc123")

    def test_git_output_raises_value_error_with_stderr(self) -> None:
        completed = Mock(returncode=128, stdout="", stderr="fatal: bad ref")
        with (
            patch("scripts.toolkit.subprocess.run", return_value=completed),
            self.assertRaisesRegex(ValueError, "git rev-parse failed: fatal: bad ref"),
        ):
            git_output(Path("/repo"), "rev-parse", "HEAD")

    def test_require_executable_rejects_a_missing_command(self) -> None:
        with (
            patch("scripts.toolkit.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "gh executable is missing"),
        ):
            require_executable("gh")

    def test_full_sha_matches_only_lowercase_40_character_hex(self) -> None:
        self.assertIsNotNone(FULL_SHA.fullmatch("a" * 40))
        self.assertIsNone(FULL_SHA.fullmatch("A" * 40))
        self.assertIsNone(FULL_SHA.fullmatch("a" * 39))


class WriteJsonTests(unittest.TestCase):
    def test_write_json_creates_parents_and_sorts_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "nested", "out.json")
            write_json(path, {"b": 1, "a": 2})
            text = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text), {"a": 2, "b": 1})
            self.assertEqual(text, '{\n  "a": 2,\n  "b": 1\n}\n')

    def test_write_json_exclusive_refuses_an_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "out.json")
            write_json(path, {"a": 1}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write_json(path, {"a": 2}, exclusive=True)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})


if __name__ == "__main__":
    unittest.main()
