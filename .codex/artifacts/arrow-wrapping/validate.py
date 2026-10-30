"""Focused negative checks for source seals, Arrow values and phase completeness."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from audit import audit_sample
from common import digest, read, save

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect  # noqa: E402


def source_guards(directory):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=directory, text=True, stderr=subprocess.DEVNULL
        ).strip()

    original = collect.ROOT
    collect.ROOT = directory
    try:
        product = directory / "Cargo.toml"
        product.write_text("fixed baseline\n")
        git("init", "-q")
        git("add", "Cargo.toml")
        git(
            "-c",
            "user.name=guard-test",
            "-c",
            "user.email=guard@example.invalid",
            "commit",
            "-qm",
            "fixture",
        )
        ref = git("rev-parse", "HEAD")
        collect.verify_product_source(ref)
        for staged in (False, True):
            product.write_text("changed product\n")
            if staged:
                git("add", "Cargo.toml")
            try:
                collect.verify_product_source(ref)
            except subprocess.CalledProcessError as error:
                assert error.returncode == 1
            else:
                raise AssertionError("changed product accepted")
        product.write_text("fixed baseline\n")
        try:
            collect.verify_product_source(ref)
        except subprocess.CalledProcessError as error:
            assert error.returncode == 1
        else:
            raise AssertionError("staged change hidden by working tree accepted")
        git("add", "Cargo.toml")
        (directory / "vendor").mkdir()
        (directory / "vendor/new.rs").write_text("// untracked product\n")
        try:
            collect.verify_product_source(ref)
        except ValueError:
            pass
        else:
            raise AssertionError("untracked product accepted")
    finally:
        collect.ROOT = original


def expect_rejected(directory, sample, seals, reason):
    try:
        audit_sample(directory, sample, seals)
    except ValueError as error:
        assert reason in str(error), str(error)
    else:
        raise AssertionError("invalid evidence accepted")


def validate(task):
    seals = read(task / "seals.json")
    collect.verify_product_source(seals["head"]["source_ref"])
    sample = read(task / "results/samples.json")[0]
    name = f"{sample['case']}-r0-p0-base"
    with tempfile.TemporaryDirectory(dir=task, prefix="negative-") as temp:
        directory = Path(temp)
        source = directory / "source"
        source.mkdir()
        source_guards(source)
        damaged = directory / "values"
        shutil.copytree(task / "results" / name, damaged)
        table = pa.ipc.open_stream(damaged / "output.arrow").read_all()
        table = table.set_column(0, table.schema.field(0), pc.add(table[0], 1))
        temporary = damaged / "altered.arrow"
        with pa.ipc.new_stream(temporary, table.schema) as writer:
            writer.write_table(table)
        temporary.replace(damaged / "output.arrow")
        changed = copy.deepcopy(sample)
        changed["output_sha256"] = digest(damaged / "output.arrow")
        expect_rejected(damaged, changed, seals, "complete Arrow")
        clocks = directory / "clocks"
        shutil.copytree(task / "results" / name, clocks)
        changed = copy.deepcopy(sample)
        changed["worker"]["plan_nested_ns"]["normal_input_binding_ns"].pop()
        save(clocks / "worker.json", changed["worker"])
        expect_rejected(clocks, changed, seals, "missing normal API span")
    save(
        task / "negative-validation.json",
        {
            "clean_product": "accepted",
            "staged_product": "rejected",
            "unstaged_product": "rejected",
            "staged_change_with_baseline_worktree": "rejected",
            "untracked_vendor": "rejected",
            "wrong_values_with_updated_hash": "rejected",
            "missing_input_binding_clock": "rejected",
        },
    )
    print("source guards and both evidence counterexamples passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    validate(parser.parse_args().task)
