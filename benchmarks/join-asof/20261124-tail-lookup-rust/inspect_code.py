"""Record optimized code identity for the maintained direct ASOF callback."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def command(*args):
    return subprocess.check_output(args, text=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect(executable):
    with tempfile.TemporaryDirectory() as work:
        section = Path(work) / "text.bin"
        # Supplying a separate output file keeps the sealed ELF byte-identical.
        command("objcopy", f"--dump-section=.text={section}", executable, Path(work) / "copy.elf")
        text_sha = digest(section.read_bytes())
    symbols = command("nm", "-C", "-S", "--defined-only", executable).splitlines()
    (function,) = (line for line in symbols if "calc_flow::operator::asof::finalize::" in line and line.endswith("::prepare_output"))
    lo, length, _, name = function.split(maxsplit=3)
    base, count = int(lo, 16), int(length, 16)
    listing = command("objdump", "-d", "--no-show-raw-insn", "--no-addresses", "-C", f"--start-address={base}", f"--stop-address={base+count}", executable)
    instructions = [s.strip() for s in listing.splitlines() if s.startswith("\t")]
    if len(instructions) < 100:
        raise ValueError("unexpected ASOF disassembly")
    opcodes = [s.split()[0] for s in instructions]
    return {
        "binary_sha256": digest(executable.read_bytes()),
        "text_section_sha256": text_sha,
        "function": name,
        "start": lo,
        "function_size_bytes": count,
        "instruction_count": len(instructions),
        "opcode_sequence_sha256": digest(("\n".join(opcodes)+"\n").encode()),
        "note": "The caller includes other settlement work. Instruction mnemonics omit operands and relocation targets. Neither code size nor this hash counts dynamically executed right-timestamp reads.",
    }


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1])), indent=2, sort_keys=True))
