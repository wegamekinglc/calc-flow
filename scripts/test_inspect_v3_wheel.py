from __future__ import annotations

import unittest

if __package__:
    from scripts.inspect_v3_wheel import find_vector_instructions
else:
    from inspect_v3_wheel import find_vector_instructions


class InspectV3WheelTests(unittest.TestCase):
    def test_finds_vector_instructions_in_a_designated_numeric_loop(self) -> None:
        disassembly = """
0000000000001000 <calc_flow::operator::rolling::kernel::mean::fill_mean_rows>:
    1000: c5 fd 59 c1             vmulpd %ymm1,%ymm0,%ymm0
    1004: c5 fb 10 0a             vmovsd (%rdx),%xmm1
0000000000002000 <calc_flow::operator::keyed::hash_rows>:
    2000: c5 fe 7f 00             vmovdqu %ymm0,(%rax)
"""

        instructions = find_vector_instructions(disassembly)

        self.assertEqual(len(instructions), 1)
        self.assertIn("fill_mean_rows", instructions[0])
        self.assertIn("vmulpd", instructions[0])

    def test_rejects_scalar_only_or_unrelated_vector_code(self) -> None:
        disassembly = """
0000000000001000 <calc_flow::operator::rolling::kernel::mean::fill_mean_rows>:
    1000: c5 fb 59 c1             vmulsd %xmm1,%xmm0,%xmm0
    1004: c5 f8 77                vzeroupper
0000000000002000 <unrelated::copy>:
    2000: c5 fd 59 c1             vmulpd %ymm1,%ymm0,%ymm0
"""

        with self.assertRaisesRegex(ValueError, "designated P2/P4 numeric loop"):
            find_vector_instructions(disassembly)


if __name__ == "__main__":
    unittest.main()
