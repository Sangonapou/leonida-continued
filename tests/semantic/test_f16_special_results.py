"""SHADER-F16-SPECIAL-RESULTS: the f16 transcendental inputs where SPIR-V leaves the result undefined, but RDNA2 defines
it (docs/design.md, "f16 transcendentals", measured on all 65,536 inputs with tools/f16_trans_sweep.py):

  - v_log_f16(+-0) = -inf (0xfc00), whatever the sign of the zero;
  - v_log_f16 of a negative operand (-inf and negative denormals included) = 0xfe00, a negative quiet NaN;
  - v_sin_f16(+-inf) and v_cos_f16(+-inf) = 0xfe00.

GLSL.std.450 `Log2` is undefined for x <= 0, and `Sin`/`Cos` of +-inf reach the driver as a NaN of unspecified sign,
so the translator replaces the operand / selects the hardware word itself. The words are compared bit for bit here
(the shared f16 harness compares NaNs by class only). Every case is translated by tools/rdna2spirv.py and executed on
tests/semantic/spirv_mini_interp.py. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f16_ref import OLD  # noqa: E402
from f16_unary_cases import run  # noqa: E402

POS0, NEG0, ONE, NEG_ONE, INF, NINF = 0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00
NEG_NAN = 0xFE00

LOG_SPECIAL = [
    (POS0, NINF), (NEG0, NINF),                       # log(+-0) = -inf
    (NEG_ONE, NEG_NAN), (0xC400, NEG_NAN),            # log(-1), log(-4)
    (NINF, NEG_NAN), (0x8001, NEG_NAN),               # log(-inf), log of the smallest negative denormal
    (0xFBFF, NEG_NAN),                                # log(-65504)
]
TRIG_SPECIAL = [(INF, NEG_NAN), (NINF, NEG_NAN)]
REGULAR = [  # (op, S0, D): ordinary inputs keep the function's own, correctly rounded result
    ("v_log_f16", 0x4800, 0x4200),   # log2(8) = 3
    ("v_log_f16", ONE, POS0),        # log2(1) = +0
    ("v_log_f16", 0x3800, NEG_ONE),  # log2(0.5) = -1
    ("v_log_f16", INF, INF),         # log2(+inf) = +inf
    ("v_log_f16", 0x0001, 0xCE00),   # log2(2^-24) = -24 (a positive denormal is not a zero)
    ("v_sin_f16", 0x3400, ONE),      # sin(0.25 turn) = 1
    ("v_sin_f16", NEG0, NEG0),       # sin(-0) = -0
    ("v_cos_f16", POS0, ONE),        # cos(0) = 1
    ("v_cos_f16", 0x3800, NEG_ONE),  # cos(0.5 turn) = -1
]


class F16SpecialResults(unittest.TestCase):
    def check(self, line, seeds, want, old=OLD, keep_high=True):
        got = run(line, seeds, old=old)
        if keep_high:
            self.assertEqual(got >> 16, old >> 16, f"{line}: high half not kept, {got:#010x}")
        self.assertEqual(got & 0xFFFF, want, f"{line} {seeds}: got {got & 0xFFFF:#06x}, want {want:#06x}")

    def test_log_of_zero_or_negative_gives_hardware_words(self):
        for a, want in LOG_SPECIAL:
            for suffix in ("_e32", "_e64"):
                with self.subTest(a=hex(a), suffix=suffix):
                    self.check(f"v_log_f16{suffix} v0, v1", {"v1": a}, want)

    def test_sin_and_cos_of_infinity_give_hardware_word(self):
        for op in ("v_sin_f16", "v_cos_f16"):
            for a, want in TRIG_SPECIAL:
                for suffix in ("_e32", "_e64"):
                    with self.subTest(op=op, a=hex(a), suffix=suffix):
                        self.check(f"{op}{suffix} v0, v1", {"v1": a}, want)

    def test_regular_inputs_unchanged(self):
        for op, a, want in REGULAR:
            with self.subTest(op=op, a=hex(a)):
                self.check(f"{op} v0, v1", {"v1": a}, want)

    def test_source_modifiers_apply_before_the_special_case(self):
        # The special case is decided on the modified operand.
        cases = [
            ("v_log_f16_e64 v0, -v1", ONE, NEG_NAN),      # log(-1)
            ("v_log_f16_e64 v0, |v1|", NEG0, NINF),       # log(+0)
            ("v_log_f16_e64 v0, |v1|", NEG_ONE, POS0),    # log(1): not a negative any more
            ("v_log_f16_e64 v0, -v1", NEG0, NINF),        # log(+0)
            ("v_sin_f16_e64 v0, -v1", INF, NEG_NAN),      # sin(-inf)
            ("v_cos_f16_e64 v0, |v1|", NINF, NEG_NAN),    # cos(+inf)
        ]
        for line, a, want in cases:
            with self.subTest(line=line, a=hex(a)):
                self.check(line, {"v1": a}, want)

    def test_op_sel_source_and_destination_halves(self):
        # op_sel:[1,0] reads the high half of v1; op_sel:[0,1] writes the high half of v0 and keeps the low half.
        for op, a, want in [("v_log_f16", NEG0, NINF), ("v_log_f16", NEG_ONE, NEG_NAN), ("v_sin_f16", INF, NEG_NAN),
                            ("v_cos_f16", NINF, NEG_NAN)]:
            with self.subTest(op=op, a=hex(a), sel="src hi"):
                self.check(f"{op}_e64 v0, v1 op_sel:[1,0]", {"v1": (a << 16) | 0x4800}, want)
            with self.subTest(op=op, a=hex(a), sel="dst hi"):
                got = run(f"{op}_e64 v0, v1 op_sel:[0,1]", {"v1": a})
                self.assertEqual(got & 0xFFFF, OLD & 0xFFFF, hex(got))
                self.assertEqual(got >> 16, want, hex(got))


if __name__ == "__main__":
    unittest.main()
