
"""
Tensor PV coefficient functions that reduce to scalar PV functions.

Each class stores the reduction formula in reduce() (symbolic, keeps
scalar PV functions unevaluated) and delegates _eval() through that formula
so that doit() returns the fully-expanded analytic expression.

Naming convention: Denner 
  - subscript 0   → metric slot (g^{μν} factor)
  - subscript 1,2 → external momentum label p_1, p_2
  Contrast with Bardin & Passarino / original PV (1979):
    Denner B_{00} = PV B_{22},  Denner B_{11} = PV B_{21}
"""

import sympy as sp
from .scalar import A0, B0
from ..base import PVFunction


class A00(PVFunction):
    """
    Rank-2 tadpole tensor coefficient: int k^mu k^nu / (k^2-m^2) = g^{munu} A_{00}(m)

    Reduction (from d * A_{00} = m^2 * A_0, expanded to O(epsilon_bar^0) in d = 4 - 2*epsilon_bar):
        A_{00}(m) = m^2/4 * A_0(m) + m^4/8
    """
    nargs = 1
    latex_name = r"A_{00}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        """Return A_{00}(m) as a sympy expression in terms of A_0(m) (A_0 kept symbolic)."""
        m = self.args[0]
        return sp.Rational(1, 4) * m**2 * A0(m) + sp.Rational(1, 8) * m**4

    def _derivative(self, sym):
        m = self.args[0]
        if sym not in self.free_symbols:
            return sp.S.Zero
        # d/dm [m^2/4 * A0(m) + m^4/8] -- A0 has no m-dependence in its derivative (returns 0)
        return sp.Rational(1, 2) * m * A0(m) + sp.Rational(1, 2) * m**3

    def _eval(self, part="full", **hints):
        m = sp.simplify(self.args[0])
        a0 = A0(m)._eval(part)
        if part == "pole":
            # m^4/8 is finite; only A_0 carries the UV pole
            return sp.Rational(1, 4) * m**2 * a0
        else:  # "full" or "finite"
            return sp.Rational(1, 4) * m**2 * a0 + sp.Rational(1, 8) * m**4


class B1(PVFunction):
    """
    Rank-1 bubble tensor coefficient: int k^mu / (D_0 D_1) = p^mu B_1(p^2, m0, m1)

    Convention: D_0 = k^2 - m0^2,  D_1 = (k+p)^2 - m1^2  (shift in propagator 1).

    Reduction derived from 2k·p = D_1 - D_0 + (m1^2 - m0^2 - p^2):
        B_1(p^2, m0, m1) = [A_0(m0) - A_0(m1) - (p^2 + m0^2 - m1^2) B_0] / (2 p^2)

    The numerator vanishes at p^2=0 (it is a 0/0 limit), so B_1 is finite there,
    but the reduction formula itself is undefined at p^2=0 — see the p^2=0 guard
    in reduce().
    """
    nargs = 3  # (p^2, m0, m1)
    latex_name = r"B_1"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        """Return B_1 as a sympy expression in terms of A_0 and B_0 (kept symbolic)."""
        p2, m0, m1 = self.args
        if p2 == sp.S.Zero:
            raise NotImplementedError(
                "B1(0, m0, m1) requires a separate limiting formula "
                "(B1 ~ dB0/dp² at p²→0) which is not yet implemented. "
                "Call set_kinematics() before reduce_pv() so that factors "
                "like p²·B1(p²,...) vanish to 0·B1(0,...) = 0 before "
                "the reduction formula is applied."
            )
        return (A0(m0) - A0(m1) - (p2 + m0**2 - m1**2) * B0(p2, m0, m1)) / (2 * p2)

    def _derivative(self, sym):
        raise NotImplementedError

    def _eval(self, part = "full", **hints):
        p2, m0, m1 = [sp.simplify(a) for a in self.args]
        a0_m0 = A0(m0)._eval(part)
        a0_m1 = A0(m1)._eval(part)
        b0_val = B0(p2, m0, m1)._eval(part)
        return (a0_m0 - a0_m1 - (p2 + m0**2 - m1**2) * b0_val) / (2 * p2)
