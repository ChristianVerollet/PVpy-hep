"""
One-point (tadpole) PV functions: A0, A00, A0000.

All one-point functions have exact closed-form analytic expressions that are
numerically stable for all kinematics (no p² dependence, single mass argument).
No Feynman-parameter integration is needed here.

Formulas from guide.tex §3.1 / §4.1:
    A0(m)    = m²(1 + 1/ε̄ − ln(m²/μ²))
    A00(m)   = m²/4 · A0(m) + m⁴/8
    A0000(m) = m⁴/24 · A0(m) + 5m⁶/144
"""

import sympy as sp
import numpy as np
from ..base import PVFunction
from ..symbols import epsilon_bar, mu

_atol = 1e-8  # mass tolerance: |m| < _atol treated as zero


# ---------------------------------------------------------------------------
# A0 — scalar tadpole
# ---------------------------------------------------------------------------

class A0(PVFunction):

    nargs = 1
    latex_name = r"A_0"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        if sp.sympify(args[0]).is_zero:
            return sp.S.Zero
        return super().__new__(cls, *args)

    def _derivative(self, _):
        return sp.S.Zero

    def _eval_massless(self, part="full"):
        return 0

    def _eval_massive(self, part="full"):
        m = sp.simplify(self.args[0])
        if part == "full":
            return sp.sympify(m**2 * (1 + 1/epsilon_bar - sp.log(m**2 / mu**2)))
        elif part == "pole":
            return sp.sympify(m**2) / epsilon_bar
        elif part == "finite":
            return sp.sympify(m**2) * (1 - sp.log(m**2 / mu**2))

    def _eval(self, part="full", **hints):
        m = sp.simplify(self.args[0])
        return sp.Piecewise(
            (self._eval_massless(part), sp.Eq(m, 0)),
            (self._eval_massive(part),  True),
        )

    def numeric(self, kernel="numpy"):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(mv, *, mu_val, part):

        def f_m_zero(m, mu_val, part):
            return 0

        def f_gen(m, mu_val, part):
            if part == "pole":
                return m**2
            elif part == "finite":
                return m**2 * (1 - np.log(m**2 / mu_val**2))

        mv = np.asarray(mv, dtype=float)
        res = np.empty_like(mv, dtype=float)

        mask_zero  = np.abs(mv) <= _atol
        mask_nzero = ~mask_zero

        if np.any(mask_zero):
            res[mask_zero]  = f_m_zero(mv[mask_zero],  mu_val, part)
        if np.any(mask_nzero):
            res[mask_nzero] = f_gen(mv[mask_nzero], mu_val, part)

        return res


# ---------------------------------------------------------------------------
# A00 — rank-2 tadpole tensor coefficient
# ---------------------------------------------------------------------------

class A00(PVFunction):
    """
    int k^mu k^nu / (k²-m²)  =  g^{μν} A_{00}(m)

    Reduction (guide.tex §3.1 / §4.1, d=4):
        A_{00}(m) = m²/4 · A_0(m) + m⁴/8
    """
    nargs = 1
    latex_name = r"A_{00}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        m = self.args[0]
        return sp.Rational(1, 4) * m**2 * A0(m) + sp.Rational(1, 8) * m**4

    def _derivative(self, _):
        return sp.S.Zero

    def _eval(self, part="full", **hints):
        m = sp.simplify(self.args[0])
        a0 = A0(m)._eval(part)
        if part == "pole":
            return sp.Rational(1, 4) * m**2 * a0
        else:
            return sp.Rational(1, 4) * m**2 * a0 + sp.Rational(1, 8) * m**4


# ---------------------------------------------------------------------------
# A0000 — rank-4 tadpole tensor coefficient
# ---------------------------------------------------------------------------

class A0000(PVFunction):
    """
    int k^mu k^nu k^rho k^sigma / (k²-m²)
        = (g^{μν}g^{ρσ} + g^{μρ}g^{νσ} + g^{μσ}g^{νρ}) · A_{0000}(m)

    Reduction (guide.tex §3.1, d=4):
        A_{0000}(m) = m⁴/24 · A_0(m) + 5m⁶/144
    """
    nargs = 1
    latex_name = r"A_{0000}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        m = self.args[0]
        return sp.Rational(1, 24) * m**4 * A0(m) + sp.Rational(5, 144) * m**6

    def _derivative(self, _):
        return sp.S.Zero

    def _eval(self, part="full", **hints):
        m = sp.simplify(self.args[0])
        a0 = A0(m)._eval(part)
        if part == "pole":
            return sp.Rational(1, 24) * m**4 * a0
        else:
            return sp.Rational(1, 24) * m**4 * a0 + sp.Rational(5, 144) * m**6
