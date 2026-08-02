"""
Two-point PV functions: B0 (and later B1, B11, B00, dB0_dp2, ...).

Numerical evaluation uses direct Feynman-parameter Gauss-Legendre quadrature.
The Feynman-parameter integrand is numerically safe for all kinematic
configurations (p²=0, equal masses, one zero mass) with no case branching.
Above threshold (p² > (m0+m1)²) the branch-point splitting is applied
automatically.

Symbolic evaluation (.doit()) uses the closed-form expressions from
guide.tex §4 — these are kept for the symbolic / algebraic path only.

Naming convention: arguments are (p2, m0, m1) throughout, matching
guide.tex §4.  m0 is the mass on propagator D0, m1 on D1.
"""

import sympy as sp
import numpy as np
from ..base import PVFunction
from ..symbols import epsilon_bar, mu
from .A_functions import A0

# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature nodes/weights on [0, 1] — shared by all B functions
# ---------------------------------------------------------------------------

_GL_N = 100
_gl_t, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_GL_X = 0.5 * (_gl_t + 1.0)      # nodes mapped from [-1,1] to [0,1]
_GL_W = 0.5 * _gl_w               # weights (×1/2 from change of variable)


def _gl_interval(p2, m0, m1, mu_val, a, b, n=0):
    """
    GL quadrature of -∫_a^b x^n·ln(|χ(x)|/μ²) dx via x=a+(b−a)sin²(πt/2).

    n=0 → B0 integrand, n=1 → B1, n=2 → B11.
    The sin²(πt/2) map vanishes at both endpoints, regularising any
    log singularity at the branch points x1, x2 above threshold.
    """
    t = _GL_X
    x = a + (b - a) * np.sin(np.pi * t / 2)**2
    jac = (b - a) * (np.pi / 2) * np.sin(np.pi * t)
    chi = m0**2 * (1 - x) + m1**2 * x - x * (1 - x) * p2
    return float(np.sum(_GL_W * (-x**n * np.log(np.abs(chi) / mu_val**2) * jac)))


def _db0_scalar(p2, m0, m1, is_above):
    """
    dB0/dp² finite part for a single kinematic point (UV-finite).

    Below threshold: GL of ∫₀¹ x(1-x)/χ dx — smooth, no singularities.

    Above threshold: χ has simple poles at x1, x2.  The real part is the
    Cauchy PV of the integral; the imaginary part is exact.

    PV strategy: subtract the residue at each pole before integrating,
    then add back the analytic PV contributions:
        PV∫₀¹ rᵢ/(x−xᵢ) dx = rᵢ·ln((1−xᵢ)/xᵢ)
    The subtracted integrand is smooth on [0,1].
    """
    t = _GL_X
    x = np.sin(np.pi * t / 2)**2
    jac = (np.pi / 2) * np.sin(np.pi * t)
    chi = m0**2 * (1 - x) + m1**2 * x - x * (1 - x) * p2

    if not is_above:
        return float(np.sum(_GL_W * (x * (1 - x) / chi) * jac)) + 0j

    # Branch points: roots of χ(x)=0,  χ = p2·x² − (p2+m0²−m1²)·x + m0²
    K = (p2 - (m0 + m1)**2) * (p2 - (m0 - m1)**2)   # Kallen (>0 above thr)
    sqrt_K = np.sqrt(K)
    b_c = -(p2 + m0**2 - m1**2)
    x1 = np.clip((-b_c - sqrt_K) / (2 * p2), 0.0, 1.0)
    x2 = np.clip((-b_c + sqrt_K) / (2 * p2), 0.0, 1.0)

    # Residues of x(1-x)/χ at x1, x2  (χ'(xᵢ) = p2·(xᵢ − x_other) = ±sqrt_K)
    r1 = x1 * (1 - x1) / (-sqrt_K)      # χ'(x1) = −sqrt_K
    r2 = x2 * (1 - x2) / (+sqrt_K)      # χ'(x2) = +sqrt_K

    # Smooth integrand after pole subtraction (finite everywhere on [0,1])
    smooth = x * (1 - x) / chi - r1 / (x - x1) - r2 / (x - x2)
    re = float(np.sum(_GL_W * smooth * jac))

    # Analytic PV terms:  PV∫₀¹ rᵢ/(x−xᵢ) dx = rᵢ·ln((1−xᵢ)/xᵢ)
    re += r1 * np.log((1 - x1) / x1) + r2 * np.log((1 - x2) / x2)

    # Imaginary part = π · d(x2−x1)/dp²  (exact)
    im = np.pi * ((m0**2 + m1**2) * p2 - (m0**2 - m1**2)**2) / (p2**2 * sqrt_K)

    return re + 1j * im


def _bfn_scalar(p2, m0, m1, mu_val, is_above, n=0):
    """
    Finite part of the Passarino-Veltman integral with integrand weight x^n,
    at a single kinematic point:

        Bn_finite = -∫₀¹ x^n · ln(χ(x)/μ²) dx

    n=0 → B0, n=1 → B1, n=2 → B11.

    Below threshold: one GL pass on [0,1] via x=sin²(πt/2).
    Above threshold: split at χ=0 roots x1,x2; Im part is exact (see note).

    Imaginary part:  Im = +π·∫_{x1}^{x2} x^n dx = +π·(x2^{n+1} - x1^{n+1})/(n+1)
    (positive, from the iε prescription p²→p²+iε, which gives ln(χ+iε)→ln|χ|−iπ
    between x1 and x2 where χ<0.)
    """
    if not is_above:
        return _gl_interval(p2, m0, m1, mu_val, 0.0, 1.0, n) + 0j

    K = (p2 - (m0 + m1)**2) * (p2 - (m0 - m1)**2)
    sqrt_K = np.sqrt(K)
    b_c = -(p2 + m0**2 - m1**2)
    x1 = np.clip((-b_c - sqrt_K) / (2 * p2), 0.0, 1.0)
    x2 = np.clip((-b_c + sqrt_K) / (2 * p2), 0.0, 1.0)

    re = (_gl_interval(p2, m0, m1, mu_val, 0.0, x1, n)
          + _gl_interval(p2, m0, m1, mu_val, x1, x2, n)
          + _gl_interval(p2, m0, m1, mu_val, x2, 1.0, n))
    im = np.pi * (x2**(n + 1) - x1**(n + 1)) / (n + 1)
    return re + 1j * im


def _b0_scalar(p2, m0, m1, mu_val, is_above):
    return _bfn_scalar(p2, m0, m1, mu_val, is_above, n=0)


# ---------------------------------------------------------------------------
# B0 — scalar bubble
# ---------------------------------------------------------------------------

class B0(PVFunction):

    nargs = 3
    latex_name = r"B_0"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def _derivative(self, sym):
        p2, m0, m1 = self.args
        if sym == p2:
            return dB0_dp2(p2, m0, m1)
        return sp.S.Zero

    # --- Symbolic evaluation methods (for .doit() / symbolic path only) ---

    def _eval_p2_zero_m1_zero(self, part="full"):
        p2, m0, _ = map(sp.simplify, self.args)
        if part == "full":
            return 1 + 1/epsilon_bar - sp.log(m0**2 / mu**2)
        elif part == "pole":
            return 1
        elif part == "finite":
            return 1 - sp.log(m0**2 / mu**2)

    def _eval_p2_zero_m0_zero(self, part="full"):
        p2, _, m1 = map(sp.simplify, self.args)
        if part == "full":
            return 1 + 1/epsilon_bar - sp.log(m1**2 / mu**2)
        elif part == "pole":
            return 1
        elif part == "finite":
            return 1 - sp.log(m1**2 / mu**2)

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        p2, _, m1 = map(sp.simplify, self.args)
        if part == "full":
            return 1/epsilon_bar - sp.log(m1**2 / mu**2)
        elif part == "pole":
            return 1
        elif part == "finite":
            return -sp.log(m1**2 / mu**2)

    def _eval_p2_zero(self, part="full"):
        p2, m0, m1 = map(sp.simplify, self.args)
        return (A0(m0).doit(part) - A0(m1).doit(part)) / (m0**2 - m1**2)

    def _eval_m0_eq_m1(self, part="full"):
        p2, m0, m1 = map(sp.simplify, self.args)
        Lambda = p2 * sp.sqrt(1 - 4 * m0**2 / p2)
        R = -(Lambda / p2) * sp.log((2 * m0**2 - p2 + Lambda) / (2 * m0**2))
        if part == "full":
            return 1/epsilon_bar - sp.log(m1**2 / mu**2) + 2 - R
        elif part == "pole":
            return 1
        elif part == "finite":
            return 2 - R - sp.log(m1**2 / mu**2)

    def _eval_general(self, part="full"):
        p2, m0, m1 = map(sp.simplify, self.args)
        Lambda = sp.sqrt(m0**4 + m1**4 + p2**2
                         + 2 * (-m0**2 * p2 - m1**2 * p2 - m0**2 * m1**2))
        R = -(Lambda / p2) * sp.log((m0**2 + m1**2 - p2 + Lambda) / (2 * m0 * m1))
        # Symmetric form (guide.tex §4 eq.318):
        # B0 = 1/eps - ln(m0*m1/μ²) - R + (m1²-m0²)/(2p²)*ln(m0²/m1²) + 2
        if part == "full":
            return (1/epsilon_bar - sp.log(m0 * m1 / mu**2) + 2
                    + ((m1**2 - m0**2) / (2 * p2)) * sp.log(m0**2 / m1**2)
                    - R)
        elif part == "pole":
            return 1
        elif part == "finite":
            return (2 - sp.log(m0 * m1 / mu**2)
                    + ((m1**2 - m0**2) / (2 * p2)) * sp.log(m0**2 / m1**2)
                    - R)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        return sp.Piecewise(
            (self._eval_p2_zero_m0_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m0, 0))),
            (self._eval_p2_zero_m1_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m1, 0))),
            (self._eval_p2_zero_m0_eq_m1(part), sp.And(sp.Eq(p2, 0), sp.Eq(m0, m1))),
            (self._eval_p2_zero(part),           sp.Eq(p2, 0)),
            (self._eval_m0_eq_m1(part),          sp.Eq(m0, m1)),
            (self._eval_general(part),           True),
        )

    # --- Numerical evaluation via Feynman-parameter quadrature ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate B0 numerically.

        Parameters
        ----------
        p2v, m0v, m1v : array-like
            Kinematic inputs (same shape).
        mu_val : float
            Renormalisation scale μ.
        part : {"pole", "finite"}
            "pole"   → coefficient of 1/ε̄, always 1 for B0.
            "finite" → UV-finite part: B0_finite = -∫₀¹ ln(χ(x)/μ²) dx.
                       Returns complex array; Im is the absorptive part.

        Notes
        -----
        Below threshold (p² < (m0+m1)²): χ > 0 everywhere, single GL pass
        over [0,1] with x=sin²(πt/2) substitution (handles massless limits).

        Above threshold: χ has two zeros x1 < x2 in (0,1). The integration
        is split into [0,x1] + [x1,x2] + [x2,1]; each piece uses the same
        substitution mapped to the subinterval to regularize the log
        singularities at x1, x2. The imaginary part (absorptive piece)
        is π(x2−x1) and is added analytically.
        """
        p2v = np.atleast_1d(np.asarray(p2v, dtype=float))
        m0v = np.atleast_1d(np.asarray(m0v, dtype=float))
        m1v = np.atleast_1d(np.asarray(m1v, dtype=float))

        if part == "pole":
            return np.ones_like(p2v)

        n = len(p2v)
        result = np.empty(n, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            # Vectorised path: all below threshold
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(_GL_W[:, np.newaxis] * (-np.log(chi / mu_val**2) * jac), axis=0)
            return result.real + 0j  # imaginary part identically zero below threshold

        for i in range(n):
            result[i] = _b0_scalar(p2v[i], m0v[i], m1v[i], mu_val, above[i])

        return result


# ---------------------------------------------------------------------------
# dB0_dp2 — ∂B0/∂p²  (UV-finite, no pole)
# ---------------------------------------------------------------------------

class dB0_dp2(PVFunction):
    """
    Derivative of B0 with respect to p².

    ∂B0/∂p² = ∫₀¹ x(1−x)/χ(x) dx   (UV-finite, no 1/ε̄ pole)

    where χ(x) = m0²(1−x) + m1²x − x(1−x)p².

    Below threshold: χ > 0, smooth GL on [0,1].
    Above threshold: χ has simple poles at x1, x2.  Real part is computed
    via Cauchy-PV pole subtraction; imaginary part is exact.
    """

    nargs = 3
    latex_name = r"\partial B_0/\partial p^2"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def _latex(self, printer):
        args = ", ".join(printer.doprint(a) for a in self.args)
        return rf"\frac{{\partial B_{{0}}}}{{\partial p^2}}\left({args}\right)"

    def _derivative(self, _):
        raise NotImplementedError

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        _, _, m1 = map(sp.simplify, self.args)
        val = sp.Rational(1, 2) / m1**2
        return val if part != "pole" else sp.S.Zero

    def _eval_p2_zero_m1_zero(self, part="full"):
        _, m0, _ = map(sp.simplify, self.args)
        val = sp.Rational(1, 2) / m0**2
        return val if part != "pole" else sp.S.Zero

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        _, m0, _ = map(sp.simplify, self.args)
        val = sp.Rational(1, 6) / m0**2
        return val if part != "pole" else sp.S.Zero

    def _eval_p2_zero(self, part="full"):
        # p²=0, m0≠m1, both nonzero   (guide.tex §4, eq ~456)
        _, m0, m1 = map(sp.simplify, self.args)
        val = sp.Rational(1, 2) * (
            m0**4 - m1**4 - 2 * m0**2 * m1**2 * sp.log(m0**2 / m1**2)
        ) / (m0**2 - m1**2)**3
        return val if part != "pole" else sp.S.Zero

    def _eval_m0_eq_m1(self, part="full"):
        # p²≠0, m0=m1=m
        p2, m0, _ = map(sp.simplify, self.args)
        Lam = p2 * sp.sqrt(1 - 4 * m0**2 / p2)
        L = sp.log((2 * m0**2 - p2 + Lam) / (2 * m0**2))
        val = (2 * m0**2 / (p2 * Lam)) * L - 1 / p2
        return val if part != "pole" else sp.S.Zero

    def _eval_general(self, part="full"):
        # p²≠0, m0≠m1   (derived by ∂/∂p² of guide.tex eq 318)
        p2, m0, m1 = map(sp.simplify, self.args)
        Lam = sp.sqrt(m0**4 + m1**4 + p2**2
                      - 2 * (m0**2 * p2 + m1**2 * p2 + m0**2 * m1**2))
        L = sp.log((m0**2 + m1**2 - p2 + Lam) / (2 * m0 * m1))
        val = (-(m1**2 - m0**2) / (2 * p2**2) * sp.log(m0**2 / m1**2)
               + ((m0**2 + m1**2) * p2 - (m0**2 - m1**2)**2) / (p2**2 * Lam) * L
               - 1 / p2)
        return val if part != "pole" else sp.S.Zero

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        return sp.Piecewise(
            (self._eval_p2_zero_m0_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m0, 0))),
            (self._eval_p2_zero_m1_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m1, 0))),
            (self._eval_p2_zero_m0_eq_m1(part), sp.And(sp.Eq(p2, 0), sp.Eq(m0, m1))),
            (self._eval_p2_zero(part),           sp.Eq(p2, 0)),
            (self._eval_m0_eq_m1(part),          sp.Eq(m0, m1)),
            (self._eval_general(part),           True),
        )

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate ∂B0/∂p² numerically.

        Parameters
        ----------
        p2v, m0v, m1v : array-like
            Kinematic inputs (same shape).
        mu_val : float
            Renormalisation scale μ (unused — ∂B0/∂p² is μ-independent).
        part : {"pole", "finite"}
            "pole"   → 0 (always — ∂B0/∂p² is UV-finite).
            "finite" → the value itself; complex above threshold.

        Notes
        -----
        Below threshold: GL of ∫₀¹ x(1−x)/χ dx via x=sin²(πt/2) sub.
        Above threshold: Cauchy-PV subtraction for Re; exact formula for Im.
        The sin²(πt/2) node distribution is retained for below-threshold to
        handle massless limits gracefully (χ→0 at x=0 or x=1 when m=0).
        """
        p2v = np.atleast_1d(np.asarray(p2v, dtype=float))
        m0v = np.atleast_1d(np.asarray(m0v, dtype=float))
        m1v = np.atleast_1d(np.asarray(m1v, dtype=float))

        if part == "pole":
            return np.zeros_like(p2v)

        n = len(p2v)
        result = np.empty(n, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(_GL_W[:, np.newaxis] * (x * (1 - x) / chi) * jac, axis=0)
            return result.real + 0j

        for i in range(n):
            result[i] = _db0_scalar(p2v[i], m0v[i], m1v[i], above[i])

        return result


# ---------------------------------------------------------------------------
# B1 — first tensor coefficient (not symmetric under m0 ↔ m1)
# ---------------------------------------------------------------------------

class B1(PVFunction):
    """
    B1(p², m0, m1) = -∫₀¹ x · ln(χ/μ²) dx  +  (1/2)·(1/ε̄)

    where χ(x) = m0²(1−x) + m1²x − x(1−x)p².

    NOT symmetric under m0 ↔ m1.  The swap identity (guide.tex eq. 339) is
        B1(p², m1, m0) = B0(p², m0, m1) − B1(p², m0, m1).

    For p²≠0 the reduction formula (guide.tex eq. 333) is used for doit():
        B1 = [A0(m1) − A0(m0) + (p²+m0²−m1²)·B0] / (2p²).

    For p²=0, the reduction formula has a 0/0 form; direct integration gives
    the p²=0 special cases below.

    NOTE: guide.tex eq. 397-398 for B1(0,m0,m1) is incorrect.  The correct
    formula (derived by direct Feynman-parameter integration) is:
        B1(0,m0,m1) = ½/ε̄ − ½ ln(m1²/μ²) + ¼
                      + m0²/[2(m0²−m1²)] − m0⁴·ln(m0²/m1²)/[2(m0²−m1²)²]
    """

    nargs = 3
    latex_name = r"B_1"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def _derivative(self, sym):
        p2, m0, m1 = self.args
        if sym == p2:
            raise NotImplementedError("dB1_dp2 not yet implemented")
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # B1(0, 0, m1) = Δ(m1)/2 + 1/4
        _, _, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2)
        finite = sp.Rational(1, 4) - sp.Rational(1, 2) * sp.log(m1**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero_m1_zero(self, part="full"):
        # B1(0, m0, 0) = Δ(m0)/2 + 3/4  (via swap: B0(0,0,m0) − B1(0,0,m0))
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2)
        finite = sp.Rational(3, 4) - sp.Rational(1, 2) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # B1(0, m, m) = Δ(m)/2
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2)
        finite = -sp.Rational(1, 2) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero(self, part="full"):
        # B1(0, m0, m1)  with m0 ≠ m1, both nonzero.
        # Reference mass in the log is m1 (not m0 — see class docstring).
        _, m0, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2)
        finite = (- sp.Rational(1, 2) * sp.log(m1**2 / mu**2)
                  + sp.Rational(1, 4)
                  + m0**2 / (2 * (m0**2 - m1**2))
                  - m0**4 * sp.log(m0**2 / m1**2) / (2 * (m0**2 - m1**2)**2))
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_general(self, part="full"):
        # p² ≠ 0 — reduction formula (guide.tex eq. 333)
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return sp.Rational(1, 2)
        return (A0(m1).doit(part) - A0(m0).doit(part)
                + f * B0(p2, m0, m1).doit(part)) / (2 * p2)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        return sp.Piecewise(
            (self._eval_p2_zero_m0_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m0, 0))),
            (self._eval_p2_zero_m1_zero(part),  sp.And(sp.Eq(p2, 0), sp.Eq(m1, 0))),
            (self._eval_p2_zero_m0_eq_m1(part), sp.And(sp.Eq(p2, 0), sp.Eq(m0, m1))),
            (self._eval_p2_zero(part),           sp.Eq(p2, 0)),
            (self._eval_general(part),           True),
        )

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate B1 numerically.

        pole   → 1/2 (constant).
        finite → B1_finite = -∫₀¹ x·ln(χ/μ²) dx, complex above threshold.

        Below threshold: single GL pass via x=sin²(πt/2).
        Above threshold: split GL at branch points x1,x2; imaginary part
        Im = +π(x2²−x1²)/2 added analytically.
        """
        p2v = np.atleast_1d(np.asarray(p2v, dtype=float))
        m0v = np.atleast_1d(np.asarray(m0v, dtype=float))
        m1v = np.atleast_1d(np.asarray(m1v, dtype=float))

        if part == "pole":
            return np.full_like(p2v, 0.5)

        n = len(p2v)
        result = np.empty(n, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(_GL_W[:, np.newaxis] * (-x * np.log(chi / mu_val**2) * jac),
                            axis=0)
            return result.real + 0j

        for i in range(n):
            result[i] = _bfn_scalar(p2v[i], m0v[i], m1v[i], mu_val, above[i], n=1)

        return result
