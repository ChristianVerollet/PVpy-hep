"""
Two-point PV functions: B0 B1, B11, B00, dB0_dp2, ...

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
    if a >= b:
        return 0.0
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


def _b00_gl_interval(p2, m0, m1, mu_val, a, b):
    """
    Compute (1/2)·∫_a^b χ(x)·(1−ln(|χ(x)|/μ²)) dx  via GL on [a,b]
    using x = a + (b−a)·sin²(πt/2) to regularise log singularities at endpoints.
    """
    a, b = float(a), float(b)
    if a >= b:
        return 0.0
    t = _GL_X                                              # nodes on [0,1]
    x = a + (b - a) * np.sin(np.pi * t / 2)**2
    jac = (b - a) * (np.pi / 2) * np.sin(np.pi * t)
    chi = m0**2 * (1 - x) + m1**2 * x - x * (1 - x) * p2
    return float(np.sum(_GL_W * (0.5 * chi * (1 - np.log(np.abs(chi) / mu_val**2)) * jac)))


def _b00_scalar(p2, m0, m1, mu_val, is_above):
    """
    Finite part of B00 at a single kinematic point:
        B00_finite = (1/2)·∫₀¹ χ(x)·(1−ln(χ(x)/μ²)) dx

    Below threshold: one GL pass on [0,1].
    Above threshold: split at χ=0 roots x1,x2; imaginary part is
        Im = (π/2)·∫_{x1}^{x2} χ dx  (negative, since χ<0 there).
    """
    if not is_above:
        return _b00_gl_interval(p2, m0, m1, mu_val, 0.0, 1.0) + 0j

    K = (p2 - (m0 + m1)**2) * (p2 - (m0 - m1)**2)
    sqrt_K = np.sqrt(K)
    b_c = -(p2 + m0**2 - m1**2)
    x1 = np.clip((-b_c - sqrt_K) / (2 * p2), 0.0, 1.0)
    x2 = np.clip((-b_c + sqrt_K) / (2 * p2), 0.0, 1.0)

    re = (_b00_gl_interval(p2, m0, m1, mu_val, 0.0, x1)
          + _b00_gl_interval(p2, m0, m1, mu_val, x1, x2)
          + _b00_gl_interval(p2, m0, m1, mu_val, x2, 1.0))

    # Analytic imaginary part: Im = (π/2)·∫_{x1}^{x2} χ dx
    # χ = p2·x² − f·x + m0²,  antiderivative = p2·x³/3 − f·x²/2 + m0²·x
    f = p2 + m0**2 - m1**2
    def _antideriv(x):
        return p2 * x**3 / 3 - f * x**2 / 2 + m0**2 * x

    im = (np.pi / 2) * (_antideriv(x2) - _antideriv(x1))
    return re + 1j * im


def _dbn_scalar(p2, m0, m1, n, is_above):
    """
    d/dp²[Bn_finite] at a single kinematic point, n=1 (B1) or n=2 (B11).

    Integrand: ∫₀¹ x^(n+1)(1−x)/χ dx  (UV-finite for all n).

    Below threshold: GL via x=sin²(πt/2).
    Above threshold: Cauchy-PV pole subtraction + analytic Im.

    Imaginary part = π·d/dp²[(x2^{n+1}−x1^{n+1})/(n+1)]
                   = π·(x2^n·dx2/dp² − x1^n·dx1/dp²)
    where dx_{1,2}/dp² = (1∓A)/(2p²) − x_{1,2}/p²  with A=(p²−m0²−m1²)/√K.
    """
    t = _GL_X
    x = np.sin(np.pi * t / 2)**2
    jac = (np.pi / 2) * np.sin(np.pi * t)
    chi = m0**2 * (1 - x) + m1**2 * x - x * (1 - x) * p2

    if not is_above:
        return float(np.sum(_GL_W * (x**(n + 1) * (1 - x) / chi) * jac)) + 0j

    K = (p2 - (m0 + m1)**2) * (p2 - (m0 - m1)**2)
    sqrt_K = np.sqrt(K)
    b_c = -(p2 + m0**2 - m1**2)
    x1 = np.clip((-b_c - sqrt_K) / (2 * p2), 0.0, 1.0)
    x2 = np.clip((-b_c + sqrt_K) / (2 * p2), 0.0, 1.0)

    r1 = x1**(n + 1) * (1 - x1) / (-sqrt_K)
    r2 = x2**(n + 1) * (1 - x2) / (+sqrt_K)

    smooth = x**(n + 1) * (1 - x) / chi - r1 / (x - x1) - r2 / (x - x2)
    re = float(np.sum(_GL_W * smooth * jac))
    re += r1 * np.log((1 - x1) / x1) + r2 * np.log((1 - x2) / x2)

    A = (p2 - m0**2 - m1**2) / sqrt_K
    dx1 = (1 - A) / (2 * p2) - x1 / p2
    dx2 = (1 + A) / (2 * p2) - x2 / p2
    im = np.pi * (x2**n * dx2 - x1**n * dx1)

    return re + 1j * im


def _db00_gl_interval(p2, m0, m1, mu_val, a, b):
    """
    GL quadrature of (1/2)·∫_a^b x(1−x)·ln(|χ|/μ²) dx via x=a+(b−a)sin²(πt/2).

    This is the finite integrand for d(B00_finite)/dp².
    """
    a, b = float(a), float(b)
    if a >= b:
        return 0.0
    t = _GL_X
    x = a + (b - a) * np.sin(np.pi * t / 2)**2
    jac = (b - a) * (np.pi / 2) * np.sin(np.pi * t)
    chi = m0**2 * (1 - x) + m1**2 * x - x * (1 - x) * p2
    return float(np.sum(_GL_W * (0.5 * x * (1 - x) * np.log(np.abs(chi) / mu_val**2) * jac)))


def _db00_scalar(p2, m0, m1, mu_val, is_above):
    """
    Finite part of d(B00)/dp² at a single kinematic point.

        d(B00_finite)/dp² = (1/2)·∫₀¹ x(1−x)·ln(χ/μ²) dx

    (The pole −1/(12ε̄) is handled separately by the _numeric_kernel.)

    Below threshold: one GL pass on [0,1].
    Above threshold: split at χ=0 roots x1,x2; imaginary part is
        Im = −(π/2)·∫_{x1}^{x2} x(1−x) dx  (negative, since Im[ln χ]=−π for χ<0).
    """
    if not is_above:
        return _db00_gl_interval(p2, m0, m1, mu_val, 0.0, 1.0) + 0j

    K = (p2 - (m0 + m1)**2) * (p2 - (m0 - m1)**2)
    sqrt_K = np.sqrt(K)
    b_c = -(p2 + m0**2 - m1**2)
    x1 = np.clip((-b_c - sqrt_K) / (2 * p2), 0.0, 1.0)
    x2 = np.clip((-b_c + sqrt_K) / (2 * p2), 0.0, 1.0)

    re = (_db00_gl_interval(p2, m0, m1, mu_val, 0.0, x1)
          + _db00_gl_interval(p2, m0, m1, mu_val, x1, x2)
          + _db00_gl_interval(p2, m0, m1, mu_val, x2, 1.0))

    # Im[ln(χ/μ²)] = −π for χ<0 (Feynman p²+iε prescription).
    # Im = (1/2)·∫_{x1}^{x2} x(1−x)·(−π) dx = −(π/2)·[(x2²−x1²)/2 − (x2³−x1³)/3]
    im = -(np.pi / 2) * ((x2**2 - x1**2) / 2 - (x2**3 - x1**3) / 3)
    return re + 1j * im

#############################################################################
############################### B-functions #################################
#############################################################################

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

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero and not (m0 - m1).is_zero:
            return (A0(m0) - A0(m1)) / (m0**2 - m1**2)
        return None

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
            return 1 / epsilon_bar
        elif part == "finite":
            return 1 - sp.log(m0**2 / mu**2)

    def _eval_p2_zero_m0_zero(self, part="full"):
        p2, _, m1 = map(sp.simplify, self.args)
        if part == "full":
            return 1 + 1/epsilon_bar - sp.log(m1**2 / mu**2)
        elif part == "pole":
            return 1 / epsilon_bar
        elif part == "finite":
            return 1 - sp.log(m1**2 / mu**2)

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        p2, _, m1 = map(sp.simplify, self.args)
        if part == "full":
            return 1/epsilon_bar - sp.log(m1**2 / mu**2)
        elif part == "pole":
            return 1 / epsilon_bar
        elif part == "finite":
            return -sp.log(m1**2 / mu**2)

    def _eval_p2_zero(self, part="full"):
        p2, m0, m1 = map(sp.simplify, self.args)
        return (A0(m0).doit(part=part) - A0(m1).doit(part=part)) / (m0**2 - m1**2)

    def _eval_both_massless(self, part="full"):
        # B0(p2, 0, 0): B0_finite = 2 - ln(-p2/mu^2), valid for p2 != 0
        p2, m0, m1 = map(sp.simplify, self.args)
        if part == "full":
            return 1/epsilon_bar + 2 - sp.log(-p2 / mu**2)
        elif part == "pole":
            return 1 / epsilon_bar
        elif part == "finite":
            return 2 - sp.log(-p2 / mu**2)

    def _eval_one_massless(self, part="full"):
        # B0(p2, 0, m) or B0(p2, m, 0): B0_finite = 2 - ln(m^2/mu^2) + (m^2-p2)/p2 * ln(1-p2/m^2)
        # B0 is symmetric in masses, so both cases use the same formula with m = nonzero mass.
        p2, m0, m1 = map(sp.simplify, self.args)
        m = m1 if m0.is_zero else m0
        fin = 2 - sp.log(m**2 / mu**2) + ((m**2 - p2) / p2) * sp.log(1 - p2 / m**2)
        if part == "full":
            return 1/epsilon_bar + fin
        elif part == "pole":
            return 1 / epsilon_bar
        elif part == "finite":
            return fin

    def _eval_m0_eq_m1(self, part="full"):
        p2, m0, m1 = map(sp.simplify, self.args)
        Lambda = p2 * sp.sqrt(1 - 4 * m0**2 / p2)
        R = -(Lambda / p2) * sp.log((2 * m0**2 - p2 + Lambda) / (2 * m0**2))
        if part == "full":
            return 1/epsilon_bar - sp.log(m1**2 / mu**2) + 2 - R
        elif part == "pole":
            return 1 / epsilon_bar
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
            return 1 / epsilon_bar
        elif part == "finite":
            return (2 - sp.log(m0 * m1 / mu**2)
                    + ((m1**2 - m0**2) / (2 * p2)) * sp.log(m0**2 / m1**2)
                    - R)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        if (m0 - m1).is_zero:
            if m0.is_zero:
                return self._eval_both_massless(part)
            return self._eval_m0_eq_m1(part)
        if m0.is_zero or m1.is_zero:
            return self._eval_one_massless(part)
        return self._eval_general(part)

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
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

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

    """

    nargs = 3
    latex_name = r"B_1"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        f = p2 + m0**2 - m1**2
        return (A0(m1) - A0(m0) + f * B0(p2, m0, m1)) / (2 * p2)

    def _derivative(self, sym):
        p2, m0, m1 = self.args
        if sym == p2:
            return dB1_dp2(p2, m0, m1)
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # B1(0, 0, m1) = Δ(m1)/2 + 1/4
        _, _, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2) / epsilon_bar
        finite = sp.Rational(1, 4) - sp.Rational(1, 2) * sp.log(m1**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero_m1_zero(self, part="full"):
        # B1(0, m0, 0) = Δ(m0)/2 + 3/4  (via swap: B0(0,0,m0) − B1(0,0,m0))
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2) / epsilon_bar
        finite = sp.Rational(3, 4) - sp.Rational(1, 2) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # B1(0, m, m) = Δ(m)/2
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2) / epsilon_bar
        finite = -sp.Rational(1, 2) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 2) / epsilon_bar + finite

    def _eval_p2_zero(self, part="full"):
        # B1(0, m0, m1)  with m0 ≠ m1, both nonzero.
        # Reference mass in the log is m1 (not m0 — see class docstring).
        _, m0, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 2) / epsilon_bar
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
            return sp.Rational(1, 2) / epsilon_bar
        return (A0(m1).doit(part=part) - A0(m0).doit(part=part)
                + f * B0(p2, m0, m1).doit(part=part)) / (2 * p2)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

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
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

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


# ---------------------------------------------------------------------------
# B11 — second tensor coefficient (not symmetric under m0 ↔ m1)
# ---------------------------------------------------------------------------

class B11(PVFunction):
    """
    B11(p², m0, m1) = -∫₀¹ x² · ln(χ/μ²) dx  +  (1/3)·(1/ε̄)

    where χ(x) = m0²(1−x) + m1²x − x(1−x)p².

    NOT symmetric under m0 ↔ m1.  The swap identity (guide.tex eq. 362) is
        B11(p², m1, m0) = B11(p², m0, m1) − 2B1(p², m0, m1) + B0(p², m0, m1).

    For p²≠0 the corrected reduction formula is used for doit():
        B11 = (1/(3p²))·[p²/6 − (m0²+m1²)/2 + A0(m1) − m0²·B0 + 2f·B1]
    where f = p²+m0²−m1².

    NOTE: guide.tex lines 352-353 have wrong signs on all variable terms,
    tracing to a sign error in the d·B00 identity at line 358 (guide writes
    +mass term; the correct identity has −(m0²+m1²)/2).

    NOTE: guide.tex lines 402-403 for B11(0,m0,m1) have a typo: extra m1²
    in the denominator of the rational term.  The correct formula is below.
    """

    nargs = 3
    latex_name = r"B_{11}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        f = p2 + m0**2 - m1**2
        return (p2 / 6 - (m0**2 + m1**2) / 2
                + A0(m1) - m0**2 * B0(p2, m0, m1)
                + 2 * f * B1(p2, m0, m1)) / (3 * p2)

    def _derivative(self, sym):
        p2, m0, m1 = self.args
        if sym == p2:
            return dB11_dp2(p2, m0, m1)
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # B11(0, 0, m1) = Δ(m1)/3 + 1/9
        _, _, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 3) / epsilon_bar
        finite = sp.Rational(1, 9) - sp.Rational(1, 3) * sp.log(m1**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 3) / epsilon_bar + finite

    def _eval_p2_zero_m1_zero(self, part="full"):
        # B11(0, m0, 0) = Δ(m0)/3 + 11/18  (via swap identity)
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 3) / epsilon_bar
        finite = sp.Rational(11, 18) - sp.Rational(1, 3) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 3) / epsilon_bar + finite

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # B11(0, m, m) = Δ(m)/3
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 3) / epsilon_bar
        finite = -sp.Rational(1, 3) * sp.log(m0**2 / mu**2)
        if part == "finite":
            return finite
        return sp.Rational(1, 3) / epsilon_bar + finite

    def _eval_p2_zero(self, part="full"):
        # B11(0, m0, m1) with m0≠m1, both nonzero.
        # Derived by direct Feynman-parameter integration; guide.tex line 402-403 has typo.
        _, m0, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.Rational(1, 3) / epsilon_bar
        D = m0**2 - m1**2
        finite = (- sp.Rational(1, 3) * sp.log(m1**2 / mu**2)
                  + sp.Rational(1, 9)
                  + m0**2 / (6 * D)
                  + m0**4 / (3 * D**2)
                  - m0**6 * sp.log(m0**2 / m1**2) / (3 * D**3))
        if part == "finite":
            return finite
        return sp.Rational(1, 3) / epsilon_bar + finite

    def _eval_general(self, part="full"):
        # B11 = (1/(3p²))·[p²/6 − (m0²+m1²)/2 + A0(m1) − m0²·B0 + 2f·B1]
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return sp.Rational(1, 3) / epsilon_bar
        return (p2 / 6 - (m0**2 + m1**2) / 2
                + A0(m1).doit(part=part)
                - m0**2 * B0(p2, m0, m1).doit(part=part)
                + 2 * f * B1(p2, m0, m1).doit(part=part)) / (3 * p2)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate B11 numerically.

        pole   → 1/3 (constant).
        finite → B11_finite = -∫₀¹ x²·ln(χ/μ²) dx, complex above threshold.

        Below threshold: single GL pass via x=sin²(πt/2).
        Above threshold: split GL at branch points x1, x2; imaginary part
        Im = +π(x2³−x1³)/3 added analytically.
        """
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

        if part == "pole":
            return np.full_like(p2v, 1.0 / 3.0)

        npts = len(p2v)
        result = np.empty(npts, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(
                _GL_W[:, np.newaxis] * (-x**2 * np.log(chi / mu_val**2) * jac),
                axis=0)
            return result.real + 0j

        for i in range(npts):
            result[i] = _bfn_scalar(p2v[i], m0v[i], m1v[i], mu_val, above[i], n=2)

        return result


# ---------------------------------------------------------------------------
# B00 — metric tensor bubble coefficient (SYMMETRIC under m0↔m1)
# ---------------------------------------------------------------------------


class B00(PVFunction):
    """
    B00(p², m0, m1) = (1/2)(1/ε̄+1)·∫₀¹χ dx − (1/2)·∫₀¹χ·ln(χ/μ²) dx

    where χ(x) = m0²(1−x) + m1²x − x(1−x)p².

    SYMMETRIC under m0↔m1 (the integrand is invariant under x↔1−x,m0↔m1).

    Pole: (m0²+m1²)/4 − p²/12  (kinematic-dependent, unlike B0/B1/B11).

    For p²≠0 the reduction formula (guide.tex corrected, verified ≲10⁻¹⁴):
        B00 = (1/6)·[−p²/3 + m0² + m1² + A0(m1) + 2m0²·B0 − f·B1]
    where f = p²+m0²−m1².

    """

    nargs = 3
    latex_name = r"B_{00}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        f = p2 + m0**2 - m1**2
        return (-p2 / 3 + m0**2 + m1**2
                + A0(m1) + 2 * m0**2 * B0(p2, m0, m1)
                - f * B1(p2, m0, m1)) / 6

    def _derivative(self, sym):
        p2, m0, m1 = self.args
        if sym == p2:
            return dB00_dp2(p2, m0, m1)
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # B00(0, 0, m1) = (m1²/4)(3/2 + Δ(m1))
        _, _, m1 = map(sp.simplify, self.args)
        pole = m1**2 / 4
        finite = sp.Rational(3, 8) * m1**2 - sp.Rational(1, 4) * m1**2 * sp.log(m1**2 / mu**2)
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero_m1_zero(self, part="full"):
        # B00(0, m0, 0) = (m0²/4)(3/2 + Δ(m0))  [by m0↔m1 symmetry]
        _, m0, _ = map(sp.simplify, self.args)
        pole = m0**2 / 4
        finite = sp.Rational(3, 8) * m0**2 - sp.Rational(1, 4) * m0**2 * sp.log(m0**2 / mu**2)
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # B00(0, m, m) = (m²/2)(1 + Δ(m))  [guide.tex line 441]
        _, m0, _ = map(sp.simplify, self.args)
        pole = m0**2 / 2
        finite = m0**2 * sp.Rational(1, 2) * (1 - sp.log(m0**2 / mu**2))
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero(self, part="full"):
        # B00(0, m0, m1) general, m0≠m1, both nonzero.
        # Symmetric form (guide.tex line 482):
        # (1/4)(m0²+m1²)(3/2 + 1/ε̄ − ln(m0m1/μ²)) − (m0⁴+m1⁴)/(8(m0²−m1²))·ln(m0²/m1²)
        _, m0, m1 = map(sp.simplify, self.args)
        pole = (m0**2 + m1**2) / 4
        finite = (sp.Rational(3, 8) * (m0**2 + m1**2)
                  - sp.Rational(1, 4) * (m0**2 + m1**2) * sp.log(m0 * m1 / mu**2)
                  - (m0**4 + m1**4) / (8 * (m0**2 - m1**2)) * sp.log(m0**2 / m1**2))
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_general(self, part="full"):
        # B00 = (1/6)[−p²/3 + m0²+m1² + A0(m1) + 2m0²·B0 − f·B1]
        # where f = p²+m0²−m1².  guide.tex lines 417-418, verified ≲10⁻¹⁴.
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return ((m0**2 + m1**2) / 4 - p2 / 12) / epsilon_bar
        return (-p2 / 3 + m0**2 + m1**2
                + A0(m1).doit(part=part)
                + 2 * m0**2 * B0(p2, m0, m1).doit(part=part)
                - f * B1(p2, m0, m1).doit(part=part)) / 6

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate B00 numerically.

        pole   → (m0²+m1²)/4 − p²/12  (kinematic-dependent).
        finite → B00_finite = (1/2)·∫₀¹χ·(1−ln(χ/μ²)) dx, complex above threshold.

        Below threshold: single vectorised GL pass via x=sin²(πt/2).
        Above threshold: split at χ=0 roots x1,x2;
            Im = (π/2)·∫_{x1}^{x2}χ dx  (exact, negative since χ<0 there).
        """
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

        if part == "pole":
            return (m0v**2 + m1v**2) / 4 - p2v / 12

        npts = len(p2v)
        result = np.empty(npts, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(
                _GL_W[:, np.newaxis] * (0.5 * chi * (1 - np.log(chi / mu_val**2)) * jac),
                axis=0)
            return result.real + 0j

        for i in range(npts):
            result[i] = _b00_scalar(p2v[i], m0v[i], m1v[i], mu_val, above[i])

        return result

#############################################################################
############################### Derivatives #################################
#############################################################################

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
    latex_name = r"\frac{\partial B_{0}}{\partial p^2}"

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
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        if (m0 - m1).is_zero:
            return self._eval_m0_eq_m1(part)
        return self._eval_general(part)

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
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

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
# dB1/dp² — first derivative of B1 w.r.t. p²  (UV-finite)
# ---------------------------------------------------------------------------

class dB1_dp2(PVFunction):
    """
    dB1/dp²(p², m0, m1) = ∫₀¹ x²(1−x)/χ dx  (UV-finite, pole = 0)

    Reduction formula at p²≠0 (from differentiating B1 = [A0(m1)−A0(m0)+fB0]/(2p²)):
        dB1/dp² = [A0(m0)−A0(m1) + (m1²−m0²)·B0 + f·p²·dB0/dp²] / (2p²²)
    where f = p²+m0²−m1².

    NOT symmetric under m0↔m1. Swap identity:
        dB1/dp²(p², m1, m0) = dB0/dp²(p², m0, m1) − dB1/dp²(p², m0, m1).
    """

    nargs = 3
    latex_name = r"\frac{\partial B_{1}}{\partial p^2}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        f = p2 + m0**2 - m1**2
        return (A0(m0) - A0(m1)
                + (m1**2 - m0**2) * B0(p2, m0, m1)
                + f * p2 * dB0_dp2(p2, m0, m1)) / (2 * p2**2)

    def _derivative(self, sym):
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # dB1/dp²(0, 0, m1) = 1/(6m1²)  [from ∫₀¹ x(1−x)/(m1²x) dx]
        _, _, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 6) / m1**2

    def _eval_p2_zero_m1_zero(self, part="full"):
        # dB1/dp²(0, m0, 0) = 1/(3m0²)  [via swap identity]
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 3) / m0**2

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # dB1/dp²(0, m, m) = 1/(12m²)  [from ∫₀¹ x²(1−x)/m² dx]
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 12) / m0**2

    def _eval_p2_zero(self, part="full"):
        # dB1/dp²(0, m0, m1) with m0≠m1, both nonzero.
        # Derived by integrating x²(1−x)/(m0²(1−x)+m1²x) analytically.
        _, m0, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        D = m1**2 - m0**2
        return ((m1**4 - 5 * m0**2 * m1**2 - 2 * m0**4) / (6 * D**3)
                + m0**4 * m1**2 * sp.log(m1**2 / m0**2) / D**4)

    def _eval_general(self, part="full"):
        # p²≠0 reduction formula
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return sp.S.Zero
        return (A0(m0).doit(part=part) - A0(m1).doit(part=part)
                + (m1**2 - m0**2) * B0(p2, m0, m1).doit(part=part)
                + f * p2 * dB0_dp2(p2, m0, m1).doit(part=part)) / (2 * p2**2)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate dB1/dp² numerically (UV-finite, μ-independent).

        pole → 0.
        finite → ∫₀¹ x²(1−x)/χ dx, complex above threshold.

        Below threshold: GL via x=sin²(πt/2).
        Above threshold: Cauchy-PV subtraction; analytic Im.
        """
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

        if part == "pole":
            return np.zeros_like(p2v)

        n_pts = len(p2v)
        result = np.empty(n_pts, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(_GL_W[:, np.newaxis] * (x**2 * (1 - x) / chi) * jac, axis=0)
            return result.real + 0j

        for i in range(n_pts):
            result[i] = _dbn_scalar(p2v[i], m0v[i], m1v[i], 1, above[i])

        return result


# ---------------------------------------------------------------------------
# dB11/dp² — first derivative of B11 w.r.t. p²  (UV-finite)
# ---------------------------------------------------------------------------

class dB11_dp2(PVFunction):
    """
    dB11/dp²(p², m0, m1) = ∫₀¹ x³(1−x)/χ dx  (UV-finite, pole = 0)

    Reduction formula at p²≠0 (from differentiating B11):
        dB11/dp² = (1/(6p²²))·[(m0²+m1²) − 2A0(m1) + 2m0²·B0
                                + 4(m1²−m0²)·B1 − 2m0²p²·dB0/dp²
                                + 4fp²·dB1/dp²]
    where f = p²+m0²−m1².
    """

    nargs = 3
    latex_name = r"\frac{\partial B_{11}}{\partial p^2}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        D = m0**2 - m1**2
        f = p2 + m0**2 - m1**2
        return ((m0**2 + m1**2)
                - 2 * A0(m1)
                + 2 * m0**2 * B0(p2, m0, m1)
                - 4 * D * B1(p2, m0, m1)
                - 2 * m0**2 * p2 * dB0_dp2(p2, m0, m1)
                + 4 * f * p2 * dB1_dp2(p2, m0, m1)) / (6 * p2**2)

    def _derivative(self, sym):
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # dB11/dp²(0, 0, m1) = 1/(12m1²)
        _, _, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 12) / m1**2

    def _eval_p2_zero_m1_zero(self, part="full"):
        # dB11/dp²(0, m0, 0) = 1/(4m0²)  [from ∫₀¹ x³/(m0²(1-x)·(1-x)) ... via direct integral]
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 4) / m0**2

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # dB11/dp²(0, m, m) = 1/(20m²)  [from ∫₀¹ x³(1−x)/m² dx]
        _, m0, _ = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        return sp.Rational(1, 20) / m0**2

    def _eval_p2_zero(self, part="full"):
        # dB11/dp²(0, m0, m1) with m0≠m1, both nonzero.
        # Derived by integrating x³(1−x)/(m0²(1−x)+m1²x) analytically.
        _, m0, m1 = map(sp.simplify, self.args)
        if part == "pole":
            return sp.S.Zero
        D = m1**2 - m0**2
        return ((m1**6 - 5 * m0**2 * m1**4 + 13 * m0**4 * m1**2 + 3 * m0**6) / (12 * D**4)
                - m0**6 * m1**2 * sp.log(m1**2 / m0**2) / D**5)

    def _eval_general(self, part="full"):
        # p²≠0 reduction formula
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return sp.S.Zero
        return ((m0**2 + m1**2)
                - 2 * A0(m1).doit(part=part)
                + 2 * m0**2 * B0(p2, m0, m1).doit(part=part)
                + 4 * (m1**2 - m0**2) * B1(p2, m0, m1).doit(part=part)
                - 2 * m0**2 * p2 * dB0_dp2(p2, m0, m1).doit(part=part)
                + 4 * f * p2 * dB1_dp2(p2, m0, m1).doit(part=part)) / (6 * p2**2)

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate dB11/dp² numerically (UV-finite, μ-independent).

        pole → 0.
        finite → ∫₀¹ x³(1−x)/χ dx, complex above threshold.

        Below threshold: GL via x=sin²(πt/2).
        Above threshold: Cauchy-PV subtraction; analytic Im.
        """
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

        if part == "pole":
            return np.zeros_like(p2v)

        n_pts = len(p2v)
        result = np.empty(n_pts, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(_GL_W[:, np.newaxis] * (x**3 * (1 - x) / chi) * jac, axis=0)
            return result.real + 0j

        for i in range(n_pts):
            result[i] = _dbn_scalar(p2v[i], m0v[i], m1v[i], 2, above[i])

        return result


# ---------------------------------------------------------------------------
# dB00/dp² — first derivative of B00 w.r.t. p²  (UV-divergent, pole = −1/12)
# ---------------------------------------------------------------------------

class dB00_dp2(PVFunction):
    """
    dB00/dp²(p², m0, m1)  (UV-divergent, pole = −1/12)

    d(B00_finite)/dp² = (1/2)·∫₀¹ x(1−x)·ln(χ/μ²) dx
    Full: dB00/dp² = −1/(12ε̄) + d(B00_finite)/dp²

    Reduction formula at p²≠0 (from differentiating B00):
        dB00/dp² = (1/6)·[−1/3 − B1 + 2m0²·dB0/dp² − f·dB1/dp²]
    where f = p²+m0²−m1².  Pole = −B1_pole/6 = −1/12.

    SYMMETRIC under m0↔m1 (inherited from B00).
    """

    nargs = 3
    latex_name = r"\frac{\partial B_{00}}{\partial p^2}"

    def __new__(cls, *args):
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}")
        return super().__new__(cls, *args)

    def reduce(self):
        p2, m0, m1 = self.args
        if p2.is_zero:
            return None
        f = p2 + m0**2 - m1**2
        pole = sp.Rational(-1, 12) / epsilon_bar
        finite = (sp.Rational(-1, 3)
                  - B1(p2, m0, m1)
                  + 2 * m0**2 * dB0_dp2(p2, m0, m1)
                  - f * dB1_dp2(p2, m0, m1)) / 6
        return pole + finite

    def _derivative(self, sym):
        return sp.S.Zero

    # --- Symbolic evaluation ---

    def _eval_p2_zero_m0_zero(self, part="full"):
        # dB00/dp²(0, 0, m1) = −Δ(m1)/12 − 5/72
        _, _, m1 = map(sp.simplify, self.args)
        pole = sp.Rational(-1, 12)
        finite = sp.log(m1**2 / mu**2) / 12 - sp.Rational(5, 72)
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero_m1_zero(self, part="full"):
        # dB00/dp²(0, m0, 0) = −Δ(m0)/12 − 5/72  [by m0↔m1 symmetry]
        _, m0, _ = map(sp.simplify, self.args)
        pole = sp.Rational(-1, 12)
        finite = sp.log(m0**2 / mu**2) / 12 - sp.Rational(5, 72)
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero_m0_eq_m1(self, part="full"):
        # dB00/dp²(0, m, m) = −Δ(m)/12
        _, m0, _ = map(sp.simplify, self.args)
        pole = sp.Rational(-1, 12)
        finite = sp.log(m0**2 / mu**2) / 12
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_p2_zero(self, part="full"):
        # dB00/dp²(0, m0, m1), m0≠m1, m0≠0, m1≠0  — guide.tex §4, eq. (548) symmetric form:
        #   −1/12·(1/ε̄ − ln(m0·m1/μ²))
        #   − (1/72)·[(5m0⁴−22m0²m1²+5m1⁴)/(m0²−m1²)²
        #              − 3·(m0⁶−3m0⁴m1²−3m0²m1⁴+m1⁶)/(m0²−m1²)³·ln(m0²/m1²)]
        _, m0, m1 = map(sp.simplify, self.args)
        D = m0**2 - m1**2
        pole = sp.Rational(-1, 12)
        finite = (sp.log(m0 * m1 / mu**2) / 12
                  - (5*m0**4 - 22*m0**2*m1**2 + 5*m1**4) / (72 * D**2)
                  + (m0**6 - 3*m0**4*m1**2 - 3*m0**2*m1**4 + m1**6) / (24 * D**3)
                  * sp.log(m0**2 / m1**2))
        if part == "pole":
            return pole / epsilon_bar
        if part == "finite":
            return finite
        return pole / epsilon_bar + finite

    def _eval_general(self, part="full"):
        # p²≠0 reduction formula: dB00/dp² = (1/6)[−1/3 − B1 + 2m0²·dB0 − f·dB1]
        p2, m0, m1 = map(sp.simplify, self.args)
        f = p2 + m0**2 - m1**2
        if part == "pole":
            return sp.Rational(-1, 12) / epsilon_bar
        # finite: B1_pole (1/2) contributes nothing to finite part
        return (sp.Rational(-1, 3)
                - B1(p2, m0, m1).doit(part=part)
                + 2 * m0**2 * dB0_dp2(p2, m0, m1).doit(part=part)
                - f * dB1_dp2(p2, m0, m1).doit(part=part)) / 6

    def _eval(self, part="full", **hints):
        p2, m0, m1 = map(sp.simplify, self.args)
        if p2.is_zero:
            if m0.is_zero:
                return self._eval_p2_zero_m0_zero(part)
            if m1.is_zero:
                return self._eval_p2_zero_m1_zero(part)
            if (m0 - m1).is_zero:
                return self._eval_p2_zero_m0_eq_m1(part)
            return self._eval_p2_zero(part)
        return self._eval_general(part)

    # --- Numerical evaluation ---

    def numeric(self):
        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m0v, m1v, mu_val, part):
        """
        Evaluate dB00/dp² numerically.

        pole   → −1/12 (constant, kinematic-independent).
        finite → (1/2)·∫₀¹ x(1−x)·ln(χ/μ²) dx, complex above threshold.

        Below threshold: GL via x=sin²(πt/2).
        Above threshold: split at χ=0 roots x1,x2;
            Im = −(π/2)·∫_{x1}^{x2} x(1−x) dx  (negative, from Im[lnχ]=−π for χ<0).
        """
        p2v, m0v, m1v = (np.asarray(a, dtype=float) for a in
                         np.broadcast_arrays(p2v, m0v, m1v))
        p2v = np.atleast_1d(p2v.copy())
        m0v = np.atleast_1d(m0v.copy())
        m1v = np.atleast_1d(m1v.copy())

        if part == "pole":
            return np.full_like(p2v, -1.0 / 12.0)

        n_pts = len(p2v)
        result = np.empty(n_pts, dtype=complex)
        above = p2v > (m0v + m1v)**2

        if not np.any(above):
            t = _GL_X[:, np.newaxis]
            x = np.sin(np.pi * t / 2)**2
            jac = (np.pi / 2) * np.sin(np.pi * t)
            chi = m0v**2 * (1 - x) + m1v**2 * x - x * (1 - x) * p2v
            result = np.sum(
                _GL_W[:, np.newaxis] * (0.5 * x * (1 - x) * np.log(chi / mu_val**2) * jac),
                axis=0)
            return result.real + 0j

        for i in range(n_pts):
            result[i] = _db00_scalar(p2v[i], m0v[i], m1v[i], mu_val, above[i])

        return result
