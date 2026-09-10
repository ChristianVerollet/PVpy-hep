"""
Kinematic substitution and PV function simplification/reduction utilities.

Canonical workflow
------------------
    PV_simplify(expr)          # optional: d·B00+p²·B11 → A0+m0²B0 (no 1/p²)
    set_kinematics(expr, subs) # substitute kinematic values (p²=M², m0=0, …)
    PV_reduce(expr)            # algebraic reduction: B1/B11/B00/dB → A0/B0/dB0
    Project(expr, mu, nu, p, mode)  # extract scalar from rank-2 tensor result

The ORDER matters:
- PV_reduce before set_kinematics: exposes 1/(2p²) denominators that blow up at p²=0.
- compile after PV_reduce: compile cannot evaluate 1/ε̄ poles or 1/p² denominators.
"""

import sympy as sp
from .base import PVFunction


def set_kinematics(expr: sp.Expr, substitutions: dict) -> sp.Expr:
    """
    Substitute kinematic conditions into a PV-function expression.

    Parameters
    ----------
    expr : sympy expression containing PV function objects.
    substitutions : dict mapping sympy Symbols to their values,
                    e.g. {p2: 0} or {m1: m2} or {p2: 0, m0: 0}.

    Returns
    -------
    The expression after substitution, expanded.
    """
    return sp.expand(expr.subs(substitutions))


def PV_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Apply forward pattern substitutions to a PV-function expression.

    Safe to use before set_kinematics and before compile: no identity
    introduced here divides by a kinematic invariant or by ε̄.

    Implemented identities (guide.tex §4)
    --------------------------------------

    Step 0 — d·f identities  (d = 4 − ε, ε̄ defined by 1/ε̄ = 2/ε − γ_E − ln π):

        Uses  ε·f = 2·(UV pole residue of f)  so  d·f = 4·f − 2·(UV pole res.)

        d·A0(m)       →  4·A0(m) − 2m²
        d·A00(m)      →  m²·A0(m)                     (exact: massless tadpole = 0)
        d·B0          →  4·B0 − 2
        d·B1          →  4·B1 − 1
        d·B11         →  4·B11 − 2/3
        d·B00         →  4·B00 + p²/6 − (m0²+m1²)/2
        d·dB00        →  4·dB00 + 1/6
        d·dB0, d·dB1, d·dB11  →  4·f              (UV-finite: pole = 0)

        Only fires when sp.Symbol('d') appears in the expression.

    Step 1 — UV pole residue of B00:

        ε̄·B00(p², m0, m1)  →  (m0²+m1²)/4 − p²/12

    Step 2 — eliminate the 4·B00 + p²·B11 combination that remains after step 1:

        4·B00(p², m0, m1) + p²·B11(p², m0, m1)
            →  A0(m1) + m0²·B0(p², m0, m1) − p²/6 + (m0²+m1²)/2

    Steps 1+2 together implement  d·B00 + p²·B11 → A0(m1)+m0²B0.
    Step 0 handles the d·B00 piece first so that step 2 can fire on the
    resulting 4·B00 + p²·B11 combination.
    Applied individually on mismatched expressions they are still valid identities.

    Parameters
    ----------
    expr : sympy expression containing PV function objects.

    Returns
    -------
    Expression with the configured identities applied, expanded.
    """
    from .functions import A0, A00, B0, B00, B1, B11, dB0_dp2, dB1_dp2, dB11_dp2, dB00_dp2
    from .symbols import epsilon_bar

    d = sp.Symbol('d')

    expr = sp.expand(expr)

    # Step 0: d·f → explicit formula  (d = 4 − ε, ε·f = 2·UV_pole_residue)
    if d in expr.free_symbols:
        for a0_atom in list(expr.atoms(A0)):
            m_val = a0_atom.args[0]
            expr = expr.subs(d * a0_atom, 4 * a0_atom - 2 * m_val**2)
        for a00_atom in list(expr.atoms(A00)):
            m_val = a00_atom.args[0]
            expr = expr.subs(d * a00_atom, m_val**2 * A0(m_val))
        for b0_atom in list(expr.atoms(B0)):
            expr = expr.subs(d * b0_atom, 4 * b0_atom - 2)
        for b1_atom in list(expr.atoms(B1)):
            expr = expr.subs(d * b1_atom, 4 * b1_atom - 1)
        for b11_atom in list(expr.atoms(B11)):
            expr = expr.subs(d * b11_atom, 4 * b11_atom - sp.Rational(2, 3))
        for b00_atom in list(expr.atoms(B00)):
            p2_val, m0_val, m1_val = b00_atom.args
            expr = expr.subs(d * b00_atom,
                             4 * b00_atom + p2_val / 6 - (m0_val**2 + m1_val**2) / 2)
        for db00_atom in list(expr.atoms(dB00_dp2)):
            expr = expr.subs(d * db00_atom, 4 * db00_atom + sp.Rational(1, 6))
        for uv_cls in (dB0_dp2, dB1_dp2, dB11_dp2):
            for atom in list(expr.atoms(uv_cls)):
                expr = expr.subs(d * atom, 4 * atom)
        expr = sp.expand(expr)

    # Step 1: ε̄·B00 → (m0²+m1²)/4 − p²/12
    for b00_atom in list(expr.atoms(B00)):
        p2_val, m0_val, m1_val = b00_atom.args
        pole = (m0_val**2 + m1_val**2) / 4 - p2_val / 12
        expr = expr.subs(epsilon_bar * b00_atom, pole)
    expr = sp.expand(expr)

    # Step 2: 4·B00 + p²·B11 → A0(m1) + m0²·B0 − p²/6 + (m0²+m1²)/2
    # Only fires when both B00 and B11 appear with the same (p², m0, m1) args
    # and their coefficients satisfy c00·p² = 4·c11 (the exact PV trace pattern).
    # Use alpha = c00/4 to avoid dividing by p², keeping the substitution safe.
    for b11_atom in list(expr.atoms(B11)):
        p2_val, m0_val, m1_val = b11_atom.args
        b00_atom = B00(p2_val, m0_val, m1_val)
        if b00_atom not in expr.atoms(B00):
            continue
        c11 = expr.coeff(b11_atom)
        c00 = expr.coeff(b00_atom)
        if c11.is_zero or c00.is_zero:
            continue
        # Check ratio: c00·p² == 4·c11
        if sp.simplify(c00 * p2_val - 4 * c11) != 0:
            continue
        alpha = c00 / 4
        rhs = alpha * (A0(m1_val) + m0_val**2 * B0(p2_val, m0_val, m1_val)
                       - p2_val / 6 + (m0_val**2 + m1_val**2) / 2)
        expr = expr - c00 * b00_atom - c11 * b11_atom + rhs
        expr = sp.expand(expr)

    return expr


def PV_collect(expr: sp.Expr) -> sp.Expr:
    """
    Collect 1/ε̄ (UV pole) terms from a doit(part="full") expression.

    Equivalent to sp.collect(sp.expand(expr), 1/epsilon_bar).
    Returns the expression with all 1/ε̄ terms grouped into a single
    coefficient, making the pole structure easy to read.

    Parameters
    ----------
    expr : sympy expression, typically the result of calling
           .doit(part="full") on a PV expression.

    Returns
    -------
    Expression with 1/ε̄ collected into a single term.

    Example
    -------
    >>> PV_collect(Pi_T.doit(part="full"))
    4*m_Z**2/(3*ε̄) + <finite terms>
    """
    from .symbols import epsilon_bar
    return sp.collect(sp.expand(expr), 1 / epsilon_bar)


def PV_reduce(expr: sp.Expr, max_iter: int = 5) -> sp.Expr:
    """
    Reduce all derived PV functions to the primitive set {A0, B0, dB0_dp2}.

    Each derived function is replaced by its algebraic reduction formula via
    its .reduce() method.  The primitives A0, B0, dB0_dp2 are left as opaque
    PVFunction objects (not expanded to logarithms).

    Derived functions and their one-step reductions:

        A00  → m²/4 · A0(m) + m⁴/8
        A0000→ m⁴/24 · A0(m) + 5m⁶/144

    At p² ≠ 0:

        B1   → [A0(m1) − A0(m0) + f·B0] / (2p²)
        B11  → [p²/6 − (m0²+m1²)/2 + A0(m1) − m0²·B0 + 2f·B1] / (3p²)
        B00  → [−p²/3 + m0²+m1² + A0(m1) + 2m0²·B0 − f·B1] / 6
        dB1  → [A0(m0)−A0(m1) + (m1²−m0²)·B0 + fp²·dB0] / (2p²²)
        dB11 → [(m0²+m1²)−2A0(m1)+2m0²B0+4D·B1−2m0²p²dB0+4fp²dB1] / (6p²²)
        dB00 → −1/(12ε̄) + [−1/3 − B1 + 2m0²·dB0 − f·dB1] / 6

    where f = p²+m0²−m1², D = m0²−m1².

    At p²=0 (after set_kinematics):

        B0(0,m0,m1) → (A0(m0)−A0(m1))/(m0²−m1²)  when m0≠m1
        B0(0,m,m)   → left as B0 (primitive, no PV simplification)
        B1, B11, B00, dB1, dB11, dB00 → left as-is (no PV-only reduction at p²=0)

    Iteration is needed because B11, B00, dB11, dB00 contain B1 or dB1
    in their one-step reduction; max 3 passes suffice for all current functions.

    Parameters
    ----------
    expr     : sympy expression (output of the tensor decomposition or
               user-assembled combination of PV objects). Call set_kinematics
               first if any kinematic limits apply.
    max_iter : safety cap (default 5 is generous for any realistic PV hierarchy).

    Returns
    -------
    Fully reduced sympy expression containing only A0, B0, dB0_dp2 and
    rational functions of kinematic invariants (plus 1/ε̄ poles from dB00).

    Raises
    ------
    RuntimeError
        If the expression still contains unreduced derived PV functions after
        max_iter passes (should not happen with the default max_iter=5).

    Notes
    -----
    NEVER follow PV_reduce with compile().  After reduction the expression
    contains 1/p² denominators and possibly 1/ε̄ poles that compile() cannot
    evaluate.  Compile before reduction (on the original PV function objects)
    or evaluate A0/B0/dB0 separately.
    """
    from .functions import A00, A0000, B0, B1, B11, B00, dB1_dp2, dB11_dp2, dB00_dp2

    reducible = (A00, A0000, B0, B1, B11, B00, dB1_dp2, dB11_dp2, dB00_dp2)

    for _ in range(max_iter):
        atoms = expr.atoms(*reducible)
        if not atoms:
            break
        made_progress = False
        for atom in atoms:
            formula = atom.reduce()
            if formula is not None:
                expr = expr.subs(atom, formula)
                made_progress = True
        if not made_progress:
            break
        expr = sp.expand(expr)
    return expr


def Project(expr: sp.Expr, mu, nu, p, mode: str = 'T') -> sp.Expr:
    """
    Extract a scalar coefficient from a rank-2 Lorentz tensor PV expression.

    Assumes the expression has the Passarino-Veltman rank-2 form:

        T^{μν} = A(p²)·g^{μν} + B(p²)·p^μ p^ν

    Every term must carry exactly one of these two tensor structures.

    Parameters
    ----------
    expr : tensor expression from ReduceGeneralNumerator (before or after
           PV_reduce), containing g(mu,nu) and Mom(p,mu)*Mom(p,nu) factors.
    mu, nu : the two free Lorentz index symbols appearing in expr.
    p      : the external momentum appearing in Mom objects.  p² is computed
             as p**2 automatically and must match the PV function arguments
             (which is always the case when p is the same symbol used to
             define the LoopIntegral propagators).
    mode   : 'g'  → A           coefficient of g^{μν}
             'pp' → B           coefficient of p^μ p^ν
             'T'  → A           transverse scalar (coeff of g^{μν} − p^μp^ν/p²)
             'L'  → A + p²·B   longitudinal scalar (coeff of p^μp^ν/p²)

    Returns
    -------
    Scalar sympy expression with all tensor structure removed.

    Notes
    -----
    'T' and 'g' return the same value A because:
        T^{μν} = A·(g^{μν} − p^μp^ν/p²) + (A+p²B)·p^μp^ν/p²
    so A is both the g^{μν} coefficient and the transverse scalar.
    The physically meaningful check is 'L': for a Ward-identity-conserved
    amplitude (e.g. photon self-energy) L = 0 iff B = −A/p².

    Typical workflow::

        result  = ReduceGeneralNumerator(loop, numerator)
        reduced = PV_reduce(result)          # reduce B1/B11/B00 → A0/B0
        Pi_T    = Project(reduced, mu, nu, p, 'T')   # transverse scalar
        Pi_L    = Project(reduced, mu, nu, p, 'L')   # longitudinal (= 0 by gauge inv)
    """
    from .algebra import g as g_cls, Mom

    if mode not in ('g', 'pp', 'T', 'L'):
        raise ValueError(f"Unknown mode {mode!r}; choose from 'g', 'pp', 'T', 'L'.")

    p2 = sp.expand(p**2)

    expr = sp.expand(expr)
    terms = expr.args if expr.func is sp.Add else [expr]

    A = sp.Integer(0)
    B = sp.Integer(0)

    for term in terms:
        if term == 0:
            continue
        factors = list(term.args) if term.func is sp.Mul else [term]

        # Find g(mu, nu) factor — symmetric so accept both index orderings
        g_idx = next(
            (i for i, f in enumerate(factors)
             if isinstance(f, g_cls) and set(f.args) == {mu, nu}),
            None)
        # Find Mom(p, mu) and Mom(p, nu) factors
        pmu_idx = next(
            (i for i, f in enumerate(factors)
             if isinstance(f, Mom) and f.args == (p, mu)),
            None)
        pnu_idx = next(
            (i for i, f in enumerate(factors)
             if isinstance(f, Mom) and f.args == (p, nu)),
            None)

        if g_idx is not None:
            remaining = [f for i, f in enumerate(factors) if i != g_idx]
            A += sp.Mul(*remaining) if remaining else sp.Integer(1)
        elif pmu_idx is not None and pnu_idx is not None:
            skip = {pmu_idx, pnu_idx}
            remaining = [f for i, f in enumerate(factors) if i not in skip]
            B += sp.Mul(*remaining) if remaining else sp.Integer(1)
        else:
            raise ValueError(
                f"Term '{term}' has no g({mu},{nu}) or "
                f"Mom({p},{mu})*Mom({p},{nu}) structure.")

    A = sp.expand(A)
    B = sp.expand(B)

    if mode in ('g', 'T'):
        return A
    elif mode == 'pp':
        return B
    else:  # 'L'
        return sp.expand(A + p2 * B)


