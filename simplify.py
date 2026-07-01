"""
Kinematic substitution and PV function reduction utilities.

Recommended workflow
--------------------
1. Compute a loop amplitude symbolically → expression in PV functions.
2. Apply kinematic conditions with set_kinematics().
3. Reduce derived PV functions to primitives with reduce_pv().

The ORDER matters.  If you expand B1 into its formula first (step 3 before
step 2), you expose a 1/(2*p²) denominator.  Substituting p²=0 afterward
gives division by zero even though the full amplitude may be finite, because
the cancellation between the numerator and denominator only becomes visible
at the level of the complete amplitude.  Substituting first lets terms like
p²·B1(p², m0, m1) collapse to 0·B1(0, m0, m1) = 0 before any formula with
a p²-denominator is touched.
"""

import sympy as sp
from .base import PVFunction


def set_kinematics(expr: sp.Expr, substitutions: dict) -> sp.Expr:
    """
    Substitute kinematic conditions into a PV-function expression.

    Parameters
    ----------
    expr : sympy expression containing PV function objects.
    substitutions : dict mapping sympy Symbols to their values, e.g.
                    {p2: 0} or {m1: m2} or {p2: 0, m0: 0}.

    Returns
    -------
    The expression after substitution, expanded.

    Example
    -------
    >>> from pvpy.symbols import p2, m0, m1
    >>> from pvpy.functions.tensors import B1
    >>> from pvpy.functions.scalar import B00
    >>> amp = p2 * B1(p2, m0, m1) + B00(p2, m0, m1)
    >>> set_kinematics(amp, {p2: 0})
    B00(0, m0, m1)
    """
    return sp.expand(expr.subs(substitutions))


def reduce_pv(expr: sp.Expr, max_iter: int = 20) -> sp.Expr:
    """
    Recursively replace derived PV functions with their reduction formulas.

    Primitive PV functions — A0, B0, B00, and their derivatives — have
    reduce() == None and are left unevaluated.  Derived functions — A00,
    B1, and any future tensor coefficients — are expanded into the
    primitives via their .reduce() method.

    Parameters
    ----------
    expr : sympy expression (output of ReduceGeneralNumerator or a
           user-assembled combination of PV objects).
    max_iter : safety cap on recursion depth (default 20 is generous for
               any realistic PV hierarchy).

    Returns
    -------
    Fully reduced sympy expression containing only primitive PV functions.

    Raises
    ------
    NotImplementedError
        If a derived PV function's reduction formula is undefined for the
        specific kinematic point (e.g. B1(0, m0, m1) whose 1/(2*p²)
        formula is singular at p²=0).  Fix: call set_kinematics() first.

    Example
    -------
    >>> from pvpy.symbols import p2, m0, m1
    >>> from pvpy.functions.tensors import A00
    >>> reduce_pv(A00(m0))
    m0**2*A0(m0)/4 + m0**4/8
    """
    for _ in range(max_iter):
        changed = False
        for atom in expr.atoms(PVFunction):
            formula = atom.reduce()
            if formula is not None:
                expr = expr.subs(atom, formula)
                changed = True
        expr = sp.expand(expr)
        if not changed:
            break
    return expr
