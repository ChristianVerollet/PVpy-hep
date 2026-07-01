"""
pvpy.dirac — user-friendly wrappers for Dirac traces.

Provides slash(), Gamma(), Gamma5(), and DiracTrace() so that traces can be
written in physics notation without knowledge of sympy's tensor index
machinery.  The result is returned in pvpy's g / Mom / Dot language and
feeds directly into ReduceGeneralNumerator or reduce_pv.

Typical usage
-------------
    from pvpy.dirac import slash, Gamma, Gamma5, DiracTrace
    import sympy as sp

    p1, p2, m = sp.symbols('p1 p2 m', positive=True)
    mu, nu    = sp.symbols('mu nu')

    # Standard (non-chiral) trace
    result = DiracTrace(
        (slash(p1) - m) * Gamma(mu) * (slash(p2) - m) * Gamma(nu)
    )
    # → 4*m**2*g(mu,nu) - 4*Dot(p1,p2)*g(mu,nu)
    #   + 4*Mom(p1,mu)*Mom(p2,nu) + 4*Mom(p1,nu)*Mom(p2,mu)

    # Chiral trace (auto-detected when Gamma5() is present)
    chi = DiracTrace(Gamma5() * slash(k) * Gamma(mu) * slash(p) * Gamma(nu))
    # → 4*I*Eps(k, mu, nu, p)

Note on the spacetime dimension
--------------------------------
The dimension d appears as the sympy Symbol 'd' in results (e.g. in
Tr[1] = d, or Tr[γ^μ γ^ν] = d·g^{μν}).  Substitute with d=4 when working
in exactly four dimensions:  result.subs(sp.Symbol('d'), 4)

Note on momentum arguments to slash()
--------------------------------------
slash() accepts a single sympy Symbol.  For a shifted momentum like p+q,
define a new symbol or expand the product manually before tracing.
"""

import sympy as sp
from sympy.physics.hep.gamma_matrices import GammaMatrix as _G, LorentzIndex
from sympy.tensor.tensor import tensor_indices, TensorHead

from .tensorial_decomposition import from_gamma_trace, gamma5_trace, G5


# ---------------------------------------------------------------------------
# Internal: unique dummy-index factory
# ---------------------------------------------------------------------------

_idx_counter = [0]


def _fresh_idx():
    """A fresh contracted (dummy) LorentzIndex; never leaks into the output."""
    name = f"_pvpy_c{_idx_counter[0]}"
    _idx_counter[0] += 1
    idx = tensor_indices(name, LorentzIndex)
    return idx[0] if isinstance(idx, (list, tuple)) else idx


def _free_idx(sym):
    """A free LorentzIndex whose name matches the user's sympy Symbol."""
    idx = tensor_indices(sym.name, LorentzIndex)
    return idx[0] if isinstance(idx, (list, tuple)) else idx


# ---------------------------------------------------------------------------
# Display-only tokens: turn a TensMul of GammaMatrix/G5/momentum-head factors
# into a readable noncommutative sympy expression (gamma^mu, slashed p, g5).
# These never appear in DiracTrace() output or in any arithmetic -- they
# exist purely so DiracMatrix.__repr__ / _repr_latex_ can show something
# nicer than an opaque tensor object.
# ---------------------------------------------------------------------------

class _GammaTok(sp.Function):
    nargs = 1
    is_commutative = False

    def _sympystr(self, printer):
        return f"γ^{{{printer._print(self.args[0])}}}"

    def _latex(self, printer):
        return r"\gamma^{%s}" % printer._print(self.args[0])


class _SlashTok(sp.Function):
    nargs = 1
    is_commutative = False

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}̸"

    def _latex(self, printer):
        return r"\not{%s}" % printer._print(self.args[0])


class _Gamma5Tok(sp.Function):
    nargs = 0
    is_commutative = False

    def _sympystr(self, printer):
        return "γ₅"

    def _latex(self, printer):
        return r"\gamma_5"


def _term_factor_tokens(tensor, momentum_map):
    """Ordered list of display tokens for one DiracMatrix term's tensor."""
    if tensor is None:
        return []
    components = list(getattr(tensor, "components", []))
    idxs = list(getattr(tensor, "get_indices", lambda: [])())

    pos = 0
    comp_info = []
    for comp in components:
        n = len(comp.index_types)
        comp_info.append((comp, idxs[pos:pos + n]))
        pos += n

    # A momentum-head component's single index is always the dummy that was
    # contracted with the GammaMatrix introduced by the same slash() call.
    idx_to_mom = {
        cidxs[0].name: momentum_map[comp]
        for comp, cidxs in comp_info
        if comp in momentum_map
    }

    tokens = []
    for comp, cidxs in comp_info:
        if comp.name == "G5":
            tokens.append(_Gamma5Tok())
        elif comp.name == "GammaMatrix":
            idx = cidxs[0]
            mom = idx_to_mom.get(idx.name)
            if mom is not None:
                tokens.append(_SlashTok(mom))
            else:
                tokens.append(_GammaTok(sp.Symbol(idx.name)))
        # momentum-head components themselves contribute no separate token;
        # they're absorbed into the preceding _SlashTok.
    return tokens


def _dirac_string_expr(tensor, momentum_map):
    """The ordered product of a term's tokens, or 1 for the identity."""
    tokens = _term_factor_tokens(tensor, momentum_map)
    expr = sp.Integer(1)
    for tok in tokens:
        expr = expr * tok
    return expr


# ---------------------------------------------------------------------------
# DiracMatrix — the algebra object
# ---------------------------------------------------------------------------

class DiracMatrix:
    """
    Sum of Dirac-space monomials.  Each monomial is a pair

        (scalar_coeff, tensor)

    where tensor is a TensMul / TensAdd of GammaMatrix objects, or None to
    represent the spinor-space identity matrix.

    Build with slash(), Gamma(), Gamma5() and combine with *, +, -.
    """

    def __init__(self, terms, momentum_map=None):
        # Drop zero-coefficient terms eagerly to keep the list short.
        self.terms = [(sp.sympify(c), t) for c, t in terms if sp.sympify(c) != 0]
        self.momentum_map = dict(momentum_map or {})

    # --- unary ---

    def __neg__(self):
        return DiracMatrix([(-c, t) for c, t in self.terms], self.momentum_map)

    # --- addition / subtraction ---

    def _merge(self, other):
        return {**self.momentum_map, **other.momentum_map}

    def __add__(self, other):
        if isinstance(other, DiracMatrix):
            return DiracMatrix(self.terms + other.terms, self._merge(other))
        if isinstance(other, (int, float)) or isinstance(other, sp.Expr):
            return DiracMatrix(
                self.terms + [(sp.sympify(other), None)], self.momentum_map)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DiracMatrix):
            return self + (-other)
        if isinstance(other, (int, float)) or isinstance(other, sp.Expr):
            return DiracMatrix(
                self.terms + [(-sp.sympify(other), None)], self.momentum_map)
        return NotImplemented

    def __rsub__(self, other):
        return (-self) + other

    # --- multiplication ---

    def __mul__(self, other):
        if isinstance(other, DiracMatrix):
            merged, result = self._merge(other), []
            for c1, t1 in self.terms:
                for c2, t2 in other.terms:
                    c = sp.expand(c1 * c2)
                    if c == 0:
                        continue
                    if t1 is None and t2 is None:
                        t = None
                    elif t1 is None:
                        t = t2
                    elif t2 is None:
                        t = t1
                    else:
                        t = (t1 * t2).expand()
                    result.append((c, t))
            return DiracMatrix(result, merged)
        if isinstance(other, (int, float)) or isinstance(other, sp.Expr):
            s = sp.sympify(other)
            return DiracMatrix(
                [(sp.expand(c * s), t) for c, t in self.terms], self.momentum_map)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)) or isinstance(other, sp.Expr):
            s = sp.sympify(other)
            return DiracMatrix(
                [(sp.expand(s * c), t) for c, t in self.terms], self.momentum_map)
        return NotImplemented

    # --- display only (not used by DiracTrace or the arithmetic above) ---

    def as_sympy(self):
        """
        Render this Dirac string as an ordinary sympy expression, for human
        display only (repr / Jupyter). Gamma matrices are wrapped in
        noncommutative marker Functions so that Mul preserves their physical
        left-to-right order instead of sympy's usual commutative sorting.
        """
        if not self.terms:
            return sp.Integer(0)
        return sp.Add(*[
            coeff * _dirac_string_expr(tensor, self.momentum_map)
            for coeff, tensor in self.terms
        ])

    def __repr__(self):
        return sp.sstr(self.as_sympy())

    def _repr_latex_(self):
        return r"$\displaystyle %s$" % sp.latex(self.as_sympy())


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def slash(p):
    """
    Slashed momentum  γ·p = γ^a p_a.

    The contracted Lorentz index is generated automatically; the user never
    has to name it.

    Parameters
    ----------
    p : sympy Symbol
        A single momentum symbol (e.g. sp.Symbol('p1')).
        For shifted momenta like p1+p2, define a new symbol or expand
        the trace product manually.
    """
    a = _fresh_idx()
    head = TensorHead(f"{p.name}_pvpy", [LorentzIndex])
    return DiracMatrix([(sp.Integer(1), _G(a) * head(-a))], {head: p})


def Gamma(mu):
    """
    Gamma matrix  γ^μ  with a free (open) Lorentz index.

    Parameters
    ----------
    mu : sympy Symbol
        Name of the free Lorentz index (e.g. sp.Symbol('mu')).
    """
    return DiracMatrix([(sp.Integer(1), _G(_free_idx(mu)))], {})


def Gamma5():
    """
    γ₅ insertion for chiral traces.

    DiracTrace() auto-detects the presence of Gamma5() and uses the chiral
    trace algorithm automatically.
    """
    return DiracMatrix([(sp.Integer(1), G5())], {})


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

_d = sp.Symbol("d")   # spacetime dimension; substitute d=4 when needed


def DiracTrace(expr):
    """
    Compute Tr[expr] and return the result in pvpy's g / Mom / Dot language.

    The presence of a Gamma5() factor is detected automatically: if any term in
    the expression contains γ₅, the chiral trace algorithm (gamma5_trace) is
    used and the output contains Eps objects.  Otherwise the standard trace
    is computed and the output contains only g, Mom, Dot.

    The spacetime dimension appears as the symbol 'd'.  Substitute d=4 when
    working in exactly four dimensions:  result.subs(sp.Symbol('d'), 4)

    Parameters
    ----------
    expr : DiracMatrix
        Built from slash(), Gamma(), Gamma5() and arithmetic operators.

    Returns
    -------
    sympy expression in terms of g, Mom, Dot  (and Eps for chiral traces).

    Raises
    ------
    TypeError
        If expr is not a DiracMatrix.

    Examples
    --------
    Standard (non-chiral) trace:

        >>> p1, p2, m = sp.symbols('p1 p2 m', positive=True)
        >>> mu, nu    = sp.symbols('mu nu')
        >>> DiracTrace((slash(p1) - m) * Gamma(mu) * (slash(p2) - m) * Gamma(nu))
        4*m**2*g(mu,nu) - 4*Dot(p1,p2)*g(mu,nu)
        + 4*Mom(p1,mu)*Mom(p2,nu) + 4*Mom(p1,nu)*Mom(p2,mu)

    Chiral trace:

        >>> k, p = sp.symbols('k p')
        >>> DiracTrace(Gamma5() * slash(k) * Gamma(mu) * slash(p) * Gamma(nu))
        4*I*Eps(k, mu, nu, p)
    """
    if not isinstance(expr, DiracMatrix):
        raise TypeError(
            "DiracTrace expects a DiracMatrix. "
            "Build it with slash(), Gamma(), Gamma5() and *, +, - operators."
        )

    from sympy.physics.hep.gamma_matrices import gamma_trace as _sp_gamma_trace

    mom_map = expr.momentum_map

    # Auto-detect γ₅: check whether any term's tensor contains the G5 TensorHead.
    has_g5 = any(
        tensor is not None
        and any(getattr(c, "name", "") == "G5"
                for c in getattr(tensor, "components", []))
        for _, tensor in expr.terms
    )

    result = sp.Integer(0)
    for coeff, tensor in expr.terms:
        if coeff == 0:
            continue
        if tensor is None:
            # Tr[identity] = d  (spinor-space dimension)
            result += coeff * _d
        else:
            raw = tensor.expand()
            if has_g5:
                contrib = gamma5_trace(raw, mom_map)
            else:
                traced = _sp_gamma_trace(raw)
                if traced == sp.S.Zero:
                    continue
                contrib = from_gamma_trace(traced, mom_map)
            result += coeff * contrib

    return sp.expand(result)
