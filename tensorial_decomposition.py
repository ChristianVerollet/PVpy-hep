"""
Tensorial decomposition engine for 1-loop integrals (Passarino-Veltman style).

Convention (IMPORTANT, fix this once and stick to it everywhere):
    Propagators are D_i = (k + q_i)^2 - m_i^2,  i = 0, ..., N-1
    with q_0 = 0 (you always have the freedom to shift k so the FIRST
    propagator is q_0 = 0). The q_i, i=1..N-1, are then your N-1
    independent "routing momenta" and ARE your independent external momenta
    for this topology.

This module only builds the ANSATZ (tensor structures + which PV
coefficient multiplies each one) and dispatches to whatever scalar PV
functions you already have implemented. It does NOT do the
Gram-matrix reduction -- that's a separate "solve for the new
coefficients" step you plug in via PV_REGISTRY (or leave symbolic until
you implement it).
"""

import itertools
from collections import defaultdict
from sympy.tensor.tensor import TensorHead
from sympy.physics.hep.gamma_matrices import gamma_trace
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence

import sympy as sp


# ---------------------------------------------------------------------------
# 1. Lorentz-tensor building blocks (pure sympy Function objects)
# ---------------------------------------------------------------------------

class g(sp.Function):
    """Metric tensor g^{mu nu}, symmetric under index exchange."""
    nargs = 2

    @classmethod
    def eval(cls, mu, nu):
        if sp.sympify(mu).sort_key() > sp.sympify(nu).sort_key():
            return cls(nu, mu)

    def _sympystr(self, printer):
        a, b = self.args
        return f"g^{{{a}{b}}}"

    def _latex(self, printer):
        a, b = self.args
        return r"g^{%s%s}" % (printer._print(a), printer._print(b))


class Mom(sp.Function):
    """Component of a momentum: Mom(p, mu) represents p^mu. Linear in p."""
    nargs = 2

    @classmethod
    def eval(cls, p, mu):
        p = sp.expand(p)
        if p.is_Add:
            return sp.Add(*[cls(term, mu) for term in p.args])
        coeff, rest = p.as_coeff_Mul()
        if coeff != 1 and rest != 0:
            return coeff * cls(rest, mu)
        if p == 0:
            return sp.Integer(0)

    def _sympystr(self, printer):
        p, mu = self.args
        p_str = printer._print(p)
        if not p.is_Atom:
            p_str = f"({p_str})"
        return f"{p_str}^{{{mu}}}"

    def _latex(self, printer):
        p, mu = self.args
        p_str = printer._print(p)
        if not p.is_Atom:
            p_str = r"\left(%s\right)" % p_str
        return r"%s^{%s}" % (p_str, printer._print(mu))


def simplify_external_dots(expr: sp.Expr, k: sp.Symbol) -> sp.Expr:
    """Auto-rewrite Dot(a,b) for a,b != k via a.a -> a^2, a.b -> ((a+b)^2-a^2-b^2)/2. Leaves Dot(k,...) untouched."""
    subs = {}
    for d in expr.atoms(Dot):
        a, b = d.args
        if a == k or b == k:
            continue
        subs[d] = a**2 if a == b else sp.expand((a + b) ** 2 - a ** 2 - b ** 2) / 2
    return expr.subs(subs) if subs else expr


# ---------------------------------------------------------------------------
# 2. LoopIntegral container
# ---------------------------------------------------------------------------

class Dot(sp.Function):
    """Scalar dot product placeholder: Dot(a, b) represents a.b (symmetric)."""
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        if sp.sympify(a).sort_key() > sp.sympify(b).sort_key():
            return cls(b, a)

    def _sympystr(self, printer):
        a, b = self.args
        return f"({printer._print(a)}.{printer._print(b)})"

    def _latex(self, printer):
        a, b = self.args
        return r"%s\!\cdot\!%s" % (printer._print(a), printer._print(b))


@dataclass
class Propagator:
    shift: sp.Expr      # q_i, a linear combination of external-momentum symbols
    mass: sp.Expr       # m_i
    power: int = 1      # n_i, exponent of D_i in the denominator


@dataclass
class LoopIntegral:
    k: sp.Symbol
    propagators: List[Propagator]
    rank: Optional[int] = None  # only needed for TensorialDecomposition (Method B)

    def __post_init__(self):
        if sp.expand(self.propagators[0].shift) != 0:
            raise ValueError(
                "Convention violation: propagators[0].shift must be 0 "
                "(choose the loop-momentum routing so the first propagator "
                "is k^2 - m0^2)."
            )

    @property
    def N(self) -> int:
        return len(self.propagators)

    @property
    def independent_momenta(self) -> Dict[int, sp.Expr]:
        """Label i (1..N-1) -> q_i. These are the basis momenta used in the ansatz."""
        return {i: self.propagators[i].shift for i in range(1, self.N)}

    @property
    def masses(self) -> List[sp.Expr]:
        return [p.mass for p in self.propagators]


# ---------------------------------------------------------------------------
# 3. Combinatorial ansatz generator (Wick-contraction structure)
# ---------------------------------------------------------------------------

def _contractions(slots: Sequence[int]):
    """Yield every (pairs, singles) split of a list of index slots."""
    if not slots:
        yield [], []
        return
    first, rest = slots[0], slots[1:]
    for pairs, singles in _contractions(rest):
        yield pairs, [first] + singles
    for i in range(1, len(slots)):
        partner = slots[i]
        remaining = slots[1:i] + slots[i + 1:]
        for pairs, singles in _contractions(remaining):
            yield [(first, partner)] + pairs, singles


def tensor_structures(rank: int, n_ext: int):
    """
    Group all contractions of `rank` open slots by their PV-coefficient label.

    Label convention matches standard PV notation: a pair of contracted
    slots contributes two leading zeros (e.g. one pair -> "00", giving B00,
    C001 etc.), followed by the sorted momentum labels of the open slots.
    """
    groups = defaultdict(list)
    for pairs, singles in _contractions(list(range(rank))):
        r = len(pairs)
        if not singles:
            groups[tuple([0] * (2 * r))].append((pairs, []))
            continue
        for moms in itertools.product(range(1, n_ext + 1), repeat=len(singles)):
            label = tuple([0] * (2 * r) + sorted(moms))
            groups[label].append((pairs, list(zip(singles, moms))))
    return dict(groups)


def _basis_tensor(group_terms, idx_symbols, momenta: Dict[int, sp.Expr]):
    total = 0
    for pairs, singles in group_terms:
        term = sp.Integer(1)
        for (a, b) in pairs:
            term *= g(idx_symbols[a], idx_symbols[b])
        for (slot, mom_label) in singles:
            term *= Mom(momenta[mom_label], idx_symbols[slot])
        total += term
    return sp.expand(total)


def pv_name(N: int, label: Tuple[int, ...]) -> str:
    letter = {1: "A", 2: "B", 3: "C", 4: "D"}[N]
    if not label:
        return letter + "0"
    return letter + "".join(str(x) for x in label)


# ---------------------------------------------------------------------------
# 4. Dispatch to your existing scalar PV functions
# ---------------------------------------------------------------------------
# Fill this in with the functions you already implemented. Anything not
# listed here falls back to an undefined sympy Function with the right
# name/arguments, so the pipeline still produces a well-formed expression
# you can fill in later as you implement more PV functions (B1, B11, C0, ...).

PV_REGISTRY: Dict[str, callable] = {
    # "A0": A0,
    # "B0": B0,
    # "B00": B00,
}


def kinematic_args(loop_integral: LoopIntegral) -> tuple:
    """
    Build the argument tuple passed to the PV function for this topology.

    *** ADAPT THIS to match the exact signature of your existing A0/B0/B00/C0 ***
    Current convention used here: (invariants..., masses...)
    with invariants = [(q_i - q_0)^2 for i in 1..N-1] in propagator order.
    """
    qs = [p.shift for p in loop_integral.propagators]
    invariants = [sp.expand((qs[i] - qs[0]) ** 2) for i in range(1, loop_integral.N)]
    return tuple(invariants + loop_integral.masses)


def get_pv_function(name: str, args: tuple) -> sp.Expr:
    if name in PV_REGISTRY:
        return PV_REGISTRY[name](*args)
    return sp.Function(name)(*args)  # placeholder, keeps the expression well-formed


# ---------------------------------------------------------------------------
# 5. The main entry point
# ---------------------------------------------------------------------------

def TensorialDecomposition(loop_integral: LoopIntegral,
                            indices: Optional[Sequence[sp.Symbol]] = None) -> sp.Expr:
    N, R = loop_integral.N, loop_integral.rank

    if R == 0:
        name = pv_name(N, ())
        return get_pv_function(name, kinematic_args(loop_integral))

    if indices is None or len(indices) != R:
        raise ValueError(f"rank-{R} integral needs exactly {R} free Lorentz indices")

    n_ext = N - 1
    groups = tensor_structures(R, n_ext)
    momenta = loop_integral.independent_momenta
    args = kinematic_args(loop_integral)

    result = 0
    for label, terms in groups.items():
        coeff = get_pv_function(pv_name(N, label), args)
        result += coeff * _basis_tensor(terms, indices, momenta)
    return sp.expand(result)


# ---------------------------------------------------------------------------
# 5b. GENERAL numerator reduction (Method A): algebraic substitution
# ---------------------------------------------------------------------------
# Handles ANY numerator built from:
#   - "spectator" tensor factors not involving k (g(mu,nu), Mom(p,mu), couplings
#     like (1-xi)) -- these ride through unchanged, attached to whatever scalar
#     PV combination the k-dependent part of their term reduces to.
#   - scalar k-dependence written with Dot(k,k), Dot(k, p) -- NOT open Mom(k,mu)
#     factors (that case is TensorialDecomposition/Method B above).
#   - propagators raised to any integer power (handled via the standard
#     d/dm^2 trick, since 1/D^n = (-1)^(n-1)/(n-1)! * d^(n-1)/d(m^2)^(n-1) [1/D]).
#
# This directly generalizes contract_with_metric/contract_with_momentum: those
# were special cases of "substitute Dot(k,k) or Dot(k,p) and simplify" applied
# only to pure-rank numerators with power-1 propagators.

def decompose_in_basis(V: sp.Expr, basis: Dict[int, sp.Expr]) -> Dict[int, sp.Expr]:
    """Solve V = sum_i c_i * basis[i] for the c_i (all linear in elementary momentum symbols)."""
    idxs = list(basis.keys())
    elem_syms = set()
    for v in list(basis.values()) + [V]:
        elem_syms |= sp.sympify(v).free_symbols
    elem_syms = sorted(elem_syms, key=str)

    c = sp.symbols(f"c0:{len(idxs)}")
    expr = sp.expand(V - sum(ci * basis[i] for ci, i in zip(c, idxs)))
    eqs = [expr.coeff(s, 1) for s in elem_syms]
    const_part = expr
    for s in elem_syms:
        const_part = const_part.subs(s, 0)
    eqs.append(const_part)

    sol = sp.solve(eqs, c, dict=True)
    if not sol:
        raise ValueError(
            f"momentum {V} is not expressible in this topology's basis momenta "
            f"{list(basis.values())} -- check your propagator routing."
        )
    sol = sol[0]
    return {i: sol.get(ci, 0) for ci, i in zip(c, idxs)}


def derivative_wrt_mass_squared(expr: sp.Expr, m: sp.Symbol, order: int) -> sp.Expr:
    """d^order/d(m^2)^order of expr, via the chain rule (expr is written in terms of m, not m^2)."""
    if order == 0:
        return expr
    Msq = sp.Dummy("Msq", positive=True)
    in_Msq = expr.subs(m, sp.sqrt(Msq))
    d = sp.diff(in_Msq, Msq, order)
    return d.subs(Msq, m**2)


def reduce_scalar_k_dependence(k_poly_expr: sp.Expr, loop_integral: LoopIntegral) -> sp.Expr:
    """
    Reduce a SCALAR polynomial in Dot(k,k)/Dot(k, V) to a sum of (mass-derivatives of)
    scalar PV functions, using the propagator powers on loop_integral.
    """
    N = loop_integral.N
    k = loop_integral.k
    qs = [p.shift for p in loop_integral.propagators]
    masses = [p.mass for p in loop_integral.propagators]
    powers = [p.power for p in loop_integral.propagators]
    D = sp.symbols(f"D0:{N}")
    basis = {i: qs[i] for i in range(1, N)}

    full_subs = {}
    for dot in k_poly_expr.atoms(Dot):
        a, b = dot.args
        if a == k and b == k:
            full_subs[dot] = D[0] + masses[0] ** 2
            continue
        other = b if a == k else a
        if N == 1:
            raise ValueError("Dot(k, V) requested but this topology has no external momenta")
        coeffs = decompose_in_basis(other, basis)
        expr_i = 0
        for i, c in coeffs.items():
            expr_i += c * (D[i] - D[0] - sp.expand(qs[i] ** 2) + masses[0] ** 2 - masses[i] ** 2) / 2
        full_subs[dot] = expr_i

    poly_expr = sp.expand(k_poly_expr.subs(full_subs)) if full_subs else sp.expand(k_poly_expr)
    poly = sp.Poly(poly_expr, *D) if N > 0 else None

    result = 0
    terms = poly.terms() if poly is not None else [((), poly_expr)]
    for monom, coeff in terms:
        net = [powers[i] - monom[i] for i in range(N)]
        if any(n < 0 for n in net):
            raise NotImplementedError(
                "numerator degree exceeds combined denominator power for some propagator "
                "in this term -- needs a second reduction pass (re-expand the surplus "
                "D_i back into k^2/k.q_i and substitute again), not yet implemented."
            )
        surviving = [i for i in range(N) if net[i] > 0]
        if not surviving:
            continue  # scaleless integral -> 0 in dim reg
        sub_qs = [qs[i] for i in surviving]
        sub_masses = [masses[i] for i in surviving]
        # use fresh dummy masses so we can differentiate symbolically even when
        # the real mass is a fixed numeric value (e.g. 0 for a massless line)
        dummies = [sp.Dummy(f"m{i}", positive=True) for i in surviving]
        sub_loop = LoopIntegral(
            k=k,
            propagators=[Propagator(sp.expand(sub_qs[j] - sub_qs[0]), dummies[j])
                         for j in range(len(surviving))],
        )
        pv = get_pv_function(pv_name(len(surviving), ()), kinematic_args(sub_loop))
        differentiated = []
        for j, i in enumerate(surviving):
            order = net[i] - 1
            if order > 0:
                pv = (-1) ** order / sp.factorial(order) * derivative_wrt_mass_squared(pv, dummies[j], order)
                differentiated.append(j)
        for j in range(len(surviving)):
            if j in differentiated:
                pv = sp.limit(pv, dummies[j], sub_masses[j])  # robust to removable singularities at m->0
            else:
                pv = pv.subs(dummies[j], sub_masses[j])
        result += coeff * pv
    return result


def _validate_numerator(numerator: sp.Expr, k: sp.Symbol) -> None:
    """Catch the dangerous silent-failure case: k used outside Dot(...)/Mom(...) wrappers."""
    stripped = numerator.subs({f: sp.Dummy() for f in numerator.atoms(Dot, Mom)})
    if k in stripped.free_symbols:
        raise ValueError(
            f"loop momentum '{k}' appears outside of Dot(k,...) or Mom(k,...) in the "
            f"numerator -- write k^2 as Dot(k,k), k.p as Dot(k,p), and an open k^mu as "
            f"Mom(k,mu). A bare power of k is not recognized and would silently pass "
            f"through unreduced."
        )


class Eps(sp.Function):
    """
    Totally antisymmetric Levi-Civita object, Eps(s1,s2,s3,s4). Each slot is
    EITHER a free Lorentz index (a bare Symbol -> open slot) OR a momentum
    expression (-> that slot is understood as contracted with that momentum,
    exactly like Mom packs a momentum+index into one object). Linear in any
    momentum slot; antisymmetric under exchange of any two slots; vanishes
    if two slots coincide.
    """
    nargs = 4

    @classmethod
    def eval(cls, *args):
        args = list(args)
        for i, a in enumerate(args):
            a = sp.sympify(a)
            a_exp = sp.expand(a)
            if a_exp.is_Add:
                return sp.Add(*[cls(*(args[:i] + [term] + args[i + 1:])) for term in a_exp.args])
            coeff, rest = a_exp.as_coeff_Mul()
            if coeff != 1 and rest != 0:
                return coeff * cls(*(args[:i] + [rest] + args[i + 1:]))
        for i in range(4):
            for j in range(i + 1, 4):
                if args[i] == args[j]:
                    return sp.Integer(0)
        keys = [sp.sympify(a).sort_key() for a in args]
        order = sorted(range(4), key=lambda i: keys[i])
        if order == [0, 1, 2, 3]:
            return None  # already canonical
        sign = sp.combinatorics.Permutation(order).signature()
        return sign * cls(*[args[i] for i in order])

    def _sympystr(self, printer):
        def fmt(a):
            s = printer._print(a)
            return f"({s})" if not sp.sympify(a).is_Atom else s
        return "eps^{" + ",".join(fmt(a) for a in self.args) + "}"

    def _latex(self, printer):
        def fmt(a):
            s = printer._print(a)
            return r"\left(%s\right)" % s if not sp.sympify(a).is_Atom else s
        return r"\epsilon^{%s}" % ",".join(fmt(a) for a in self.args)


# ---------------------------------------------------------------------------
# 5d. Gamma traces (no gamma-5)
# ---------------------------------------------------------------------------

def from_gamma_trace(trace_expr, momentum_map: Dict[object, sp.Expr]) -> sp.Expr:
    """
    Convert output of sympy.physics.hep.gamma_matrices.gamma_trace into our
    g(mu,nu) / Mom(p,mu) / Dot(a,b) language.

    momentum_map: {TensorHead: our_momentum_symbol}
    Contracted index pairs (same name, opposite is_up) -> Dot(...).
    Free indices -> open Mom(...) or g(...) slots.
    """
    from sympy.tensor.tensor import TensAdd
    trace_expr = trace_expr.expand()
    out = 0
    for term in trace_expr.args if isinstance(trace_expr, (sp.Add, TensAdd)) else [trace_expr]:
        coeff = getattr(term, "coeff", term)
        components = getattr(term, "components", [])
        idxs = list(getattr(term, "get_indices", lambda: [])())

        occurrences = []  # (kind, momentum_symbol_or_None, index)
        pos = 0
        for comp in components:
            n = len(comp.index_types)
            comp_idxs = idxs[pos:pos + n]
            pos += n
            if comp.name == "metric":
                occurrences.append(("metric", None, comp_idxs[0]))
                occurrences.append(("metric", None, comp_idxs[1]))
            else:
                mom_symbol = momentum_map.get(comp)
                if mom_symbol is None:
                    raise KeyError(f"no momentum_map entry for tensor head {comp}")
                occurrences.append(("mom", mom_symbol, comp_idxs[0]))

        used = [False] * len(occurrences)
        pairs, free = [], []
        for i in range(len(occurrences)):
            if used[i]:
                continue
            for j in range(i + 1, len(occurrences)):
                if not used[j] and occurrences[i][2].name == occurrences[j][2].name \
                        and occurrences[i][2].is_up != occurrences[j][2].is_up:
                    pairs.append((occurrences[i], occurrences[j]))
                    used[i] = used[j] = True
                    break
            if not used[i]:
                free.append(occurrences[i])

        term_val = sp.sympify(coeff)
        for (k1, m1, _idx1), (k2, m2, _idx2) in pairs:
            if k1 == "metric" and k2 == "metric":
                term_val *= sp.Symbol("d")
            else:
                term_val *= Dot(m1, m2)

        free_metric_idxs = [sp.Symbol(idx.name) for kind, _, idx in free if kind == "metric"]
        free_mom = [(mom, sp.Symbol(idx.name)) for kind, mom, idx in free if kind == "mom"]
        if len(free_metric_idxs) == 2:
            term_val *= g(*free_metric_idxs)
        elif len(free_metric_idxs) == 1:
            raise NotImplementedError("metric with one free / one contracted-to-momentum index "
                                       "not yet supported by this bridge")
        for mom, idx in free_mom:
            term_val *= Mom(mom, idx)

        out += term_val
    return sp.expand(out)


# ---------------------------------------------------------------------------
# 5e. Gamma-5 traces
# ---------------------------------------------------------------------------
# Convention (state explicitly, flip GAMMA5_TRACE_COEFF if yours differs):
#   Tr[g5 * gamma^i1 gamma^i2 gamma^i3 gamma^i4] = GAMMA5_TRACE_COEFF * Eps(i1,i2,i3,i4)
# Standard choice: GAMMA5_TRACE_COEFF = -4*I  (g5 = i*g0g1g2g3, Tr[1]=4, eps^{0123}=+1)

GAMMA5_TRACE_COEFF = -4 * sp.I

G5 = TensorHead("G5", [])  # zero-index marker: insert G5() at gamma5's position in the product


def gamma5_trace(expr, momentum_map: Dict[object, sp.Expr]) -> sp.Expr:
    """
    Trace of an expression containing exactly one G5() marker per term
    (terms with zero or an even number of G5 insertions should instead be
    passed through ordinary gamma_trace, since G5**2 = 1). Returns a sympy
    expression in g()/Mom()/Dot()/Eps() language directly (bypasses
    gamma_trace entirely, since it doesn't know about G5).
    """
    from sympy.tensor.tensor import TensAdd
    expr = expr.expand()
    out = 0
    for term in expr.args if isinstance(expr, (sp.Add, TensAdd)) else [expr]:
        coeff = sp.sympify(getattr(term, "coeff", term))
        components = list(getattr(term, "components", []))
        idxs = list(getattr(term, "get_indices", lambda: [])())

        n_g5 = sum(1 for c in components if c.name == "G5")
        if n_g5 == 0:
            out += from_gamma_trace(gamma_trace(term), momentum_map)
            continue
        if n_g5 != 1:
            raise NotImplementedError("more than one G5 marker in a single term not supported")

        g5_pos = next(i for i, c in enumerate(components) if c.name == "G5")
        n_gammas_before = sum(1 for c in components[:g5_pos] if c.name == "GammaMatrix")
        sign = (-1) ** n_gammas_before  # move G5 to the front, past each gamma it anticommutes with

        gamma_components = [c for c in components if c.name == "GammaMatrix"]
        if len(gamma_components) not in (0, 1, 2, 3, 4):
            raise NotImplementedError(
                f"gamma5 trace with {len(gamma_components)} gamma matrices not implemented "
                f"(only 0-4 supported; 0-3 vanish, 4 uses the standard Eps identity)"
            )
        if len(gamma_components) < 4:
            continue  # trace of g5 with <4 gammas vanishes identically

        # walk through components (skipping the G5 marker), separating the 4 gamma
        # "slots" themselves from momentum-head occurrences that decorate a slash
        gamma_slots = []   # list of (gamma_index)
        mom_occurrences = []  # list of (momentum_symbol, index)
        pos = 0
        for comp in components:
            n = len(comp.index_types)
            comp_idxs = idxs[pos:pos + n]
            pos += n
            if comp.name == "G5":
                continue
            elif comp.name == "GammaMatrix":
                gamma_slots.append(comp_idxs[0])
            else:
                mom_symbol = momentum_map.get(comp)
                if mom_symbol is None:
                    raise KeyError(f"no momentum_map entry for tensor head {comp}")
                mom_occurrences.append((mom_symbol, comp_idxs[0]))

        # decorate each gamma slot: pair with a momentum occurrence sharing its index
        # name (opposite is_up) if one exists, else it stays a free Lorentz index
        eps_args = []
        used_mom = [False] * len(mom_occurrences)
        for g_idx in gamma_slots:
            decorated = False
            for j, (mom_sym, m_idx) in enumerate(mom_occurrences):
                if not used_mom[j] and g_idx.name == m_idx.name and g_idx.is_up != m_idx.is_up:
                    eps_args.append(mom_sym)
                    used_mom[j] = True
                    decorated = True
                    break
            if not decorated:
                eps_args.append(sp.Symbol(g_idx.name))

        if len(eps_args) != 4:
            raise NotImplementedError(
                "a gamma slot in the gamma5 trace appears self-contracted with another "
                "gamma slot directly (not via a momentum head) -- this reduces the rank "
                "below 4 and isn't handled here."
            )

        out += coeff * sign * GAMMA5_TRACE_COEFF * Eps(*eps_args)
    return sp.expand(out)


# ---------------------------------------------------------------------------
# 5c. Unified reducer: per-term open k-indices (Method B ansatz) combined
#     with leftover scalar k-dependence (Method A), in the same numerator.
#     This is what gamma-matrix traces actually produce.
# ---------------------------------------------------------------------------

def ReduceGeneralNumerator(loop_integral: LoopIntegral, numerator: sp.Expr) -> sp.Expr:
    k = loop_integral.k
    numerator = sp.expand(simplify_external_dots(sp.expand(numerator), k))

    result = 0
    for term in numerator.as_ordered_terms():
        factors = term.as_ordered_factors()
        open_k = [f for f in factors if isinstance(f, Mom) and f.args[0] == k]
        scalar_k = [f for f in factors if isinstance(f, Dot) and k in f.args]
        spectator = [f for f in factors if f not in open_k and f not in scalar_k]
        spectator_part = sp.Mul(*spectator) if spectator else sp.Integer(1)

        if not open_k:
            # pure Method A: same as ReduceLoopIntegral's per-term logic
            k_part = sp.Mul(*scalar_k) if scalar_k else sp.Integer(1)
            _validate_numerator(k_part, k)
            result += spectator_part * reduce_scalar_k_dependence(k_part, loop_integral)
            continue

        # Mixed term: R_open free k-indices, plus possibly Dot(k,k)/Dot(k,V) factors
        # that pin down EXTRA (dummy) k-indices. Build the full open-rank tensor at
        # rank = R_open + 2*deg(Dot(k,k)) + sum(deg(Dot(k,V_j))) via TensorialDecomposition,
        # then contract the dummy slots away with the existing contraction helpers.
        free_idx_syms = [f.args[1] for f in open_k]
        dummy_pairs, dummy_single = [], []
        for f in scalar_k:
            a, b = f.args
            partner = b if a == k else a
            if partner == k:  # Dot(k,k)
                d1, d2 = sp.Dummy(), sp.Dummy()
                dummy_pairs.append((d1, d2))
            else:
                dummy_single.append((sp.Dummy(), partner))

        all_indices = free_idx_syms + [d for pair in dummy_pairs for d in pair] + [d for d, _ in dummy_single]
        rank = len(all_indices)
        tensor_loop = LoopIntegral(k=k, propagators=loop_integral.propagators, rank=rank)
        tensor_result = TensorialDecomposition(tensor_loop, indices=all_indices)

        for d1, d2 in dummy_pairs:
            tensor_result = contract_with_metric(tensor_result, d1, d2)
        for d, V in dummy_single:
            tensor_result = contract_with_momentum(tensor_result, d, V)

        result += spectator_part * tensor_result
    return sp.expand(result)
    """
    General entry point: numerator is a sympy expression built from
    spectator tensors (g(mu,nu), Mom(p,mu), coupling factors) times scalar
    k-dependence written as Dot(k,k) / Dot(k, p). Open Mom(k,mu) factors are
    NOT supported here -- use TensorialDecomposition for genuine open-k rank.
    """
    k = loop_integral.k
    numerator = sp.expand(numerator)
    numerator = simplify_external_dots(numerator, k)
    numerator = sp.expand(numerator)
    _validate_numerator(numerator, k)
    result = 0
    for term in numerator.as_ordered_terms():
        factors = term.as_ordered_factors()
        k_factors = [f for f in factors if isinstance(f, Dot) and k in f.args]
        open_k_factors = [f for f in factors if isinstance(f, Mom) and f.args[0] == k]
        if open_k_factors:
            raise NotImplementedError(
                f"term '{term}' has an open k-index (Mom(k,...)) mixed with other "
                f"factors -- ReduceLoopIntegral only handles fully-contracted scalar "
                f"k-dependence (Dot(k,k), Dot(k,p)). Genuine open k^mu indices need "
                f"TensorialDecomposition; combining both in one numerator isn't wired "
                f"up yet."
            )
        spectator_factors = [f for f in factors if f not in k_factors]
        k_part = sp.Mul(*k_factors) if k_factors else sp.Integer(1)
        spectator_part = sp.Mul(*spectator_factors) if spectator_factors else sp.Integer(1)
        result += spectator_part * reduce_scalar_k_dependence(k_part, loop_integral)
    return sp.expand(result)


# ---------------------------------------------------------------------------
# 6. Self-consistency check: trace identity
# ---------------------------------------------------------------------------
# Cheap, very useful sanity check that doesn't need numerics: contracting
# rank-2 result with g_{mu nu} must reproduce (up to a 1/d or d factor,
# depending on your scheme) a combination of lower-rank scalar integrals.
# Wire this up once you have d-dimensional g^{mu}_{mu} = d available.


# ---------------------------------------------------------------------------
# 7. Contraction helpers, for fully-contracted ("scalar") numerators
#    like k^2 or k.p1 -- built on top of an open-index decomposition.
# ---------------------------------------------------------------------------

def contract_with_momentum(expr: sp.Expr, mu: sp.Symbol, p: sp.Expr) -> sp.Expr:
    """Sum free index `mu` against an external momentum p (numerator ~ k.p1 case)."""
    expr = sp.expand(expr)
    out = 0
    for term in expr.as_ordered_terms():
        new_factors, consumed = [], False
        for f in term.as_ordered_factors():
            if isinstance(f, g) and mu in f.args:
                other = f.args[0] if f.args[1] == mu else f.args[1]
                new_factors.append(Mom(p, other))
                consumed = True
            elif isinstance(f, Mom) and f.args[1] == mu:
                new_factors.append(sp.expand(f.args[0] * p))  # q.p -> dot product
                consumed = True
            else:
                new_factors.append(f)
        out += sp.Mul(*new_factors) if consumed else term
    return sp.expand(out)


def contract_with_metric(expr: sp.Expr, mu: sp.Symbol, nu: sp.Symbol,
                          d: sp.Symbol = sp.Symbol("d")) -> sp.Expr:
    """Sum the index pair (mu, nu) against g_{mu nu} (numerator ~ k^2 case)."""
    expr = sp.expand(expr)
    out = 0
    for term in expr.as_ordered_terms():
        factors = term.as_ordered_factors()
        f_mu = next((f for f in factors if (isinstance(f, g) and mu in f.args)
                     or (isinstance(f, Mom) and f.args[1] == mu)), None)
        f_nu = next((f for f in factors if (isinstance(f, g) and nu in f.args)
                     or (isinstance(f, Mom) and f.args[1] == nu)), None)
        rest = [f for f in factors if f not in (f_mu, f_nu)]
        if isinstance(f_mu, g) and f_mu == f_nu:           # g(mu,nu) itself
            out += sp.Mul(*rest) * d
        elif isinstance(f_mu, Mom) and isinstance(f_nu, Mom):  # Mom(p,mu)*Mom(q,nu)
            out += sp.Mul(*rest) * sp.expand(f_mu.args[0] * f_nu.args[0])
        else:
            raise ValueError(f"unhandled term when contracting mu,nu: {term}")
    return sp.expand(out)


if __name__ == "__main__":
    mu, nu, rho = sp.symbols("mu nu rho")
    p1 = sp.Symbol("p1")
    m0, m1 = sp.symbols("m0 m1")

    # --- N=1 (tadpole), rank 0 and rank 2 ---
    tadpole = LoopIntegral(
        k=sp.Symbol("k"),
        propagators=[Propagator(shift=sp.Integer(0), mass=m0)],
        rank=0,
    )
    print("A0 :", TensorialDecomposition(tadpole))

    tadpole_r2 = LoopIntegral(k=sp.Symbol("k"),
                               propagators=[Propagator(shift=sp.Integer(0), mass=m0)],
                               rank=2)
    print("A^{mu nu} :", TensorialDecomposition(tadpole_r2, indices=[mu, nu]))

    # --- N=2 (bubble), rank 0, 1, 2 ---
    bubble = lambda r: LoopIntegral(
        k=sp.Symbol("k"),
        propagators=[Propagator(shift=sp.Integer(0), mass=m0),
                     Propagator(shift=p1, mass=m1)],
        rank=r,
    )
    print("B0 :", TensorialDecomposition(bubble(0)))
    print("B^{mu} :", TensorialDecomposition(bubble(1), indices=[mu]))
    print("B^{mu nu} :", TensorialDecomposition(bubble(2), indices=[mu, nu]))

    # scalar numerators built from the open-index results above
    d = sp.Symbol("d")
    k_dot_p1 = contract_with_momentum(TensorialDecomposition(bubble(1), indices=[mu]), mu, p1)
    print("k.p1 :", k_dot_p1)
    k_squared = contract_with_metric(TensorialDecomposition(bubble(2), indices=[mu, nu]), mu, nu, d=d)
    print("k^2  :", k_squared)

    # --- N=3 (triangle), rank 0 and rank 3 (your hardest current case) ---
    p2 = sp.Symbol("p2")
    m2 = sp.Symbol("m2")
    triangle = lambda r: LoopIntegral(
        k=sp.Symbol("k"),
        propagators=[Propagator(shift=sp.Integer(0), mass=m0),
                     Propagator(shift=p1, mass=m1),
                     Propagator(shift=p1 + p2, mass=m2)],
        rank=r,
    )
    print("C0 :", TensorialDecomposition(triangle(0)))
    print("C^{mu nu rho} :", TensorialDecomposition(triangle(3), indices=[mu, nu, rho]))

    # --- gamma_5: standard chiral self-energy trace structure ---
    from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex
    from sympy.tensor.tensor import tensor_indices as _ti
    mu_t, nu_t, ga, gb = _ti("mu,nu,a,b", LorentzIndex)
    k_head = TensorHead("k", [LorentzIndex])
    p_head = TensorHead("p", [LorentzIndex])
    g5_expr = G5() * G(ga) * k_head(-ga) * G(mu_t) * \
        (G(gb) * k_head(-gb) + G(gb) * p_head(-gb)) * G(nu_t)
    g5_result = gamma5_trace(g5_expr, {k_head: sp.Symbol("k"), p_head: p1})
    print("Tr[g5 slash(k) g^mu slash(k+p) g^nu] :", g5_result)