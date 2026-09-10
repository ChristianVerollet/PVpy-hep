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

import functools
import itertools
import operator
from collections import defaultdict
from sympy.tensor.tensor import TensorHead
from sympy.physics.hep.gamma_matrices import gamma_trace
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence

import sympy as sp

from .algebra import g, Mom, Dot, Eps, simplify_external_dots
from .functions import A0, A00, B0, B1, B11, B00


# ---------------------------------------------------------------------------
# 2. LoopIntegral container
# ---------------------------------------------------------------------------

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
        """Label i (1..N-1) -> (q_0 - q_i), the external momentum basis for the PV ansatz.

        The PV decomposition convention is k^μ = p_ext^μ * B1 + ...,
        where p_ext = q_0 - q_i is the momentum flowing into propagator i.
        Using the propagator shift q_i directly (= -p_ext for standard routing)
        would give wrong signs for all odd-rank tensor coefficients.
        """
        q0 = self.propagators[0].shift
        return {i: sp.expand(q0 - self.propagators[i].shift) for i in range(1, self.N)}

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
    "A0":  A0,
    "A00": A00,
    "B0":  B0,
    "B1":  B1,
    "B00": B00,
    "B11": B11,
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
        if not hasattr(term, "components"):
            # Plain scalar (e.g. a trace term that vanished identically to 0).
            # NB: can't use `getattr(term, "coeff", term)` here -- every plain
            # sympy Expr (Integer, Symbol, ...) already has an unrelated
            # `.coeff` *method* (Expr.coeff), so the getattr default never
            # triggers and silently returns a bound method instead of `term`.
            out += term
            continue
        coeff = term.coeff
        components = term.components
        idxs = list(term.get_indices())

        occurrences = []  # (kind, momentum_symbol_or_None, index, comp_id)
        pos = 0
        for comp_id, comp in enumerate(components):
            n = len(comp.index_types)
            comp_idxs = idxs[pos:pos + n]
            pos += n
            if comp.name == "metric":
                occurrences.append(("metric", None, comp_idxs[0], comp_id))
                occurrences.append(("metric", None, comp_idxs[1], comp_id))
            else:
                mom_symbol = momentum_map.get(comp)
                if mom_symbol is None:
                    raise KeyError(f"no momentum_map entry for tensor head {comp}")
                occurrences.append(("mom", mom_symbol, comp_idxs[0], comp_id))

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
        for (k1, m1, _idx1, _c1), (k2, m2, _idx2, _c2) in pairs:
            if k1 == "metric" and k2 == "metric":
                term_val *= sp.Symbol("d")
            else:
                term_val *= Dot(m1, m2)

        # Free metric indices must be grouped by the tensor component (i.e. the
        # individual metric(.,.) factor) they came from -- a term can contain
        # several independent, fully-free metric factors (e.g. the 4-gamma
        # trace tr[g^mu g^nu g^rho g^sigma] produces g(mu,nu)*g(rho,sigma)-like
        # terms), and lumping all their indices into one flat list would mix
        # indices from different factors together.
        free_metric_by_comp: Dict[int, List[sp.Symbol]] = {}
        for kind, _, idx, comp_id in free:
            if kind == "metric":
                free_metric_by_comp.setdefault(comp_id, []).append(sp.Symbol(idx.name))
        free_mom = [(mom, sp.Symbol(idx.name)) for kind, mom, idx, _ in free if kind == "mom"]

        for idx_list in free_metric_by_comp.values():
            if len(idx_list) == 2:
                term_val *= g(*idx_list)
            elif len(idx_list) == 1:
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
#   Tr[γ₅ γ^{i1} γ^{i2} γ^{i3} γ^{i4}] = GAMMA5_TRACE_COEFF * Eps(i1,i2,i3,i4)
#
# Metric (+,−,−,−), ε_{0123} = +1 (lower, "physics" convention).
# γ₅ = −i γ^0 γ^1 γ^2 γ^3  (Itzykson-Zuber / Package-X sign)
# → Tr[γ₅ γ_μ γ_ν γ_ρ γ_σ] = −4i ε_{μνρσ}  (lower ε)
# → Tr[γ₅ γ^μ γ^ν γ^ρ γ^σ] = +4i ε^{μνρσ}  (upper ε, ε^{0123} = −1 from ε_{0123}=+1)
#
# Use GAMMA5_TRACE_COEFF = −4*I (P&S sign) if you define γ₅ = +i γ^0 γ^1 γ^2 γ^3 instead.

GAMMA5_TRACE_COEFF = 4 * sp.I

G5 = TensorHead("G5", [])  # zero-index marker: insert G5() at gamma5's position in the product


def _cancel_g5_pairs(term):
    """
    Reduce a TensMul's G5() count to 0 or 1 using G5**2 = 1.

    Repeatedly takes the first two G5 factors (in physical left-to-right
    order, via term.args -- see CLAUDE.md bug #6, zero-index TensorHeads
    keep their position under multiplication) and slides the second one
    left to meet the first: each GammaMatrix it passes contributes a (-1)
    from the anticommutator {gamma5, gamma^mu} = 0, then the adjacent
    G5*G5 = 1 pair is simply dropped. Momentum-head factors (the slash()
    partner of a GammaMatrix) are ordinary vectors, not gamma matrices, so
    they contribute no sign and are left in place.
    """
    factors = list(getattr(term, "args", [term]))
    g5_idx = [i for i, f in enumerate(factors)
              if getattr(getattr(f, "component", None), "name", None) == "G5"]
    if len(g5_idx) < 2:
        return term

    sign = 1
    while len(g5_idx) >= 2:
        i, j = g5_idx[0], g5_idx[1]
        n_between = sum(
            1 for f in factors[i + 1:j] if f.component.name == "GammaMatrix"
        )
        sign *= (-1) ** n_between
        del factors[j]
        del factors[i]
        g5_idx = [i for i, f in enumerate(factors)
                  if getattr(getattr(f, "component", None), "name", None) == "G5"]

    reduced = functools.reduce(operator.mul, factors) if factors else sp.Integer(1)
    return sign * reduced


def gamma5_trace(expr, momentum_map: Dict[object, sp.Expr]) -> sp.Expr:
    """
    Trace of an expression containing any number of G5() markers per term.
    Pairs of G5's are pre-cancelled via G5**2 = 1 (see _cancel_g5_pairs
    below), which anticommutes the second G5 of each pair leftward past
    every intervening GammaMatrix -- physically this is what lets a chiral
    (V-A)-type coupling squared, e.g.
    `(g_V - g_A*Gamma5())*Gamma(mu)*...*(g_V + g_A*Gamma5())*Gamma(nu)`,
    reduce its g_A**2 term (two G5's, not adjacent) back to an ordinary
    trace. After cancellation, at most one G5 remains per term. Returns a
    sympy expression in g()/Mom()/Dot()/Eps() language directly (bypasses
    gamma_trace entirely for the single-G5 case, since it doesn't know
    about G5).
    """
    from sympy.tensor.tensor import TensAdd
    expr = expr.expand()
    out = 0
    for term in expr.args if isinstance(expr, (sp.Add, TensAdd)) else [expr]:
        term = _cancel_g5_pairs(term)

        if not hasattr(term, "components"):
            # All G5's cancelled with nothing left (e.g. a bare Gamma5()*Gamma5()
            # term) -- what remains is Tr[term * 1] = term * d.
            out += term * sp.Symbol("d")
            continue

        coeff = sp.sympify(term.coeff)
        components = list(term.components)
        idxs = list(term.get_indices())

        n_g5 = sum(1 for c in components if c.name == "G5")
        if n_g5 == 0:
            out += from_gamma_trace(gamma_trace(term), momentum_map)
            continue

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
    """
    Reduce a numerator expression to PV functions.

    Handles:
    - Spectator factors (g, Mom(p,...), couplings, Eps with no k-slot)
    - Scalar k-dependence: Dot(k,k), Dot(k,V)
    - Open k-indices: Mom(k, mu)
    - Eps(k, s2, s3, s4): k in an Eps slot is treated as a rank-1 open index
      (via a dummy Lorentz index), reduced by TensorialDecomposition, then
      contracted back into the Eps slot with contract_with_eps.
    """
    k = loop_integral.k
    numerator = sp.expand(simplify_external_dots(sp.expand(numerator), k))

    result = 0
    for term in numerator.as_ordered_terms():
        factors = term.as_ordered_factors()
        open_k = [f for f in factors if isinstance(f, Mom) and f.args[0] == k]
        scalar_k = [f for f in factors if isinstance(f, Dot) and k in f.args]
        eps_factors = [f for f in factors if isinstance(f, Eps)]
        spectator = [f for f in factors
                     if f not in open_k and f not in scalar_k and f not in eps_factors]

        # For each Eps with k in a slot, introduce a dummy Lorentz index so the
        # k-dependence feeds into TensorialDecomposition as an extra open index.
        # Eps(k, s2, s3, s4) -> dummy d, Eps(d, s2, s3, s4) + virtual Mom(k, d).
        # contract_with_eps later substitutes the reduced p_i back into slot d.
        eps_dummies = []    # (dummy_idx, Eps_with_dummy_replacing_k)
        eps_spectators = [] # Eps factors with no k-slot (pure spectators)
        for eps_f in eps_factors:
            k_slot = next((i for i, a in enumerate(eps_f.args) if a == k), None)
            if k_slot is None:
                eps_spectators.append(eps_f)
            else:
                d = sp.Dummy("eps_mu")
                new_args = list(eps_f.args)
                new_args[k_slot] = d
                eps_dummies.append((d, Eps(*new_args)))

        spectator_part = (sp.Mul(*(spectator + eps_spectators))
                          if (spectator + eps_spectators) else sp.Integer(1))

        # Effective open k-indices: explicit Mom(k, mu) + one virtual per Eps k-slot
        effective_open_k = open_k + [Mom(k, d) for d, _ in eps_dummies]

        if not effective_open_k:
            k_part = sp.Mul(*scalar_k) if scalar_k else sp.Integer(1)
            _validate_numerator(k_part, k)
            result += spectator_part * reduce_scalar_k_dependence(k_part, loop_integral)
            continue

        # Mixed term: build open-rank tensor, contract extra dummy slots back down.
        free_idx_syms = [f.args[1] for f in effective_open_k]
        dummy_pairs, dummy_single = [], []
        for f in scalar_k:
            a, b = f.args
            partner = b if a == k else a
            if partner == k:
                d1, d2 = sp.Dummy(), sp.Dummy()
                dummy_pairs.append((d1, d2))
            else:
                dummy_single.append((sp.Dummy(), partner))

        all_indices = (free_idx_syms
                       + [d for pair in dummy_pairs for d in pair]
                       + [d for d, _ in dummy_single])
        rank = len(all_indices)
        tensor_loop = LoopIntegral(k=k, propagators=loop_integral.propagators, rank=rank)
        tensor_result = TensorialDecomposition(tensor_loop, indices=all_indices)

        for d1, d2 in dummy_pairs:
            tensor_result = contract_with_metric(tensor_result, d1, d2)
        for d, V in dummy_single:
            tensor_result = contract_with_momentum(tensor_result, d, V)
        for eps_d, eps_with_d in eps_dummies:
            tensor_result = contract_with_eps(tensor_result, eps_d, eps_with_d)

        result += spectator_part * tensor_result
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


def contract_with_eps(expr: sp.Expr, dummy_idx, eps_obj: "Eps") -> sp.Expr:
    """
    Replace a free dummy index in an Eps slot by the momentum/index carried by the
    corresponding Mom(p, dummy_idx) or g(dummy_idx, other) factor in each term of expr.

    Used after TensorialDecomposition to close the loop between an Eps k-slot and the
    rank-1 (or higher) tensor reduction:
        Eps(k, s2, s3, s4) → introduce dummy d, reduce → Mom(p_i, d) × PV_i
        → contract_with_eps → PV_i × Eps(p_i, s2, s3, s4)
    """
    expr = sp.expand(expr)
    eps_args = list(eps_obj.args)
    slot = eps_args.index(dummy_idx)

    out = 0
    for term in expr.as_ordered_terms():
        replacement, rest = None, []
        for f in term.as_ordered_factors():
            if isinstance(f, Mom) and f.args[1] == dummy_idx:
                replacement = f.args[0]          # momentum p
            elif isinstance(f, g) and dummy_idx in f.args:
                replacement = f.args[0] if f.args[1] == dummy_idx else f.args[1]
            else:
                rest.append(f)
        if replacement is None:
            raise ValueError(
                f"Eps dummy index {dummy_idx} not found in term '{term}'. "
                "The rank passed to TensorialDecomposition was probably wrong."
            )
        new_eps_args = eps_args[:]
        new_eps_args[slot] = replacement
        out += sp.Mul(*rest) * Eps(*new_eps_args)
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

    # --- ReduceGeneralNumerator with Eps from a chiral trace ---
    # Self-energy: Tr[g5 k/ g^mu (k/+p/) g^nu] / (k^2*(k-p)^2) = 4i*eps^{k,mu,nu,p}
    # After integration, k^alpha -> p1^alpha * B1, so Eps(p1,mu,nu,p1) = 0.
    k_sym = sp.Symbol("k")
    bubble_li = LoopIntegral(
        k=k_sym,
        propagators=[Propagator(shift=sp.Integer(0), mass=m0),
                     Propagator(shift=p1, mass=m1)],
        rank=0,
    )
    g5_reduced = ReduceGeneralNumerator(bubble_li, g5_result)
    print("ReduceGeneralNumerator(Eps chiral self-energy) :", g5_reduced,
          " [expect 0 by antisymmetry]")