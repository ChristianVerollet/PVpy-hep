"""
pvpy.algebra — Lorentz-tensor building blocks.

These are the custom sympy Function objects that form the "object language"
used throughout pvpy:  g, Mom, Dot, Eps.

All other pvpy modules import from here.  Users can also import directly:

    from pvpy.algebra import g, Mom, Dot, Eps
"""

import sympy as sp


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
    """Component of a momentum: Mom(p, mu) represents p^mu.  Linear in p."""
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


class Dot(sp.Function):
    """Scalar dot product placeholder: Dot(a, b) represents a·b (symmetric)."""
    nargs = 2

    @classmethod
    def eval(cls, a, b):
        if sp.sympify(a).sort_key() > sp.sympify(b).sort_key():
            return cls(b, a)

    def _sympystr(self, printer):
        a, b = self.args
        if a == b:
            a_str = printer._print(a)
            if not a.is_Atom:
                a_str = f"({a_str})"
            return f"{a_str}^2"
        return f"({printer._print(a)}.{printer._print(b)})"

    def _latex(self, printer):
        a, b = self.args
        if a == b:
            a_str = printer._print(a)
            if not a.is_Atom:
                a_str = r"\left(%s\right)" % a_str
            return r"%s^{2}" % a_str
        return r"%s\!\cdot\!%s" % (printer._print(a), printer._print(b))


class Eps(sp.Function):
    """
    Totally antisymmetric Levi-Civita object, Eps(s1,s2,s3,s4).

    Each slot is either a free Lorentz index (a bare Symbol → open slot) or a
    momentum expression (→ that slot is contracted with that momentum, exactly
    like Mom packs momentum+index together).  Linear in any momentum slot;
    antisymmetric under exchange of any two slots; vanishes if two slots
    coincide.
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


def contract(expr: sp.Expr, d: sp.Symbol = sp.Symbol("d")) -> sp.Expr:
    """
    Apply Einstein summation: contract all repeated Lorentz indices automatically.

    Works on any expression built from g, Mom, Dot, Eps.  Indices are detected
    by their position in each object:
        g(a, b)        — both a and b are indices
        Mom(p, mu)     — only mu is an index; p is a momentum
        Eps(a, b, c, e) — all four slots (can be indices or momenta)

    An index that appears in two different factors is contracted.  Iteration
    continues until no repeated indices remain.

    Rules applied:
        g(a, a)             → d   (spacetime dimension, default sp.Symbol('d'))
        g(a, b)*Mom(p, a)   → Mom(p, b)
        g(a, b)*g(a, c)     → g(b, c)
        Mom(p, a)*Mom(q, a) → Dot(p, q)
        g(a, b)*Eps(a,...) → Eps(b,...)
        Mom(p, a)*Eps(a,...) → Eps(p,...)
    """
    if not isinstance(expr, sp.Basic):
        raise TypeError(
            f"contract() expects a sympy expression (g/Mom/Dot/Eps), "
            f"got {type(expr).__name__}.\n"
            f"Hint: take the trace first — contract(DiracTrace(expr) * other), "
            f"not contract(expr * other)."
        )
    expr = sp.expand(expr)
    if expr.is_Add:
        return sp.expand(sp.Add(*[contract(t, d) for t in expr.args]))
    return sp.expand(_contract_term(expr, d))


def _contract_term(term: sp.Expr, d: sp.Symbol) -> sp.Expr:
    for _ in range(100):
        new = _try_one_contraction(term, d)
        if new is None:
            return term
        term = sp.expand(new)
    raise RuntimeError("contract: did not converge — possible circular rule")


def _try_one_contraction(term: sp.Expr, d: sp.Symbol):
    """Apply the first contraction rule found; return the result or None."""
    from collections import defaultdict

    factors = term.as_ordered_factors() if term.is_Mul else [term]
    tensor_factors, scalar_part = [], []
    for f in factors:
        if isinstance(f, (g, Mom, Eps)):
            tensor_factors.append(f)
        elif (isinstance(f, sp.Pow) and isinstance(f.base, (g, Mom, Eps))
              and f.exp.is_integer and f.exp.is_positive):
            # sympy collapses e.g. g(mu,nu)*g(mu,nu) → g(mu,nu)**2 ;
            # expand back into n copies so the contraction rules can fire.
            tensor_factors.extend([f.base] * int(f.exp))
        else:
            scalar_part.append(f)

    if not tensor_factors:
        return None

    # Map index symbol → positions in tensor_factors where it appears.
    # Convention: for Mom(p, mu) only mu is tracked (p is a momentum, not an index).
    idx_map = defaultdict(list)
    for i, f in enumerate(tensor_factors):
        if isinstance(f, g):
            for arg in f.args:
                if isinstance(arg, sp.Symbol):
                    idx_map[arg].append(i)
        elif isinstance(f, Mom):
            arg = f.args[1]          # second arg is the Lorentz index
            if isinstance(arg, sp.Symbol):
                idx_map[arg].append(i)
        elif isinstance(f, Eps):
            for arg in f.args:       # any Symbol slot (index or contracted momentum)
                if isinstance(arg, sp.Symbol):
                    idx_map[arg].append(i)

    scalar = sp.Mul(*scalar_part) if scalar_part else sp.Integer(1)

    for idx, fi_list in idx_map.items():
        if len(fi_list) < 2:
            continue
        i, j = fi_list[0], fi_list[1]

        # Self-trace: g(a, a) — the metric contributes 'a' twice from the same factor
        if i == j:
            rest = [f for k, f in enumerate(tensor_factors) if k != i]
            return scalar * d * (sp.Mul(*rest) if rest else sp.Integer(1))

        f1, f2 = tensor_factors[i], tensor_factors[j]
        result = _contract_pair(f1, f2, idx)
        if result is None:
            continue
        rest = [f for k, f in enumerate(tensor_factors) if k != i and k != j]
        rest_mul = sp.Mul(*rest) if rest else sp.Integer(1)
        return scalar * result * rest_mul

    return None


def _contract_pair(f1: sp.Expr, f2: sp.Expr, idx: sp.Symbol):
    """
    Contract two factors sharing Lorentz index `idx`.  Returns the replacement
    expression, or None for unrecognised combinations (e.g. Eps*Eps).
    """
    # Normalise: put g before Mom before Eps so cases below only go one way.
    if not isinstance(f1, g) and isinstance(f2, g):
        f1, f2 = f2, f1
    if isinstance(f1, Eps) and isinstance(f2, (g, Mom)):
        f1, f2 = f2, f1

    if isinstance(f1, g):
        a, b = f1.args
        other = b if a == idx else a    # the non-contracted slot of the metric

        if isinstance(f2, Mom):         # g(a,b) * Mom(p,a)  →  Mom(p,b)
            return Mom(f2.args[0], other)

        if isinstance(f2, g):           # g(a,b) * g(a,c)   →  g(b,c)
            c, e = f2.args
            other2 = e if c == idx else c
            return g(other, other2)

        if isinstance(f2, Eps):         # g(a,b) * Eps(a,…)  →  Eps(b,…)
            new_args = [other if x == idx else x for x in f2.args]
            return Eps(*new_args)

    if isinstance(f1, Mom):
        if isinstance(f2, Mom):         # Mom(p,a) * Mom(q,a)  →  Dot(p,q)
            return Dot(f1.args[0], f2.args[0])

        if isinstance(f2, Eps):         # Mom(p,a) * Eps(a,…)  →  Eps(p,…)
            p = f1.args[0]
            new_args = [p if x == idx else x for x in f2.args]
            return Eps(*new_args)

    return None   # Eps*Eps and other unhandled combinations left as-is


def simplify_external_dots(expr: sp.Expr, k: sp.Symbol) -> sp.Expr:
    """
    Rewrite Dot(a,b) for a,b != k via the polarisation identity
    a·b = ((a+b)² − a² − b²) / 2.  Leaves Dot(k,...) untouched.
    """
    subs = {}
    for d in expr.atoms(Dot):
        a, b = d.args
        if a == k or b == k:
            continue
        subs[d] = a**2 if a == b else sp.expand((a + b) ** 2 - a ** 2 - b ** 2) / 2
    return expr.subs(subs) if subs else expr
