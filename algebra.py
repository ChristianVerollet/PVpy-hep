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
