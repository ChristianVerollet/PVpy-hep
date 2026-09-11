# pvpy/numeric.py
import numpy as np
import sympy as sp
from pvpy.base import PVFunction
from pvpy.utils import pv_symbol_key
from IPython.display import display, Markdown

def compile(expr, *, part="finite", mu = 1.0):

    expr = sp.sympify(expr)

    pv_nodes = list(expr.atoms(PVFunction))

    # --- collect kinematic symbols from PV function arguments
    # Use free_symbols so that p**2 → {p}, not missed by isinstance(a, Symbol) check
    seen = set()
    args = []
    for pv in pv_nodes:
        for a in pv.args:
            for sym in a.free_symbols:
                if sym not in seen:
                    seen.add(sym)
                    args.append(sym)

    # args = sorted(args, key=lambda s: str(s))
    args = sorted(args, key = pv_symbol_key)

    display(Markdown("**Callable signature:**"))
    display(sp.Tuple(*args))

    # --- numeric wrappers (μ captured by closure!)
    def make_wrapper(cls):
        def wrapper(*kin_vals):
            return cls._numeric_kernel(*kin_vals, mu_val = current_mu[0], part = part)
        return wrapper

    pv_numeric = {cls.__name__: make_wrapper(cls) for cls in PVFunction.__subclasses__()}

    # lambdify WITHOUT μ
    f_raw = sp.lambdify(args, expr, modules = [pv_numeric, "numpy"])

    # mutable holder for μ
    current_mu = [mu]

    # user-facing function
    def f(*kin_vals, mu_val = mu):
        current_mu[0] = mu_val
        return f_raw(*kin_vals)

    return f
