# base.py

import sympy as sp
import re

class PVFunction(sp.Expr):

    # ----- Initialization ----- #
    # Ihnerit from sympy Expr

    is_commutative = True

    def __new__(cls, *args):
        args = tuple(sp.sympify(a) for a in args)
        return sp.Expr.__new__(cls, *args)

    @property
    def args(self):
        return self._args
        
    @property
    def free_symbols(self):
        s = set()
        for a in self.args:
            if isinstance(a, sp.Expr):
                s |= a.free_symbols
        return s
    
    # ----- Reduction to simpler PV functions ----- #

    def reduce(self):
        """
        Return a sympy expression for this function in terms of simpler PV functions,
        or None if this is a primitive that cannot be further reduced.

        Primitive functions (A0, B0, B00, ...) return None.
        Derived functions (A00, B1, B11, ...) override this to return their formula.

        Always call set_kinematics() before reduce_pv() so that terms whose
        coefficient vanishes under the kinematic conditions disappear before
        the reduction formulas (which can have apparent denominator singularities
        such as 1/p² in B1) are applied.
        """
        return None

    # ----- Symbolic evaluation ----- #

    def _eval(self, part = "full", **hints):
        raise NotImplementedError

    def doit(self, part = "full", **hints):

        if part not in ["full", "pole", "finite"]:
            raise ValueError("part must be 'full', 'pole' or 'finite'")
        
        return self._eval(part = part)
    
    # ----- Derivatives ----- #

    def _eval_derivative(self, sym):
        """
        Default derivative:
        ∂/∂sym PV(...) = new PVFunction object if supported,
        otherwise leave unevaluated.
        """
        if sym not in self.free_symbols:
            return sp.S.Zero

        return self._derivative(sym)
    
    # ----- Numeric evaluation ----- #

    def _numeric_kernel(self, part = "finite", **kwargs):
        raise NotImplementedError

    def numeric(self, *, kernel="numpy", part="full", **subs):
        return self._numeric_kernel(kernel=kernel, part=part, **subs)
    
    # ----- Define nice printing ------ #

    def __repr__(self):
        name = self.__class__.__name__
        args = ", ".join(map(str, self.args))
        return f"{name}({args})"

    __str__ = __repr__

    def _sympystr(self, printer):
        name = self.__class__.__name__
        args = ", ".join(printer.doprint(a) for a in self.args)
        return f"{name}({args})"

   
    def _latex(self, printer):
        fname = getattr(self, "latex_name", self.__class__.__name__)
        args = ", ".join(printer.doprint(a) for a in self.args)
        return rf"{fname}\left({args}\right)"


    def _repr_latex_(self):
        return "$$" + sp.latex(self) + "$$"
