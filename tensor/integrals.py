

import sympy as sp
from pvpy.tensor.objects import LoopMomentum


class LoopIntegral:

    def __init__(self, numerator, propagators, loop_momenta, dimension = 4):

        """
        propagators: tuple of the form ( (momentum, mass) , ... )

        """
        self.numerator = sp.sympify(numerator)
        self.propagators = propagators
        self.dimension = dimension

        # ALWAYS a tuple
        self.loop_momenta = tuple(loop_momenta)

    # ----------------------
    # Basic metadata
    # ----------------------

    @property
    def nprop(self):
        return len(self.propagators)

    @property
    def rank(self):
        return len(self.numerator.atoms(LoopMomentum))

    # ----------------------
    # Display
    # ----------------------

    def __repr__(self):

        return f"LoopIntegral(rank={self.rank}" , f"nprop={self.nprop})"
    
    def _repr_latex_(self):

        denom_terms = []

        for mom, mass in self.propagators:

            denom_terms.append( (mom**2 - mass**2)) # sympy object

        num = sp.latex(self.numerator)

        denom = " ".join(denom_terms)

        measure = " ".join([rf"\frac{{d^{self.dimension} {sp.latex(k)}}}{{(2\pi)^{self.dimension}}}" for k in self.loop_momenta])

        return rf"""$$\int {measure}\; \frac{{{num}}}{{{denom}}}$$ """