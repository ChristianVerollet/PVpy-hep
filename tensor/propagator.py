import sympy as sp

class Propagator:

    def __init__(self, momentum, mass):

        self.momentum = momentum
        self.mass = mass

    def _repr_latex_(self):

        return rf"$({sp.latex(self.momentum)})^2 - {sp.latex(self.mass)}^2$"