import sympy as sp

# ==========================
# Lorentz index
# ==========================

class Index(sp.Symbol):
    pass


# ==========================
# Metric tensor
# ==========================

class Metric(sp.Function):

    nargs = 2

    @classmethod
    def eval(cls, mu, nu):
        return None

# ==========================
# External momentum
# ==========================

class Momentum(sp.Function):

    nargs = 2

    @classmethod
    def eval(cls, label, index):
        return None


# ==========================
# Loop momentum
# ==========================

class LoopMomentum(sp.Function):

    nargs = 1

    @classmethod
    def eval(cls, index):
        return None