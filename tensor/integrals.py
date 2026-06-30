#################################################
################ Loopintegral ###################
#################################################


import sympy as sp
from pvpy.tensor.objects import LoopMomentum
from pvpy.functions.scalar import A0, B0, B00

# Helper, to put elsewhere after to keep this class clean
def propagator(p,m):

    return p**2 - m**2

def depends_on_loop(self, expr):

    return any(expr.has(k) for k in self.loop_momenta )

# Reducer functions

def reduce_A0(I):

    q, m = I.propagators[0]

    if not I.depends_on_loop(q):
        raise ValueError("Propagator does not depend on loop momentum.")
    
    return I.numerator * A0(m)

# Table for (N-point, rank)
REDUCTION_TABLE = {(1,0): reduce_A0 }
                            # (1,1): reduce_A1,
                            # (2,0): reduce_B0,
                            # (2,1): reduce_B1,
                            # (2,2): reduce_B2  }


    
class LoopIntegral:

    def __init__(self, numerator, propagators, loop_momenta, dimension = 4):

        """
        numerator: symbolic expression
        propagators: tuple of the form ( (momentum, mass) , ... )
        loop_momenta: symbol of the the momenta that is integrated in the loop

        """
        self.numerator = numerator
        self.propagators = propagators
        self.dimension = dimension
        self.loop_momenta = loop_momenta 
        self.N = len(propagators) # Number of points in the loop -> N points function
        self.denominator = sp.prod(propagator(p[0], p[1])for p in self.propagators)
        self.sympy_expr = self.numerator / self.denominator

    @property
    def rank(self):
        # tensorial rank of the loop
        if hasattr(self.numerator, "free_indices"):
            return len(self.numerator.free_indices)
        return 0

    @property
    def loop_rank(self):

        rank = 0

        for factor in sp.Mul.make_args(self.numerator):

            if isinstance(factor, LoopMomentum):
                rank += 1

        return rank
    
    @property
    def indices(self):
        if hasattr(self.numerator, "free_indices"):
            return self.numerator.free_indices
        return {}
    
    @property
    def topology(self):
        return (self.N, self.rank)
    
    def tensorial_reduction(self):

        key = (self.N, self.rank)

        if key not in REDUCTION_TABLE:
            raise NotImplementedError(
                f"Reduction {key} not implemented."
            )

        return REDUCTION_TABLE[key](self)