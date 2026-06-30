"""
Relation between the more complicated PV tensorial functions and the simpler scalar functions

"""

from pvpy.functions.scalar import A0, B0, B00
import sympy as sp
from ..base import PVFunction

class A00(PVFunction):

    nargs = 1 # number of **required** arguments
    latex_name = r"A_{00}"

    def __new__(cls, *args):

        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}" )
        
        return super().__new__(cls, *args)

    def _derivative(self, sym):
        return sp.S.Zero
    
    def _eval_massless(self, part = "full"):
        # Massless case

        if part == "full":
            return 0
        
        elif part == "pole":
            return 0
        
        elif part == "finite":
            return 0
  
    def _eval_massive(self, part = "full"):
        # Massive case
        m = sp.simplify( self.args[0] )

        if part == "full":
            return sp.sympify( m**2*( 1 + 1/epsilon  - sp.log(m**2 / mu**2) ) )
        
        elif part == "pole":
            return sp.sympify( m**2 )
        
        elif part == "finite":
            return sp.sympify( m**2 ) * ( 1 - sp.log(m**2 / mu**2) )
    
    def _eval(self, part = "full", **hints):
        """
        Apply kinematics to the A0 PV function
        """
        m = sp.simplify( self.args[0] )

        return sp.Piecewise( (self._eval_massless(part), m == 0 ), (self._eval_massive(part), m != 0 ) )

    def numeric(self, kernel = "numpy"):

        return self._numeric_kernel
    
    @staticmethod
    def _numeric_kernel(mv, *, mu_val, part):

        def f_m_zero(m, mu_val, part):
        
            if part == "pole":
                return 0
        
            elif part == "finite":
                return 0
        
        def f_gen(m, mu_val, part):

            if part == "pole":
                return m**2 
        
            elif part == "finite":
                return m**2 * (1 - np.log( m**2 / mu_val**2 ) )

        mv = np.asarray(mv, dtype=float)

        # allocate result
        res = np.empty_like(mv, dtype=float)

        # Check for artificial divergencies
        # mass < A_num_tol are considered 0 to avoid artificial divergencies from the log and time consuming calculations

        mask_m_zero = np.abs(mv) <= atol
        mask_m_nzero = ~mask_m_zero

        if np.any(mask_m_zero):
            res[mask_m_zero] = f_m_zero( mv[mask_m_zero], mu_val, part)

        if np.any(mask_m_nzero):
            res[mask_m_nzero] = f_gen(  mv[mask_m_nzero], mu_val, part)
        
        return res
