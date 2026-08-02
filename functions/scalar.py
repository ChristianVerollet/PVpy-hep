###############################################################
############### Definition of PV functions ####################
###############################################################

import sympy as sp
import numpy as np
from ..base import PVFunction
from ..symbols import epsilon_bar, mu

atol = 1e-8 # Used for the tolerance when p² ~ 0 and for A0 where 1 mass argument is involved
scale_tol = 1e-4 # Scale tolerance for all other functions to use the expanded function to avoid brutal shift in the values

# ------- Define PV scalar functions ------- #

class A0(PVFunction):

    nargs = 1 # number of **required** arguments
    latex_name = r"A_0"

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
            return sp.sympify( m**2*( 1 + 1/epsilon_bar  - sp.log(m**2 / mu**2) ) )
        
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

class B0(PVFunction):

    nargs = 3
    latex_name = r"B_0"

    def __new__(cls, *args):
        """
        Optional mu: if mu_arg is None, use default mu symbol.
        """
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}" )

        return super().__new__(cls, *args)
    
    def _derivative(self, sym):
        p2, m1, m2 = self.args

        if sym == p2:
            return dB0_dp2(p2, m1, m2)

        else:
            raise NotImplementedError
        
    def _eval_p2_zero_m2_zero(self, part = "full"):
        # Special case: p2 = 0, m2 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return 1 + 1/epsilon_bar - sp.log(m1**2 / mu**2)
        
        elif part == "pole":
            return 1
        
        elif part == "finite":
            return 1 - sp.log(m1**2 / mu**2)

    def _eval_p2_zero_m1_zero(self, part = "full"):
        # Special case: p2 = 0, m2 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return 1 + 1/epsilon_bar - sp.log(m2**2 / mu**2)
        
        elif part == "pole":
            return 1
        
        elif part == "finite":
            return 1 - sp.log(m2**2 / mu**2)
             
    def _eval_p2_zero_m1_eq_m2(self, part = "full"):
        # Special case: p2 = 0, m1 = m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
        
        if part == "full":
            return 1/epsilon_bar - sp.log(m2**2 / mu**2)
        
        elif part == "pole":
            return 1
        
        elif part == "finite":
            return - sp.log(m2**2 / mu**2)
        
    def _eval_p2_zero(self, part = "full"):
        # Special case: p2 = 0, m1 != m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        return  ( A0(m1).doit(part) - A0(m2).doit(part) ) / (m1**2 - m2**2) 
    
    def _eval_m1_eq_m2(self, part = "full"):
        # Special case: m1 = m2, p2 != 0
        p2, m1, m2 = map(sp.simplify, self.args)

        Lambda = p2 * sp.sqrt( 1 - 4 * m1**2/p2 )
        R = - (Lambda/p2) * sp.log( (2 * m1**2 - p2 + Lambda)/(2 * m1 ** 2) )

        if part == "full":
            return 1/epsilon_bar - sp.log(m2**2 / mu**2) + 2 - R
        
        elif part == "pole":
            return 1
        
        elif part == "finite":
            return 2 - R - sp.log(m2**2 / mu**2)
    
    def _eval_general(self, part = "full"):
        # General case: p2 != 0, m1 != m2, m1 != 0, m2 != 0
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        Lambda = sp.sqrt( m1**4 + m2**4 + p2**2 + 2 * (- m1**2 * p2 - m2**2 * p2 - m1**2 * m2**2 ) )
        R = - (Lambda/p2) * sp.log( (m1**2 + m2**2 - p2 + Lambda)/(2 * m1 * m2) )

        if part == "full":
            return 1/epsilon_bar - sp.log(m2**2 / mu**2) + 2 + ( (m2**2 - m1**2 + p2)/(2 * p2) ) * sp.log( m1**2/m2**2) - R
        
        elif part == "pole":
            return 1 
        
        elif part == "finite":
            return 2 + ( (m2**2 - m1**2 + p2)/(2 * p2) ) * sp.log( m1**2/m2**2) - R - sp.log(m2**2 / mu**2)
 
    def _eval(self, part = "full", **hints):
        """
        Apply kinematics to the B0 function
        """

        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        return sp.Piecewise   ( (self._eval_p2_zero_m1_zero(part), p2 == 0 and m1 == 0 ),
                                (self._eval_p2_zero_m2_zero(part), p2 == 0 and m2 == 0 ),
                                (self._eval_p2_zero_m1_eq_m2(part), p2 == 0 and m1 == m2 ),
                                (self._eval_p2_zero(part),  p2 ==0 and m1 != m2),
                                (self._eval_m1_eq_m2(part), p2 != 0 and m1 == m2),
                                (self._eval_general(part), True )   )
    
    def numeric(self):

        return self._numeric_kernel
    
    @staticmethod
    def _numeric_kernel(p2v, m1v, m2v, mu_val, part):
            
        # if not ( np.ndim(p2v) == np.ndim(m1v) == np.ndim(m2v) ):
        #     raise ValueError("Input arrays must have the same number of dimensions.")
        
        # ------ functions for non-zero p² ----------- #

        def f_zero(p2, m1, m2, mu_val, part):
            # p2 = 0, m1 = 0, m2 = 0
            if part == "pole":
                return  1
        
            elif part == "finite":
                return  0
            
        def f_general(p2, m1, m2, mu_val, part):

            Lambda = np.sqrt( m1**4 + m2**4 + p2**2 + 2 * (- m1**2 * p2 - m2**2 * p2 - m1**2 * m2**2 ) )

            R = - (Lambda/p2) * np.log( (m1**2 + m2**2 - p2 + Lambda)/(2 * m1 * m2) )
            
            if part == "pole":
                return 1
            
            elif part == "finite":
                return - np.log(m2**2 / mu_val**2) + 2 + ( (m1**2 - m2**2 + p2)/(2 * p2) ) * np.log( m1**2/m2**2) - R
            
        def f_m_eq(p2, m1, m2, mu_val, part): # to update

            Lambda = p2 * np.sqrt( 1 - 4 * m1**2/p2 )
            R = - (Lambda/p2) * np.log( (2 * m1**2 - p2 + Lambda)/(2 * m1 ** 2) )

            if part == "pole":
                return 1
            
            elif part == "finite":
                return - np.log(m2**2 / mu_val**2) + 2 - R

        # ---------- functions for p² = 0 ----------- #

        def f_p2_zero_m2_zero(p2, m1, m2, mu_val, part):
        
            if part == "pole":
                return  1
        
            elif part == "finite":
                return  1 - np.log( m1**2 / mu_val**2 )
            
        def f_p2_zero_m1_zero(p2, m1, m2, mu_val, part):
        
            if part == "pole":
                return  1
        
            elif part == "finite":
                return  1 - np.log( m2**2 / mu_val**2 )
            
        def f_p2_zero_m_non_zero(p2, m1, m2, mu_val, part):
            
            scale = np.maximum(m1**2, m2**2)
            r = np.abs(m1**2 - m2**2) / scale
            
            def F_general(m1, m2, mu_val):

                return 1 + ( m2**2 * np.log( m2**2 / mu_val**2 ) - m1**2 * np.log( m1**2 / mu_val**2) ) / (m1**2 - m2**2)

            def F_expanded(m1, m2, mu_val):

                m_sq = 0.5 * (m1**2 + m2**2) # Caughtion mass is already squared!
                Delta  = 0.5 * (m1**2 - m2**2)

                return - np.log(m_sq / mu_val**2) + Delta**2 / (6 * m_sq**2) + Delta**4 / (20 * m_sq **4)
            
            mask = r < scale_tol
            result_finite = np.empty_like(m1)
            result_finite[mask]  = F_expanded(m1[mask], m2[mask], mu_val)
            result_finite[~mask] = F_general(m1[~mask], m2[~mask], mu_val)

            if part == "pole":
                return  1
        
            elif part == "finite":
                return  result_finite
            
        p2v = np.asarray(p2v, dtype=float)
        m1v = np.asarray(m1v, dtype=float)
        m2v = np.asarray(m2v, dtype=float)

        # allocate result
        res = np.empty_like(p2v, dtype=float)

        # Masks for p2 ≈ 0
        mask_p2_zero = np.abs(p2v) <= atol
        mask_m1_zero = np.abs(m1v) <= atol
        mask_m2_zero = np.abs(m2v) <= atol

        # Masks for p2 ≈ 0
        mask_p2_zero_m_zero   = mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_zero_m1_zero  = mask_p2_zero & mask_m1_zero & ~mask_m2_zero
        mask_p2_zero_m2_zero  = mask_p2_zero & ~mask_m1_zero & mask_m2_zero
        mask_p2_zero_m_non_zero    = mask_p2_zero &  ~mask_m1_zero & ~mask_m2_zero

        # Masks for p2 ≠ 0
        mask_p2_nzero_m_zero   = ~mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_nzero_m1_zero  = ~mask_p2_zero & mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m2_zero  = ~mask_p2_zero & ~mask_m1_zero & mask_m2_zero # Need to implement the associated numpy function
        mask_p2_non_zero_m_non_zero     = ~mask_p2_zero & ~mask_m1_zero & ~mask_m2_zero


        if np.any(mask_p2_zero_m_zero):
            res[mask_p2_zero_m_zero] = f_zero( p2v[mask_p2_zero_m_zero], m1v[mask_p2_zero_m_zero], m2v[mask_p2_zero_m_zero], mu_val, part)

        if np.any(mask_p2_zero_m_non_zero):
            res[mask_p2_zero_m_non_zero] = f_p2_zero_m_non_zero( p2v[mask_p2_zero_m_non_zero], m1v[mask_p2_zero_m_non_zero], m2v[mask_p2_zero_m_non_zero], mu_val, part)

        if np.any(mask_p2_zero_m1_zero):
            res[mask_p2_zero_m1_zero] = f_p2_zero_m1_zero( p2v[mask_p2_zero_m1_zero], m1v[mask_p2_zero_m1_zero], m2v[mask_p2_zero_m1_zero], mu_val, part)

        if np.any(mask_p2_zero_m2_zero):
            res[mask_p2_zero_m2_zero] = f_p2_zero_m2_zero( p2v[mask_p2_zero_m2_zero], m1v[mask_p2_zero_m2_zero], m2v[mask_p2_zero_m2_zero], mu_val, part)
        
        if np.any(mask_p2_nzero_m_zero):
            res[mask_p2_nzero_m_zero] = f_zero( p2v[mask_p2_nzero_m_zero], m1v[mask_p2_nzero_m_zero], m2v[mask_p2_nzero_m_zero], mu_val, part)

        # if np.any(mask_p2_nzero_m_eq):
        #     res[mask_p2_nzero_m_eq] = f_m_eq( p2v[mask_p2_nzero_m_eq], m1v[mask_p2_nzero_m_eq], m2v[mask_p2_nzero_m_eq], mu_val, part)

        # if np.any(mask_p2_nzero_m_non_zero):
        #     res[mask_p2_nzero_m_non_zero] = f_general( p2v[mask_p2_nzero_m_non_zero], m1v[mask_p2_nzero_m_non_zero], m2v[mask_p2_nzero_m_non_zero], mu_val, part)

        return res
   
class B00(PVFunction):

    nargs = 3
    latex_name = r"B_{00}"

    def __new__(cls, *args, mu_arg = None):
        """
        Optional mu: if mu_arg is None, use default mu symbol.
        """
        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}" )
        
        return super().__new__(cls, *args)

    def _derivative(self, sym):
        p2, m1, m2 = self.args

        if sym == p2:
            return dB00_dp2(p2, m1, m2)

        else:
            raise NotImplementedError
    
    def _eval_o2_zero_m1_zero(self, part = "full"):
        # Special case: p2 = 0, m1 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return sp.sympify( sp.Rational(1,4) * m2**2 * ( sp.Rational(3,2) + 1/epsilon_bar - sp.log( m2**2 / mu**2 ) ) )
        
        elif part == "pole":
            return sp.sympify( sp.Rational(1,4) * m2**2  )
        
        elif part == "finite":
            return sp.sympify( sp.Rational(1,4) * m2**2 * ( sp.Rational(3,2) - sp.log( m2**2 / mu**2 ) ) )
        
    def _eval_p2_zero_m2_zero(self, part = "full"):
        # Special case: p2 = 0, m2 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return sp.sympify( sp.Rational(1,4) * m1**2 * ( sp.Rational(3,2) + 1/epsilon_bar - sp.log( m1**2 / mu**2 ) ) )
        
        elif part == "pole":
            return sp.sympify( sp.Rational(1,4) * m1**2  )
        
        elif part == "finite":
            return sp.sympify( sp.Rational(1,4) * m1**2 * ( sp.Rational(3,2) - sp.log( m1**2 / mu**2 ) ) )
        
    def _eval_p2_zero_m1_eq_m2(self, part = "full"):
        # Special case: p2 = 0, m1 = m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
        
        if part == "full":
            return sp.sympify( sp.Rational(1,2) * m1**2 * (1 + 1/epsilon_bar - sp.log( m2**2 / mu**2 ) ) )
        
        elif part == "pole":
            return sp.sympify( sp.Rational(1,2) * m1**2  )
        
        elif part == "finite":
            return sp.sympify( sp.Rational(1,2) * m1**2 * (1 - sp.log( m2**2 / mu**2 ) ) )
        
    def _eval_p2_zero(self, part = "full"):
        # Special case: p2 = 0, m1 != m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
        
        if part == "full":
            return sp.sympify(  sp.Rational(1,4) * ( m1**2 + m2**2 ) * ( sp.Rational(3,2) + 1/epsilon_bar - sp.log( m2**2 / mu**2 ) )
                            +  ( m1**4 / (4 * (m1**2 - m2**2))) * sp.log( m1**2 / m2**2 ) , rational=True )
        elif part == "pole":
            return  sp.sympify(  sp.Rational(1,4) * ( m1**2 + m2**2 )  , rational=True )
        
        elif part == "finite":
            return sp.sympify(  sp.Rational(3,8) * ( m1**2 + m2**2 ) +  ( m1**4 / (4 * (m1**2 - m2**2))) * sp.log( m1**2 / m2**2 ) , rational=True )
    
    def _eval_m1_eq_m2(self, part = "full"):
        # Special case: m1 = m2, p2 != 0
        
        return 10 # To be implement
    
    def _eval_general(self, part = "full"):
        # General case: p2 != 0, m1 != m2
        
        return 20 # To be implement

    def _eval(self, part = "full", **hints):
        """
        Apply kinematics to the B00 function and return the right symbolic analytic expression
        """

        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
    
        return sp.Piecewise   ( (self._eval_o2_zero_m1_zero(part), p2 == 0 and m1 == 0 ),
                                (self._eval_p2_zero_m2_zero(part), p2 == 0 and m2 == 0 ),
                                (self._eval_p2_zero_m1_eq_m2(part), p2 == 0 and m1 == m2 ),
                                (self._eval_p2_zero(part),  p2 ==0 and m1 != m2),
                                (self._eval_m1_eq_m2(part), p2 != 0 and m1 == m2),
                                (self._eval_general(part), True )   )
    
    def numeric(self):

        return self._numeric_kernel
    
    @staticmethod
    def _numeric_kernel(p2v, m1v, m2v, mu_val, part):
            
        if not ( np.ndim(p2v) == np.ndim(m1v) == np.ndim(m2v) ):
            raise ValueError("Input arrays must have the same number of dimensions.")
        
        # --------- Functions when p² != 0 ------------#

        def f_general(p2, m1, m2, mu_val, part):
            return 20 # To implement
    
        def f_m_eq(p2, m1, m2, mu_val, part):
            return 10 # To implement

        def f_zero(p2, m1, m2, mu_val, part):
            # p2 = 0, m1 = 0, m2 = 0
            if part == "pole":
                return  0
        
            elif part == "finite":
                return  0
        
        # --------- Functions when p² = 0 -------------#

        def f_p2_zero_m1_zero(p2, m1, m2, mu_val, part):

            if part == "pole":
                return  0.25 * m2**2 
        
            elif part == "finite":
                return 0.25 * m2**2 * ( 3/2 - np.log( m2**2 / mu_val**2 ) )
            
        def f_p2_zero_m2_zero(p2, m1, m2, mu_val, part):

            if part == "pole":
                return  0.25 * m1**2 
        
            elif part == "finite":
                return 0.25 * m1**2 * ( 3/2 - np.log( m1**2 / mu_val**2 ) )
            
        def f_p2_zero_m_non_zero(p2, m1, m2, mu_val, part):
            
            scale = np.maximum(m1**2, m2**2)
            r = np.abs(m1**2 - m2**2) / scale
            
            def F_general(m1, m2, mu_val):

                return (1/4) * ( m1**2 + m2**2 ) * ( 3/2 - np.log(m1 * m2/ mu_val**2) ) -  ( (m1**4 + m2**4) / (8 * (m1**2 - m2**2) ) ) * np.log( m1**2 / m2**2 )

            def F_expanded(m1, m2, mu_val):

                m_sq = 0.5 * (m1**2 + m2**2) # Caughtion mass is squared!
                Delta  = 0.5 * (m1**2 - m2**2)

                return  0.5 * m_sq *( 1 -  np.log(m_sq / mu_val**2) ) - Delta**2 / (12 * m_sq) - Delta**4 / (120 * m_sq**3)
            
            mask = r < scale_tol
            result_finite = np.empty_like(m1)
            result_finite[mask]  = F_expanded(m1[mask], m2[mask], mu_val)
            result_finite[~mask] = F_general(m1[~mask], m2[~mask], mu_val)

            if part == "pole":
                return 0.25 * ( m1**2 + m2**2 ) 
        
            elif part == "finite":
                return  result_finite
            
        p2v = np.asarray(p2v, dtype=float)
        m1v = np.asarray(m1v, dtype=float)
        m2v = np.asarray(m2v, dtype=float)

        # Allocate result
        res = np.empty_like(p2v, dtype=float)

        # Masks for p2 ≈ 0
        mask_p2_zero = np.abs(p2v) <= atol
        mask_m1_zero = np.abs(m1v) <= atol
        mask_m2_zero = np.abs(m2v) <= atol

        # Masks for p2 ≈ 0
        mask_p2_zero_m_zero   = mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_zero_m1_zero  = mask_p2_zero & mask_m1_zero & ~mask_m2_zero
        mask_p2_zero_m2_zero  = mask_p2_zero & ~mask_m1_zero & mask_m2_zero
        mask_p2_zero_m_non_zero     = mask_p2_zero & ~mask_m1_zero & ~mask_m2_zero

        # Masks for p2 ≠ 0
        mask_p2_nzero_m_zero   = ~mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_nzero_m1_zero  = ~mask_p2_zero & mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m2_zero  = ~mask_p2_zero & ~mask_m1_zero & mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m_non_zero     = ~mask_p2_zero & ~mask_m1_zero & ~mask_m2_zero


        if np.any(mask_p2_zero_m_zero):
            res[mask_p2_zero_m_zero] = f_zero( p2v[mask_p2_zero_m_zero], m1v[mask_p2_zero_m_zero], m2v[mask_p2_zero_m_zero], mu_val, part)

        if np.any(mask_p2_zero_m_non_zero):
            res[mask_p2_zero_m_non_zero] = f_p2_zero_m_non_zero( p2v[mask_p2_zero_m_non_zero], m1v[mask_p2_zero_m_non_zero], m2v[mask_p2_zero_m_non_zero], mu_val, part)

        if np.any(mask_p2_zero_m1_zero):
            res[mask_p2_zero_m1_zero] = f_p2_zero_m1_zero( p2v[mask_p2_zero_m1_zero], m1v[mask_p2_zero_m1_zero], m2v[mask_p2_zero_m1_zero], mu_val, part)

        if np.any(mask_p2_zero_m2_zero):
            res[mask_p2_zero_m2_zero] = f_p2_zero_m2_zero( p2v[mask_p2_zero_m2_zero], m1v[mask_p2_zero_m2_zero], m2v[mask_p2_zero_m2_zero], mu_val, part)
        
        if np.any(mask_p2_nzero_m_zero):
            res[mask_p2_nzero_m_zero] = f_zero( p2v[mask_p2_nzero_m_zero], m1v[mask_p2_nzero_m_zero], m2v[mask_p2_nzero_m_zero], mu_val, part)

        return res
    
# ------- Define derivative of PV scalar functions ------- #

class dB0_dp2(PVFunction):

    nargs = 3
    latex_name = r"B_{0}"   # base name only

    def __new__(cls, *args):

        if len(args) != cls.nargs:
            raise TypeError(
                f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}" )
        
        return super().__new__(cls, *args)
    
    def _latex(self, printer):
        p2, m1, m2 = self.args
        args = ", ".join(printer.doprint(a) for a in self.args)
        return rf"\frac{{\partial B_{{0}}}}{{\partial p^2}}\left({args}\right)"

    def _derivative(self, sym):
        p2, m1, m2 = self.args

        raise NotImplementedError
        
    def _eval_p2_zero_m1_zero(self, part = "full"):
        # Special case: p2 = 0, m1 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return sp.Rational(1/2) * 1/m2**2
        
        elif part == "pole":
            return 0
        
        elif part == "finite":
            return sp.Rational(1/2) * 1/m2**2
        
    def _eval_p2_zero_m2_zero(self, part = "full"):
        # Special case: p2 = 0, m2 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return sp.Rational(1/2) * 1/m1**2
        
        elif part == "pole":
            return 0
        
        elif part == "finite":
            return sp.Rational(1/2) * 1/m1**2
        
    def _eval_p2_zero_m1_eq_m2(self, part = "full"):
        # Special case: p2 = 0, m1 = m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
        
        if part == "full":
            return  sp.Rational(1/6) * 1/m1**2,
        
        elif part == "pole":
            return 0
        
        elif part == "finite":
            return  sp.Rational(1/6) * 1/m1**2,
        
    def _eval_p2_zero(self, part = "full"):
        # Special case: p2 = 0, m1 != m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        if part == "full":
            return sp.sympify(  sp.Rational(1,2) * 1 / (m1**2 - m2**2)**3 ) * ( m1**4 - m2**4 - 2 * m1**2 * m2**2 * sp.log( m1**2 / m2**2 )  )
        
        elif part == "pole":
            return 0
        
        elif part == "finite":
            return  sp.sympify(  sp.Rational(1,2) * 1 / (m1**2 - m2**2)**3 ) * ( m1**4 - m2**4 - 2 * m1**2 * m2**2 * sp.log( m1**2 / m2**2 )  )
          
    def _eval_m1_eq_m2(self, part = "full"):
        # Special case: m1 = m2, p2 != 0
        
        return 10
    
    def _eval_general(self, part = "full"):
        # General case: p2 != 0, m1 != m2
        
        return 20
    
    def _eval(self, part = "full", **hints):
        """
        Apply kinematics to the B0 function
        """

        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        return sp.Piecewise   ( (self._eval_p2_zero_m1_eq_m2(part), p2 == 0 and m1 == m2 ),
                                (self._eval_p2_zero_m1_zero(part), p2 == 0 and m1 == 0 ),
                                (self._eval_p2_zero_m2_zero(part), p2 == 0 and m2 == 0 ),
                                (self._eval_p2_zero(part),  p2 ==0 and m1 != m2),
                                (self._eval_m1_eq_m2(part), p2 != 0 and m1 == m2),
                                (self._eval_general(part), True )   )
    
    def numeric(self):

        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m1v, m2v, mu_val, part):

        if not ( np.ndim(p2v) == np.ndim(m1v) == np.ndim(m2v) ):
            raise ValueError("Input arrays must have the same number of dimensions.")

        # ------- Functions when p² != 0 ----------#

        def f_m_eq(p2, m1, m2, mu_val, part):
            # Special case: m1 = m2, p2 != 0
            
            raise NotImplementedError
        
        def f_general(p2, m1, m2, mu_val, part):
            # General case: p2 != 0, m1 != m2
            
            raise NotImplementedError 
        
        # ------- Functions when p² = 0 -----------#

        def f_zero(p2, m1, m2, mu_val, part):
            # p2 = 0, m1 = 0, m2 = 0
            if part == "pole":
                return  0
        
            elif part == "finite":
                return  0
              
        def f_p2_zero_m1_zero(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m1 = 0
            
            if part == "pole":
                return 0
            
            elif part == "finite":
                return 0.5 / m2**2
            
        def f_p2_zero_m2_zero(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m2 = 0
            
            if part == "pole":
                return 0
            
            elif part == "finite":
                return 0.5 / m1**2
            
        def f_p2_zero_m_non_zero(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m1 != 0 and m2 != 0
            
            scale = np.maximum(m1**2, m2**2)
            r = np.abs(m1**2 - m2**2) / scale
            
            def F_general(m1, m2, mu_val):

                return  0.5 * ( 1 / (m1**2 - m2**2)**3 ) * ( m1**4 - m2**4 - 2 * m1**2 * m2**2 * np.log( m1**2 / m2**2 )  )
            
            def F_expanded(m1, m2, mu_val):

                m_sq = 0.5 * (m1**2 + m2**2) # Caughtion mass is squared!
                Delta  = 0.5 * (m1**2 - m2**2)

                return 1/(6 * m_sq) + Delta**2 / (30 * m_sq**3) + Delta**4 / (70 * m_sq **5)
            
            mask = r < scale_tol
            result_finite = np.empty_like(m1)
            result_finite[mask]  = F_expanded(m1[mask], m2[mask], mu_val)
            result_finite[~mask] = F_general(m1[~mask], m2[~mask], mu_val)

            if part == "pole":
                return 0
            
            elif part == "finite":
                return result_finite
            
        p2v = np.asarray(p2v, dtype=float)
        m1v = np.asarray(m1v, dtype=float)
        m2v = np.asarray(m2v, dtype=float)

        # allocate result
        res = np.empty_like(p2v, dtype=float)

        # Masks for p2 ≈ 0
        mask_p2_zero = np.abs(p2v) <= atol
        mask_m1_zero = np.abs(m1v) <= atol
        mask_m2_zero = np.abs(m2v) <= atol

        # Masks for p2 ≈ 0
        mask_p2_zero_m_zero   = mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_zero_m1_zero  = mask_p2_zero & mask_m1_zero & ~mask_m2_zero
        mask_p2_zero_m2_zero  = mask_p2_zero & ~mask_m1_zero & mask_m2_zero
        mask_p2_zero_m_non_zero     = mask_p2_zero & ~mask_m1_zero & ~mask_m2_zero

        # Masks for p2 ≠ 0
        mask_p2_nzero_m_zero   = ~mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_nzero_m1_zero  = ~mask_p2_zero & mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m2_zero  = ~mask_p2_zero & ~mask_m1_zero & mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m_non_zero    = ~mask_p2_zero & ~mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function
  

        if np.any(mask_p2_zero_m_zero):
            res[mask_p2_zero_m_zero] = f_zero( p2v[mask_p2_zero_m_zero], m1v[mask_p2_zero_m_zero], m2v[mask_p2_zero_m_zero], mu_val, part)

        if np.any(mask_p2_zero_m_non_zero):
            res[mask_p2_zero_m_non_zero] = f_p2_zero_m_non_zero( p2v[mask_p2_zero_m_non_zero], m1v[mask_p2_zero_m_non_zero], m2v[mask_p2_zero_m_non_zero], mu_val, part)

        if np.any(mask_p2_zero_m1_zero):
            res[mask_p2_zero_m1_zero] = f_p2_zero_m1_zero( p2v[mask_p2_zero_m1_zero], m1v[mask_p2_zero_m1_zero], m2v[mask_p2_zero_m1_zero], mu_val, part)

        if np.any(mask_p2_zero_m2_zero):
            res[mask_p2_zero_m2_zero] = f_p2_zero_m2_zero( p2v[mask_p2_zero_m2_zero], m1v[mask_p2_zero_m2_zero], m2v[mask_p2_zero_m2_zero], mu_val, part)
        
        if np.any(mask_p2_nzero_m_zero):
            res[mask_p2_nzero_m_zero] = f_zero( p2v[mask_p2_nzero_m_zero], m1v[mask_p2_nzero_m_zero], m2v[mask_p2_nzero_m_zero], mu_val, part)

        if np.any(mask_p2_nzero_m_non_zero):
            res[mask_p2_nzero_m_non_zero] = f_general( p2v[mask_p2_nzero_m_non_zero], m1v[mask_p2_nzero_m_non_zero], m2v[mask_p2_nzero_m_non_zero], mu_val, part)

        return res
    
class dB00_dp2(PVFunction):
    
    nargs = 3
    latex_name = r"B_{00}"   # base name only

    def __new__(cls, *args):

        if len(args) != cls.nargs:
            raise TypeError(f"{cls.__name__} expects {cls.nargs} arguments, got {len(args)}" )
        
        return super().__new__(cls, *args)
    
    def _latex(self, printer):
        args = ", ".join(printer.doprint(a) for a in self.args)
        return rf"\frac{{\partial {self.latex_name}}}{{\partial p^2}}\left({args}\right)"

    def _derivative(self, sym):
        p2, m1, m2 = self.args

        raise NotImplementedError
    
    def _eval_p2_zero_m1_zero(self, part = "full"):
        # Special case: p2 = 0, m1 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return - sp.Rational(1,12) * ( 1/epsilon_bar - sp.log( m2**2 / mu**2 ) ) - sp.Rational(5,72)
        
        elif part == "pole":
            return  - sp.Rational(1,12) 
        
        elif part == "finite":
            return  sp.Rational(1,12) * sp.log( m2**2 / mu**2 ) - sp.Rational(5,72)
        
    def _eval_p2_zero_m2_zero(self, part = "full"):
        # Special case: p2 = 0, m2 = 0
        p2, m1, m2 = map(sp.simplify, self.args) 
        
        if part == "full":
            return - sp.Rational(1,12) * ( 1/epsilon_bar - sp.log( m1**2 / mu**2 ) ) - sp.Rational(5,72)
        
        elif part == "pole":
            return  - sp.Rational(1,12) 
        
        elif part == "finite":
            return  sp.Rational(1,12) * sp.log( m1**2 / mu**2 ) - sp.Rational(5,72)
               
    def _eval_p2_zero_m1_eq_m2(self, part = "full"):
        # Special case: p2 = 0, m1 = m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity
        
        if part == "full":
            return - sp.Rational(1,12) * ( 1/epsilon_bar - sp.log( m2**2 / mu**2 ) ) 
        
        elif part == "pole":
            return  - sp.Rational(1,12) 
        elif part == "finite":
            return  sp.Rational(1,12) * sp.log( m2**2 / mu**2 )
        
    def _eval_p2_zero(self, part = "full"):
        # Special case: p2 = 0, m1 != m2
        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        if part == "full":
            return ( - sp.Rational(1,12) * ( 1/epsilon_bar - sp.log( m1*m2 / mu**2 ) ) 
                    - sp.Rational(1,72) * (  - 3 * (m1**6 - 3*m1**2*m2**4 - 3*m1**4*m2**2 + m2**6)/(m1**2 - m2**2)**3 * sp.log(m1**2 / m2**2) 
                                          + (5*m1**4 - 22*m1**2*m2**2 + 5*m2**4)/(m1**2 - m2**2)**2 ) )
    
        elif part == "pole":
            return - sp.Rational(1,12) 
        
        elif part == "finite":
            return  ( - sp.Rational(1,12) * ( - sp.log( m1*m2 / mu**2 ) ) 
                    - sp.Rational(1,72) * (  - 3 * (m1**6 - 3*m1**2*m2**4 - 3*m1**4*m2**2 + m2**6)/(m1**2 - m2**2)**3 * sp.log(m1**2 / m2**2) 
                                          + (5*m1**4 - 22*m1**2*m2**2 + 5*m2**4)/(m1**2 - m2**2)**2 ) )
    
    def _eval_m1_eq_m2(self, part = "full"):
        # Special case: m1 = m2, p2 != 0
        
        return 10 # To implemente
    
    def _eval_general(self, part = "full"):
        # General case: p2 != 0, m1 != m2
        
        return 20 # To implemente
    
    def _eval(self, part = "full", **hints):
        """
        Apply kinematics to the B0 function
        """

        p2, m1, m2 = map(sp.simplify, self.args) # rename for clarity

        return sp.Piecewise   ( (self._eval_p2_zero_m1_zero(part), p2 == 0 and m1 == 0 ),
                                (self._eval_p2_zero_m2_zero(part), p2 == 0 and m2 == 0 ),
                                (self._eval_p2_zero_m1_eq_m2(part), p2 == 0 and m1 == m2 ),
                                (self._eval_p2_zero(part),  p2 ==0 and m1 != m2),
                                (self._eval_m1_eq_m2(part), p2 != 0 and m1 == m2),
                                (self._eval_general(part), True )   )
    
    def numeric(self):

        return self._numeric_kernel

    @staticmethod
    def _numeric_kernel(p2v, m1v, m2v, mu_val, part):
        
        if not ( np.ndim(p2v) == np.ndim(m1v) == np.ndim(m2v) ):
            raise ValueError("Input arrays must have the same number of dimensions.")
        
        # ------------ Functions when p² != 0 -------------- #
        # to be completed

        def f_m_eq(p2, m1, m2, mu_val, part ):
            # Special case: m1 = m2, p2 != 0
            
            raise "To implement" # To implement
        
        def f_general(p2, m1, m2, mu_val, part):
            # General case: p2 != 0, m1 != m2
            
            return "To implement" # To implement
        
        # ------------ Functions when p² ~ 0 --------------- #

        def f_zero(p2, m1, m2, mu_val, part):
            # p2 = 0, m1 = 0, m2 = 0
            if part == "pole":
                return  - 1/12
        
            elif part == "finite":
                return  0

        def f_p2_zero_m_eq(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m1 = m2

            if part == "pole":
                return - 1/12 
            
            elif part == "finite":
                return  1/12 *  np.log( m2**2 / mu_val**2 ) 
        
        def f_p2_zero_m1_zero(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m1 = 0
            
            if part == "pole":
                return - 1/12
            
            elif part == "finite":
                return  ( - 1/12 * ( - np.log( m2**2 / mu_val**2 ) ) - (5/72) )
            
        def f_p2_zero_m2_zero(p2, m1, m2, mu_val, part):
            # Special case: p2 = 0, m2 = 0
            
            if part == "pole":
                return - 1/12
            
            elif part == "finite":
                return  ( - 1/12 * ( - np.log( m1**2 / mu_val**2 ) ) - (5/72) )
        
        def f_p2_zero_m_non_zero(p2, m1 , m2, mu_val, part):
            # Special case:, p2 = 0, m1 != m2, m1 =! 0, m2 =! 0,

            scale = np.maximum(m1**2, m2**2)
            r = np.abs(m1**2 - m2**2) / scale
            
            def F_general(m1, m2, mu_val):

                return  ( - 1/12 * ( - np.log( m1*m2 / mu_val**2 ) ) - (1/72) * ( - 3 * (m1**6 - 3*m1**2*m2**4 - 3*m1**4*m2**2 + m2**6)/(m1**2 - m2**2)**3 * np.log(m1**2 / m2**2) 
                                          + (5*m1**4 - 22*m1**2*m2**2 + 5*m2**4)/(m1**2 - m2**2)**2 ) )
            
            def F_expanded(m1, m2, mu_val):

                m_sq = 0.5 * (m1**2 + m2**2) # Caughtion mass is squared!
                Delta  = 0.5 * (m1**2 - m2**2)

                return  1/12 *  np.log( m_sq / mu_val**2 ) - Delta**2/( 120 * m_sq**2) - Delta**4/ (560 * m_sq**4)
            
            mask = r < scale_tol
            result_finite = np.empty_like(m1)
            result_finite[mask]  = F_expanded(m1[mask], m2[mask], mu_val)
            result_finite[~mask] = F_general(m1[~mask], m2[~mask], mu_val)

            if part == "pole":
                return - 1/12
            
            elif part == "finite":
                return  result_finite
        

        p2v = np.asarray(p2v, dtype=float)
        m1v = np.asarray(m1v, dtype=float)
        m2v = np.asarray(m2v, dtype=float)

        # allocate result
        res = np.empty_like(p2v, dtype=float)

      
        # Masks for p2 ≈ 0
        mask_p2_zero = np.abs(p2v) <= atol
        mask_m1_zero = np.abs(m1v) <= atol
        mask_m2_zero = np.abs(m2v) <= atol

        # Masks for p2 ≈ 0
        mask_p2_zero_m_zero   = mask_p2_zero & mask_m1_zero & mask_m2_zero
        mask_p2_zero_m1_zero  = mask_p2_zero & mask_m1_zero & ~mask_m2_zero
        mask_p2_zero_m2_zero  = mask_p2_zero & ~mask_m1_zero & mask_m2_zero
        mask_p2_zero_m_non_zero     = mask_p2_zero &  ~mask_m1_zero & ~mask_m2_zero

        # Masks for p2 ≠ 0
        mask_p2_nzero_m_zero   = ~mask_p2_zero & mask_m1_zero & mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m1_zero  = ~mask_p2_zero & mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m2_zero  = ~mask_p2_zero & ~mask_m1_zero & mask_m2_zero # Need to implement the associated numpy function
        mask_p2_nzero_m_non_zero    = ~mask_p2_zero  & ~mask_m1_zero & ~mask_m2_zero # Need to implement the associated numpy function

        if np.any(mask_p2_zero_m_zero):
            res[mask_p2_zero_m_zero] = f_zero( p2v[mask_p2_zero_m_zero], m1v[mask_p2_zero_m_zero], m2v[mask_p2_zero_m_zero], mu_val, part)

        if np.any(mask_p2_zero_m_non_zero):
            res[mask_p2_zero_m_non_zero] = f_p2_zero_m_eq( p2v[mask_p2_zero_m_non_zero], m1v[mask_p2_zero_m_non_zero], m2v[mask_p2_zero_m_non_zero], mu_val, part)

        if np.any(mask_p2_zero_m_non_zero):
            res[mask_p2_zero_m_non_zero] = f_p2_zero_m_non_zero(  p2v[mask_p2_zero_m_non_zero], m1v[mask_p2_zero_m_non_zero], m2v[mask_p2_zero_m_non_zero], mu_val , part)

        if np.any(mask_p2_zero_m1_zero):
            res[mask_p2_zero_m1_zero] = f_p2_zero_m1_zero( p2v[mask_p2_zero_m1_zero], m1v[mask_p2_zero_m1_zero], m2v[mask_p2_zero_m1_zero], mu_val, part)

        if np.any(mask_p2_zero_m2_zero):
            res[mask_p2_zero_m2_zero] = f_p2_zero_m2_zero( p2v[mask_p2_zero_m2_zero], m1v[mask_p2_zero_m2_zero], m2v[mask_p2_zero_m2_zero], mu_val, part)
        
        if np.any(mask_p2_nzero_m_zero):
            res[mask_p2_nzero_m_zero] = f_zero( p2v[mask_p2_nzero_m_zero], m1v[mask_p2_nzero_m_zero], m2v[mask_p2_nzero_m_zero], mu_val, part)

        if np.any(mask_p2_nzero_m_non_zero):
            res[mask_p2_nzero_m_non_zero] = f_general( p2v[mask_p2_nzero_m_non_zero], m1v[mask_p2_nzero_m_non_zero], m2v[mask_p2_nzero_m_non_zero], mu_val, part)

        return res