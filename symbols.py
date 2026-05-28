# Definition of commonly used symbols and constants

import sympy as sp

p2, m, m1, m2 = sp.symbols('p^2 m m_1 m_2 ', positive=True)
mu = sp.Symbol("mu", positive=True, latex_name=r"\mu")
epsilon = sp.Symbol("epsilon", latex_name=r"\epsilon")

__all__ = ["p2", "m", "m1", "m2", "mu", "epsilon"]