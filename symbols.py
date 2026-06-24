# Definition of commonly used symbols and constants

import sympy as sp

p2, m, m1, m2, m3 = sp.symbols('p^2 m m_1 m_2 m_3 ', positive=True)
k, p_1, p_2, p_3 = sp.symbols('k, p_1 p_2 p_3 ', positive=True)
mu = sp.Symbol("mu", positive=True, latex_name=r"\mu")
epsilon = sp.Symbol("epsilon", latex_name=r"\epsilon")

__all__ = ["p2", "m", "m1", "m2", "mu", "epsilon"]