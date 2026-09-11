# Dimensional regularisation convention: d = 4 - epsilon  (epsilon is the standard regulator)
# epsilon_bar is defined by:  1/epsilon_bar = 2/epsilon - gamma_E - ln(pi)  (MS-bar combination)
# UV poles in PV functions appear as 1/epsilon_bar.

import sympy as sp

# Masses
m, m_0, m_1, m_2, m_3, m_4 = sp.symbols('m m_0 m_1 m_2 m_3 m_4', positive=True)

# Momenta
k, p, p_1, p_2, p_3, p_4 = sp.symbols('k p p_1 p_2 p_3 p_4', positive=True)

# Squared momenta
k2, p2, p_12, p_22, p_32, p_42 = sp.symbols('k^2 p^2 p_1^2 p_2^2 p_3^2 p_4^2', positive=True)

# Lorentz indices (plain Symbols, not positive — indices can be any value)
mu, nu, rho, sigma = sp.symbols('mu nu rho sigma')

epsilon_bar = sp.Symbol(r"\bar{\varepsilon}")   # 1/epsilon_bar = 2/epsilon - gamma_E - ln(pi)
