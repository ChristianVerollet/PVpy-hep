# Dimensional regularisation convention: d = 4 - epsilon  (epsilon is the standard regulator)
# epsilon_bar is defined by:  1/epsilon_bar = 2/epsilon - gamma_E - ln(pi)  (MS-bar combination)
# UV poles in PV functions appear as 1/epsilon_bar.

import sympy as sp

# Masses
m, m_0, m_1, m_2, m_3, m_4 = sp.symbols('m m_0 m_1 m_2 m_3 m_4', positive=True)

# Momenta
k, p, p_1, p_2, p_3, p_4 = sp.symbols('k p p_1 p_2 p_3 p_4', positive=True)

# Squared momenta — canonical kinematic invariants (use these, not p**2 Pow objects)
k2, p2, p_12, p_22, p_32, p_42 = sp.symbols('k^2 p^2 p_1^2 p_2^2 p_3^2 p_4^2', positive=True)

# Mapping used by Dot.eval and simplify_external_dots: Dot(p,p) → p2 Symbol.
# Only EXTERNAL momenta are listed here — the loop momentum k is intentionally
# excluded so that Dot(k,k) remains a Dot object for the reduction engine to find.
_dot_squares = {p: p2, p_1: p_12, p_2: p_22, p_3: p_32, p_4: p_42}

# Mapping used by kinematic_args: p**2 Pow → p2 Symbol
_pow_to_sym = {sym**2: sq for sym, sq in _dot_squares.items()}

# Lorentz indices (plain Symbols, not positive — indices can be any value)
mu, nu, rho, sigma = sp.symbols('mu nu rho sigma')

epsilon_bar = sp.Symbol(r"\bar{\varepsilon}")   # 1/epsilon_bar = 2/epsilon - gamma_E - ln(pi)
