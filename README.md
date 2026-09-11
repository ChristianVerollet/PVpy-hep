# PVpy-hep — Passarino-Veltman Python for HEP

A Python package for symbolic and numeric 1-loop computations in Quantum Field Theory.

[Download the tutorial notebook:](https://github.com/ChristianVerollet/PVpy-hep/raw/main/pvpy_tutorial.ipynb)

`pvpy` provides two essential layers:

- **Symbolic layer:** Enables to write loop integrals, perform tensorial Passarino-Veltman (PV) reduction, set kinematics of the functions, reduce them into simpler expressions, and then to expand them to closed-form analytic expressions depending on the kinematic, all in a `sympy` inhereting form. Closed forms of first derivative of two-points functions w.r.t $p^2$ are also implemented.
- **Numeric layer:** Enables to turn any PV expression into a vectorised `numpy` callable with correct handling of all kinematic limits (equal masses, massless propagators, above-threshold imaginary parts).

> **Limits:** Symbolic form of three-points functions (C-functions) in different kinematic cases not yet
> implementd. Four-points functions (D-functions) for box diagrams not implement at all yet.

## Installation

```bash
pip install PVpy-hep
```

## Minimalist example

```python
from pvpy import Propagator, LoopIntegral, ReduceGeneralNumerator
from pvpy import Gamma, slash, DiracTrace, Project
from pvpy import PV_simplify, set_kinematics, PV_reduce
from pvpy import k, p, m_0, m_1, mu, nu
from pvpy import compile
import sympy as sp

# Define a loop: here QED vacuum polarisation
prop = [Propagator(0, m_0), Propagator(-p, m_1)]
loop = LoopIntegral(k, prop)
numerator = DiracTrace(Gamma(mu) * (slash(k) + m_0) * Gamma(nu) * (slash(k - p) + m_1))
result = ReduceGeneralNumerator(loop, numerator)

# If using .ipynb then use
display(result)
# for a nice symbolic printing of the expression

# Project the tensorial expression: here the transverse part
Pi_T = Project(result, mu, nu, p, 'T')   

# Set the kinematic and replace the potential d = 4 - epsilon
m_Z = sp.Symbol('m_Z', positive=True)
Pi_T_kin = set_kinematics(PV_simplify(Pi_T), {p**2: m_Z**2, m_0: m_0, m_1: m_1})

# Symbolic: Reduce the expression to a simpler form using kinematic and show the closed form if needed
Pi_T_red = PV_reduce(Pi_T_kin)
print(Pi_T_red.doit(part = 'full'))

# Numeric:  transform the symbolic expression into a vectorized numpy callable
f = compile(Pi_T_kin)
print(f(0.511e-3, 91.2))   # m_electron, m_Z in GeV
```

## Requirements

- Python ≥ 3.9
- sympy ≥ 1.12
- numpy ≥ 1.24
- scipy ≥ 1.10

## Status

Version 0.1.0 — alpha release. A, B functions (scalar and tensor) and their $p^2$ derivatives are fully implemented and validated. C functions (3-point) are in progress.

## Reference

C. Verollet, *pvpy: a Python package for symbolic and numeric 1-loop computations* (in preparation).
