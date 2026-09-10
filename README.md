# PVpy-hep — Passarino-Veltman Python for HEP

A Python package for symbolic and numeric 1-loop computations in quantum field theory.

`PVpy-hep` provides two layers:

- **Symbolic layer** — write loop integrals, perform tensorial Passarino-Veltman (PV) reduction, simplify and set kinematics, then expand to closed-form analytic expressions via `sympy`.
- **Numeric layer** — turn any PV expression into a vectorised `numpy` callable with correct handling of all kinematic limits (equal masses, massless propagators, above-threshold imaginary parts).

## Installation

```bash
pip install PVpy-hep
```

## Quick start

```python
from pvpy import Propagator, LoopIntegral, ReduceGeneralNumerator
from pvpy import Gamma, slash, DiracTrace, Project
from pvpy import PV_simplify, set_kinematics, PV_reduce
from pvpy.symbols import k, p, m_0, m_1, mu, nu
import sympy as sp

# QED vacuum polarisation
prop = [Propagator(0, m_0), Propagator(-p, m_1)]
loop = LoopIntegral(k, prop)
numerator = DiracTrace(Gamma(mu) * (slash(k) + m_0) * Gamma(nu) * (slash(k - p) + m_1))
result = ReduceGeneralNumerator(loop, numerator)

Pi_T = Project(result, mu, nu, p, 'T')   # transverse scalar

m_Z = sp.Symbol('m_Z', positive=True)
Pi_T_kin = set_kinematics(PV_simplify(Pi_T), {p**2: m_Z**2, m_0: m_0, m_1: m_1})

# Symbolic: closed form
Pi_T_red = PV_reduce(Pi_T_kin)
print(Pi_T_red.doit(part='full'))

# Numeric: numpy callable
from pvpy.numeric import compile
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

C. Verollet, *PVpy-hep: a Python package for symbolic and numeric 1-loop computations* (in preparation).
