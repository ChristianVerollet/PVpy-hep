# pvpy/functions/ — PV scalar function implementation

See `../CLAUDE.md` for overall project conventions (propagator signs, Dirac algebra,
package structure, design decisions, known bugs). This file covers the numerical
architecture and per-function implementation details of `A_functions.py` and
`B_functions.py`, plus the roadmap for C functions.

---

## Primary numerical architecture: Feynman-parameter integration

**All B (and C) function numerical evaluation uses direct Feynman-parameter
quadrature as the primary path.** The closed-form analytical expressions
in `guide.tex §4` are kept as the **symbolic path** (`.doit()`) and as
**validation benchmarks** — not as production numerics.

### Motivation

1. **Safety with no kinematic branching:** the Feynman-parameter integral
   is safe for all mass degeneracies (`m0=m1`, `m0=0`, `p²=0`) and for
   general kinematics, with no special-case branching needed. The only
   special treatment is `p² > (m0+m1)²` (above threshold).
2. **Consistency with C functions:** the analytic special-case approach
   would need to be repeated for C functions, which is essentially
   impossible (the kinematic degeneracy tree for 3-point functions is
   far more complex). Numerical integration unifies B and C under the
   same approach.
3. **Error reduction:** the sign bug already found in the old `scalar.py`
   came from transcribing a complex closed-form. The integral form is
   three lines per function and trivially verified.

### Architecture

The UV-divergent pole `1/ε̄` is **always kept symbolic** — it cannot be
integrated away and cancels in renormalized physical quantities. Every
function splits as:
```
Bi(p², m0, m1) = (UV pole) / epsilon_bar + finite_integral(p², m0, m1)
```
The finite part is computed numerically.

Core building block — shared across all functions:
```python
χ(x) = m0²*(1-x) + m1²*x - x*(1-x)*p2
```

Feynman-parameter integrals (finite parts):
```
B0_finite   = -∫₀¹ ln(χ/μ²) dx              (pole = 1)
B1_finite   = -∫₀¹ x·ln(χ/μ²) dx            (pole = 1/2)
B11_finite  = -∫₀¹ x²·ln(χ/μ²) dx           (pole = 1/3)
B00_finite  =  (1/2)·∫₀¹ χ·(1−ln(χ/μ²)) dx  (pole = (m0²+m1²)/4 − p²/12)

dB0/dp²     =  ∫₀¹ x(1−x)/χ dx              (UV-finite)
dB1/dp²     =  ∫₀¹ x²(1−x)/χ dx             (UV-finite)
dB11/dp²    =  ∫₀¹ x³(1−x)/χ dx             (UV-finite)
dB00/dp²    = −1/(12ε̄) + (1/2)·∫₀¹ x(1−x)·ln(χ/μ²) dx   (pole = −1/12)
```

### GL quadrature implementation

- N=100 GL nodes/weights pre-computed on [0,1] at import time (`_GL_X`, `_GL_W`)
- **x = sin²(πt/2) substitution**: Jacobian vanishes at endpoints, regularising
  massless log singularities; nodes cluster near x=0,1 automatically.
  Applied on every sub-interval (including above-threshold splits).
- **IMPORTANT**: `_gl_interval` has `if a >= b: return 0.0` guard at top.
  Without it, calling with a=b=x2=1.0 (massless m1=0) computes chi=0 → log(0) = nan.
- **Below threshold** (`p² < (m0+m1)²`): single GL pass on [0,1], real result.
- **Above threshold** (`p² > (m0+m1)²`): χ has zeros x1 < x2 in (0,1);
  split at x1,x2; imaginary parts added analytically (exact, no quadrature):
  ```
  Im B0   = +π(x2−x1)
  Im B1   = +π(x2²−x1²)/2
  Im B11  = +π(x2³−x1³)/3
  Im B00  = (π/2)·∫_{x1}^{x2} χ dx  < 0  [via antiderivative p2x³/3 − fx²/2 + m0²x]
  ```
- **Broadcasting fix**: all `_numeric_kernel` functions use `np.broadcast_arrays`
  before `atleast_1d` to handle scalar m0/m1 with vector p2.

### dBn/dp² above threshold: Cauchy-PV subtraction

The integrand x^(n+1)(1-x)/χ has simple poles (not log singularities) at x1, x2.

**Real part** — subtract residues then add analytic PV terms:
```
Re = GL[x^(n+1)(1-x)/χ - r₁/(x-x₁) - r₂/(x-x₂)] + r₁·ln((1-x₁)/x₁) + r₂·ln((1-x₂)/x₂)
rᵢ = xᵢ^(n+1)(1-xᵢ) / χ'(xᵢ),   χ'(x₁) = −√K,  χ'(x₂) = +√K
```
**Imaginary part** (exact):
```
Im dBn/dp² = π·(x2^n·dx2/dp² − x1^n·dx1/dp²)
where dx_{1,2}/dp² = (1∓A)/(2p²) − x_{1,2}/p²,   A = (p²−m0²−m1²)/√K
```
Special case n=0 (dB0/dp²):
```
Im dB0/dp² = π·[(m0²+m1²)·p² − (m0²−m1²)²] / (p²²·√K)
```

**dB00/dp² above threshold** — `_db00_scalar` uses split GL on log integrand:
```
Im dB00/dp² = −(π/2)·∫_{x1}^{x2} x(1−x) dx  (negative, Im[lnχ]=−π for χ<0)
```

### Validation status (all functions complete, 2026-08-04)

| Function | Validated | Method | Accuracy |
|---|---|---|---|
| B0 | ✓ | scipy.quad + analytical ref | ≲10⁻¹² below, ~3×10⁻⁸ above |
| B1 | ✓ | reduction formula + swap identity | ≲10⁻¹⁵ all |
| B11 | ✓ | GL + reduction formula + swap identity | ≲10⁻¹⁵ below, exact above |
| B00 | ✓ | GL + reduction formula + p²=0 symbolic | ≲10⁻¹⁵ below, ~10⁻⁸ above |
| dB0/dp² | ✓ | analytical formula + fin.diff. | ≲10⁻¹⁴ all |
| dB1/dp² | ✓ | PV subtraction + fin.diff. | ≲10⁻¹¹ below, ~10⁻⁹ above |
| dB11/dp² | ✓ | PV subtraction + fin.diff. | ≲10⁻¹¹ below, ~10⁻⁹ above |
| dB00/dp² | ✓ | GL (log weight) + fin.diff. | ≲10⁻¹³ all |

---

## Complete formula reference from `guide.tex §4`

**Conventions:** `f = p²+m0²-m1²`, `D = m0²−m1²`, `λ² = λ(p²,m0²,m1²)`.

### Non-zero momentum (p²≠0), general reduction formulas

```
B1   = [A0(m1) − A0(m0) + f·B0] / (2p²)
B1(p²,m1,m0) = B0 − B1(p²,m0,m1)           (swap identity)

B00  = (1/6)·[−p²/3 + m0²+m1² + A0(m1) + 2m0²·B0 − f·B1]
B11  = (1/(3p²))·[p²/6 − (m0²+m1²)/2 + A0(m1) − m0²·B0 + 2f·B1]
B11(p²,m1,m0) = B11 − 2·B1 + B0              (swap identity)
```
Note: the old guide.tex had wrong signs for B00/B11 (from a wrong d·B00 identity). These
are the CORRECTED formulas as implemented in `B_functions.py`.

Derivative reduction formulas at non-zero p²:
```
dB1/dp²   = [A0(m0)−A0(m1) + (m1²−m0²)·B0 + f·p²·dB0/dp²] / (2p²²)
dB11/dp²  = [(m0²+m1²)−2A0(m1)+2m0²·B0+4D·B1−2m0²p²·dB0+4fp²·dB1] / (6p²²)
dB00/dp²  = [−1/3 − B1 + 2m0²·dB0/dp² − f·dB1/dp²] / 6     (pole = −1/12)
```
These are the "fully reduced" forms (everything in terms of A0/B0/dB0/dB1).

### Zero momentum (p²=0) — special cases

**m0=m1=m:**
```
B0=Δ(m),  B1=Δ(m)/2,  B11=Δ(m)/3,  B00=(m²/2)(1+Δ(m))
dB0 = 1/(6m²),  dB1 = 1/(12m²),  dB11 = 1/(20m²),  dB00 = −Δ(m)/12
```
Note: dB1 and dB11 are UV-FINITE at p²=0 (pole of B1 is constant in p², differentiates to zero).
The old guide.tex values `dB1=Δ/2−1/(12m²)` and `dB11=Δ/3−1/(20m²)` are WRONG (cascade error).

**m0=0, m1=m:**
```
B0=Δ(m)+1,  B1=Δ(m)/2+1/4,  B11=Δ(m)/3+1/9,  B00=(m²/4)(3/2+Δ(m))
dB0 = 1/(2m²),  dB1 = 1/(6m²),  dB11 = 1/(12m²),  dB00 = −Δ(m)/12 − 5/72
```
Similarly UV-finite for dB1, dB11.

**m1=0, m0=m:**
Use swap identities; dB00 = −Δ(m0)/12 − 5/72 by symmetry.

**General m0≠m1, both nonzero:**
Read `guide.tex §4.2/§4.3` for B0/B1/B11/B00 at p²=0.
For dB00/dp²(0,m0,m1), use the symmetric form (guide.tex eq. 548):
```
dB00/dp²(0,m0,m1) = −1/12·(1/ε̄ − ln(m0·m1/μ²))
  − (1/72)·[(5m0⁴−22m0²m1²+5m1⁴)/(m0²−m1²)²
             − 3·(m0⁶−3m0⁴m1²−3m0²m1⁴+m1⁶)/(m0²−m1²)³·ln(m0²/m1²)]
```
(guide.tex line 542 is the non-symmetric form with a known typo: `-5m1^2` → `-5m1^4`; line 548 is correct.)

---

## `A_functions.py` and `B_functions.py` — implementation status

### `_eval` dispatch pattern — `.is_zero`, not `sp.Piecewise`

All `_eval` methods use:
```python
def _eval(self, part="full", **hints):
    p2, m0, m1 = map(sp.simplify, self.args)
    if p2.is_zero:
        if m0.is_zero:       return self._eval_p2_zero_m0_zero(part)
        if m1.is_zero:       return self._eval_p2_zero_m1_zero(part)
        if (m0-m1).is_zero:  return self._eval_p2_zero_m0_eq_m1(part)
        return self._eval_p2_zero(part)
    return self._eval_general(part)
```
`.is_zero` returns `True` only when sympy can prove zero; `None` for unknowns → falls
through to general formula. This means `B0(p2, m_0, m_1).doit()` returns a single clean
formula while `B0(0, m, m).doit()` correctly dispatches to the equal-mass case.
**Do NOT revert to `sp.Piecewise`.**

### `A_functions.py`

| Function | Status | Notes |
|---|---|---|
| `A0(m)` | ✓ complete | symbolic + numeric kernel; `A0(0)=0` |
| `A00(m)` | ✓ complete | symbolic only (algebraic reduction to A0) |
| `A0000(m)` | ✓ complete | symbolic only |

### `B_functions.py` — all functions complete (2026-08-04)

**General helper `_bfn_scalar(p2, m0, m1, mu_val, is_above, n)`**: handles B0/B1/B11
with integrand weight x^n. `_gl_interval(p2, m0, m1, mu_val, a, b, n)` is the GL kernel.
B00 has its own `_b00_scalar`/`_b00_gl_interval` (weight χ).
dB1/dB11 use `_dbn_scalar(p2, m0, m1, n, is_above)` (PV subtraction, n=1 or 2).
dB00 uses `_db00_scalar`/`_db00_gl_interval` (GL of x(1-x)·ln|χ/μ²|).

#### B0 — complete

Pole = 1. Symbolic: six kinematic dispatch cases. Numerical: `_bfn_scalar(n=0)`.

Known bug in the old `scalar.py._eval_general`: sign wrong on `(m1²-m0²±p²)/(2p²)·ln`.
Fixed in `B_functions.py` using the symmetric guide.tex eq. 318.
Note: guide.tex eq. 311 has a typo (`+p²` should be `−p²`); eq. 318 is correct.

#### dB0/dp² — complete

UV-finite (pole=0). Numerical: Cauchy-PV above threshold (see architecture section).
Symbolic: derived by ∂/∂p² of guide.tex eq. 318.
Note: guide.tex eq. ~425 for dB0/dp² is algebraically wrong; the implementation uses
the formula derived from differentiating B0 directly.

#### B1 — complete

Pole = 1/2. **NOT symmetric under m0↔m1.** Swap identity: `B1(p²,m1,m0) = B0 − B1(p²,m0,m1)`.
p²=0 general: `½/ε̄ − ½ln(m1²/μ²) + ¼ + m0²/[2D] − m0⁴ln(m0²/m1²)/[2D²]`
p²≠0: reduction formula `[A0(m1) − A0(m0) + f·B0] / (2p²)`.
Note: guide.tex line 386 claiming "B1 symmetric under mass exchange" is WRONG.

#### dB1/dp² — complete

UV-finite (pole=0). Numerical: `_dbn_scalar(n=1)`.
p²=0 general: `−(m1⁴−5m0²m1²−2m0⁴)/(6D³) + m0⁴m1²·ln(m1²/m0²)/D⁴`
p²=0 m0=m1=m: `1/(12m²)`, p²=0 m0=0: `1/(6m²)`, p²=0 m1=0: `1/(3m²)`
p²≠0: `[A0(m0)−A0(m1) + (m1²−m0²)·B0 + f·p²·dB0/dp²] / (2p²²)`

#### B11 — complete

Pole = 1/3. **NOT symmetric under m0↔m1.** Swap identity: `B11(p²,m1,m0) = B11 − 2B1 + B0`.
p²=0 general: `Δ(m1)/3 + (11m0⁴−7m0²m1²+2m1⁴)/[18D²] − m0⁶·ln(m0²/m1²)/[3D³]`
p²≠0 reduction (CORRECTED): `(1/(3p²))·[p²/6 − (m0²+m1²)/2 + A0(m1) − m0²·B0 + 2f·B1]`
Note: guide.tex lines 352-353 are wrong (signs). guide.tex lines 402-403 have a typo (extra m1² in denominator).

#### dB11/dp² — complete

UV-finite (pole=0). Numerical: `_dbn_scalar(n=2)`.
p²=0 general: `(m1⁶−5m0²m1⁴+13m0⁴m1²+3m0⁶)/(12D⁴) − m0⁶m1²·ln(m0²/m1²)/D⁵`
p²=0 m0=m1=m: `1/(20m²)`, p²=0 m0=0: `1/(12m²)`, p²=0 m1=0: `1/(4m²)`
p²≠0: `[(m0²+m1²)−2A0(m1)+2m0²B0+4D·B1−2m0²p²·dB0+4fp²·dB1] / (6p²²)`

#### B00 — complete

Pole = `(m0²+m1²)/4 − p²/12` (kinematic-dependent). **SYMMETRIC under m0↔m1.**
p²=0 general: `(m0²+m1²)/4·(1/ε̄+3/2−ln(m0m1/μ²)) − (m0⁴+m1⁴)/(8D)·ln(m0²/m1²) / (m0²-m1²)`
p²≠0 reduction (CORRECTED): `(1/6)·[−p²/3 + m0²+m1² + A0(m1) + 2m0²·B0 − f·B1]`
Numerical: `_b00_scalar` (weight χ in GL).
Note: guide.tex line 450 `B00(0,0,m) = (m²/2)(3/2+Δ)` is WRONG by factor 2; correct is `(m²/4)(3/2+Δ)`.

#### dB00/dp² — complete

Pole = −1/12 (kinematic-independent). **SYMMETRIC under m0↔m1.**
p²=0 general: see formula reference section above (guide.tex eq. 548).
p²=0 m0=m1=m: `−Δ(m)/12` (finite part = `ln(m²/μ²)/12`)
p²=0 m0=0 or m1=0: `−Δ/12 − 5/72`
p²≠0: `[−1/3 − B1 + 2m0²·dB0/dp² − f·dB1/dp²] / 6`
Numerical: `_db00_scalar` (GL of x(1-x)·ln|χ/μ²|, split at x1,x2 above threshold).

---

## Roadmap: 3-point functions (planned, not started)

The 2-point sector (A, B, dB) is now **complete and validated**. C functions are next.

**Architecture consistency:** the Feynman-parameter numerical approach used for B
functions extends directly to C functions. Same `_GL_X`/`_GL_W` infrastructure,
same split-interval strategy, same PV subtraction for derivative integrals.

### `C0` (scalar triangle)

User has derived the formula. **Strategy:** direct Feynman-parameter 2D numerical
integration over the simplex as the primary numerical path (consistent with B approach).
The dilogarithm closed form implemented as symbolic `.doit()` and validation cross-check.
One general formula valid across the entire complex kinematic plane via `+iε` prescription
— no case-splitting by kinematic region.

### Tensor coefficients (C1, C2, C11, C12, C22, C001, C002, C111, C112, C122, C222, ...)

**Two-tier strategy:**

1. **Algebraic path (away from Gram-determinant singularity):** standard PV reduction
   to A0/B0/B1/C0 with rational coefficients involving `1/det(Gram)` (2×2 matrix of
   dot products of external momenta q1, q2). Unavoidable for 3-point tensors.

2. **Numeric fallback (near small Gram determinant):** direct Feynman-parameter 2D
   integration carrying the numerator through — no Gram determinant. Structurally
   identical to C0 numeric integration but with monomial-in-x2,x3 weight. Specific
   weight for each C_ijk must be worked out by hand (user is doing this). General
   technique: Davydychev's papers on N-point tensor reduction via Feynman parameters.

### Why `.doit()` on tensor C-functions is not meant to be human-readable

Fully expanding e.g. C11 via algebraic reduction substitutes dilogarithm C0/B0 forms
into a linear combination weighted by 1/det(Gram) — a wall of Li2 terms. This is
expected, not a bug. Keep `.reduce()` (stopping at C0/B0/A0 as opaque symbols) as the
human-facing algebra-friendly form; reserve full `.doit()`/numeric evaluation for when
actual numeric kinematics are plugged in.
