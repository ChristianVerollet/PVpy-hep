# tensorial_decomposition.py -- project context for Claude Code

This file is a handoff summary of a long design conversation with chat-Claude.
It exists so a fresh Claude Code session has the reasoning trail, not just the
code. Read this before making changes -- several non-obvious bugs were
already found and fixed once; don't reintroduce them.

## What this is

A Python/sympy library for 1-loop Feynman integral reduction, in the spirit
of Package-X / COLLIER, but built incrementally and kept readable. Two
layers, per the project's own split:

1. **Tensorial decomposition** (this file): write a loop integral's
   numerator/denominator and get back a sympy expression in terms of
   Passarino-Veltman (PV) functions, metric tensors, and external momenta.
2. **PV function implementations** (separate, user-owned files: `A0`, `B0`,
   `B00` exist; user is implementing `B1`, `B11`, `C0`, `C00`, etc.
   themselves in their own module and will import them into `PV_REGISTRY`).
   This file does NOT implement PV functions -- it only orchestrates which
   one to call with which arguments.

## Core convention -- fix this in your head before touching anything

Propagators are `D_i = (k + q_i)^2 - m_i^2`, `i = 0..N-1`, with **q_0 = 0
always** (you always have the freedom to shift `k` so the first propagator
has no offset). The `q_i`, `i=1..N-1`, are then the topology's independent
external momenta. `LoopIntegral.__post_init__` enforces `propagators[0].shift
== 0` and will raise if violated -- this is intentional, not a bug to "fix"
by relaxing it.

## The object language (custom sympy Function subclasses)

- `g(mu, nu)` -- metric tensor, canonicalizes index order (symmetric).
- `Mom(p, mu)` -- momentum component `p^mu`. **Linear in `p`**:
  `Mom(k+p, mu) -> Mom(k,mu) + Mom(p,mu)`, `Mom(2*p,mu) -> 2*Mom(p,mu)`.
  This linearity was added specifically so gamma-trace output (which
  produces shifted momenta like `(k+p)^mu`) can be recognized as
  k-dependent.
- `Dot(a, b)` -- scalar dot product placeholder, symmetric. Used for
  *fully-contracted* k-dependence (`Dot(k,k)` = k^2, `Dot(k,p)` = k.p).
  External dot products (`Dot(p1,p2)`, neither arg is k) get
  auto-simplified by `simplify_external_dots` via the polarization identity
  `a.b = ((a+b)^2-a^2-b^2)/2`; this runs automatically inside
  `ReduceLoopIntegral`/`ReduceGeneralNumerator`.
  **Display only** (`_sympystr`/`_latex`, `eval` untouched): when `a == b`,
  `Dot(a,a)` prints as `a^2` (or `\left(...\right)^{2}` in LaTeX if `a`
  isn't atomic, e.g. `Dot(k+p_1,k+p_1)` -> `(k + p_1)^2`) instead of the
  literal `(a.a)`. This is purely cosmetic -- the object is still a real
  `Dot` instance internally (`type(Dot(k,k)) is Dot`, not `Pow`), so the
  substitution algebra and the `_validate_numerator` hard rule below still
  see and pattern-match on `Dot(k,k)` exactly as before. Works for *any*
  symbol automatically; the `k2`/`p2`/`p_12`-style placeholder symbols in
  `symbols.py` are no longer needed just for pretty-printing (still useful
  if you want a literal standalone symbol to substitute in for further
  simplification).
- `Eps(s1,s2,s3,s4)` -- Levi-Civita object for gamma_5 traces. Each slot is
  *either* a free Lorentz index *or* a momentum expression (mirrors how
  `Mom` packs momentum+index together). Totally antisymmetric: vanishes on
  repeated args, canonicalizes with permutation-parity sign.
- `G5` -- a **zero-index TensorHead marker** (`sympy.tensor.tensor`), not
  part of our own language. You insert `G5()` at gamma_5's literal position
  inside a sympy.physics.hep.gamma_matrices product, e.g.
  `G5() * GammaMatrix(a) * k(-a) * ...`. It exists purely so the position
  of gamma_5 relative to the other gammas survives sympy's tensor
  multiplication (needed because gamma_5 *anti*commutes with gamma^mu --
  the sign of the final trace depends on how many gammas you'd have to
  move it past to reach the front).

**Hard rule, enforced by `_validate_numerator`:** the loop momentum `k`
must NEVER appear as a bare sympy symbol/power outside `Dot(k,...)` or
`Mom(k,...)`. A bare `k**2` is silently treated as an inert "spectator"
factor (wrong answer, no crash) if this guard isn't there -- this was a
real bug caught during development. Don't remove the guard to "simplify"
input handling.

## Two reduction engines -- know which one a numerator needs

| | `TensorialDecomposition` (Method B) | `ReduceLoopIntegral` / `ReduceGeneralNumerator` (Method A) |
|---|---|---|
| numerator form | pure `k^{mu1}...k^{muR}`, set via `LoopIntegral.rank` | any expression: spectator tensors (`g`, `Mom(p,.)`, couplings) times k-dependence written with `Dot`/`Mom(k,.)` |
| mechanism | Wick-contraction combinatorics (`tensor_structures`) enumerating all pair/momentum-label patterns | algebraic substitution `k^2 -> D_0+m_0^2`, `2k.q_i -> D_i-D_0-q_i^2+m_0^2-m_i^2`, expand as polynomial in `D_i`, match powers against propagator powers |
| propagator powers | only power 1 | **any integer power**, via `d/dm^2` trick (needs derivatives of PV functions, which the user has) |
| when it's the right tool | you want the open-index Lorentz tensor itself, to combine with other open structures later | numerator is/contains scalar k-dependence, OR (via `ReduceGeneralNumerator`) a per-term mix of open k-indices and scalar k^2/k.p pieces -- e.g. **gamma trace output, always** |

`ReduceGeneralNumerator` is the union: for each additive term it checks
whether that term has open `Mom(k,.)` factors. If none, it's pure Method A.
If some, it builds the full open-rank tensor via `TensorialDecomposition` at
the term's *effective* rank (open indices + 2x power of `Dot(k,k)` + power
of each `Dot(k,q_i)`), then uses `contract_with_metric`/
`contract_with_momentum` to collapse the extra dummy slots back down.
This is the one you want for anything coming out of a gamma trace.

Key insight from the design discussion, worth remembering: Method A is
**strictly more useful** whenever the final answer doesn't need to stay an
open tensor, because it expresses everything directly via low-point
power-1 PV functions (`A0`, `B0`, `C0`) -- it never needs `B1`/`B00`/`B11`/
etc. to be implemented. Going through Method B's tensor ansatz and then
contracting (e.g. `contract_with_metric(TensorialDecomposition(...))`)
instead pulls in `B00`/`B11` placeholders even when the final contracted
answer is expressible purely via `A0`/`B0`. Prefer Method A whenever
possible.

## Package structure (as of current session)

```
pvpy/
  __init__.py              — re-exports everything listed below
  algebra.py               — g, Mom, Dot, Eps, contract, simplify_external_dots
  tensorial_decomposition.py — LoopIntegral, Propagator, TensorialDecomposition,
                               ReduceGeneralNumerator, contract_with_metric/momentum/eps,
                               from_gamma_trace, gamma5_trace, G5, GAMMA5_TRACE_COEFF
                               (imports algebra objects via `from .algebra import ...`
                               and re-exports them for backward compatibility)
  dirac.py                 — DiracMatrix, slash, Gamma, Gamma5, DiracTrace
  symbols.py               — pre-defined sympy Symbols (momenta, masses, indices)
  base.py                  — PVFunction base class
  simplify.py              — set_kinematics, reduce_pv
  functions/
    __init__.py            — imports A0, A00, A0000 from A_functions.py;
                             B0, dB0_dp2, B1 from B_functions.py (canonical)
    A_functions.py         — A0, A00, A0000  ← canonical, use this
    B_functions.py         — B0, dB0_dp2, B1 ← canonical, all validated
    scalar.py              — LEGACY: A0, B0, B00 — DO NOT IMPORT (pollutes subclass registry)
    tensors.py             — LEGACY: A00, B1 — superseded
```

`algebra.py` was split out of `tensorial_decomposition.py` to give a clean
import layer: `from pvpy.algebra import g, Mom, Dot, Eps, contract`. Old
imports like `from pvpy.tensorial_decomposition import g, Mom` still work
via the re-export.

### `scalar.py` / `tensors.py` retirement plan

`scalar.py` and `tensors.py` will be deleted once `A_functions.py` and
`B_functions.py` are fully validated and `functions/__init__.py` is updated
to import from them instead. Do not add features to `scalar.py`/`tensors.py`;
all new work goes into `A_functions.py` and `B_functions.py`.

## Convention — metric, ε, γ₅ sign (settled, don't change)

Metric: (+,−,−,−).  Levi-Civita: `ε_{0123} = +1` (lower indices, "physics"
convention).

γ₅ definition: **γ₅ = −i γ⁰γ¹γ²γ³** (Itzykson-Zuber / Package-X sign, not
Peskin-Schroeder).  This determines the trace:
  - `Tr[γ₅ γ^μ γ^ν γ^ρ γ^σ] = −4i ε^{μνρσ}` (upper ε)
  - `Tr[γ₅ γ_μ γ_ν γ_ρ γ_σ] = −4i ε_{μνρσ}` (lower ε)

**`GAMMA5_TRACE_COEFF = +4*sp.I`** in `tensorial_decomposition.py`.

If you see it as `−4*sp.I` that's the Peskin-Schroeder sign (`γ₅=+iγ⁰...`);
changing it flips only the chiral cross-term `g_V*g_A`, not the `g_V²`/`g_A²`
pieces, so the sign difference is subtle and easy to miss without an explicit
comparison against Package-X output (which was done to confirm the current sign).

## Gamma matrices / traces

- `from_gamma_trace(trace_expr, momentum_map)` converts ordinary
  `sympy.physics.hep.gamma_matrices.gamma_trace` output into our
  `g`/`Mom`/`Dot` language. Works by reading `term.components` and
  `term.get_indices()` directly (NOT `as_ordered_factors()` -- tensor Mul
  doesn't decompose that way, see "bugs already found" below), pairing
  indices with matching name + opposite `is_up` into `Dot`s, leaving
  singletons as open `Mom`/`g` slots.
- `gamma5_trace(expr, momentum_map)` handles `gamma_5` insertions via the
  `G5()` marker described above. Convention:
  `Tr[g5 * gamma^i1 gamma^i2 gamma^i3 gamma^i4] = GAMMA5_TRACE_COEFF * Eps(i1,i2,i3,i4)`,
  with `GAMMA5_TRACE_COEFF = -4*I` (states the convention explicitly --
  flip the sign if the user's metric/epsilon convention differs).
  - **Any number of `G5` markers per term is now supported** (previously:
    exactly one, else `NotImplementedError`). `_cancel_g5_pairs(term)` runs
    first and reduces the count to 0 or 1 via `G5**2 = 1`: it walks
    `term.args` (physical left-to-right order -- zero-index `TensorHead`s
    keep their position under multiplication, bug #6), repeatedly slides
    the *second* `G5` of the first pair found leftward to meet the first
    one, picking up a `(-1)` for every `GammaMatrix` factor passed (from
    `{gamma5, gamma^mu} = 0`; momentum-head factors are ordinary vectors
    and contribute no sign), then drops the adjacent pair. This is exactly
    what's needed for a chiral vector/axial-vector coupling squared, e.g.
    `(slash(p1)-m)*(g_V-g_A*Gamma5())*Gamma(mu)*(slash(p2)-m)*(g_V+g_A*Gamma5())*Gamma(nu)`
    -- the `g_A**2` term has two non-adjacent `G5`'s. Verified by hand
    against the standard V/A trace identity: the `g_V**2` piece reproduces
    the plain vector trace exactly, `g_A**2` flips the sign of the
    momentum cross-terms relative to it, and the `g_V*g_A` cross term
    vanishes (needs a 3-gamma trace, identically zero) -- also verified
    that a *triple* `G5` insertion reduces to the same result as a single
    one, as required by `G5**2=1`.
  - If cancellation leaves 0 `G5`'s, the term is a plain trace and is
    routed through `from_gamma_trace(gamma_trace(term), momentum_map)`
    (`term * Symbol("d")` directly, without calling `gamma_trace`, in the
    degenerate edge case where *every* factor was a `G5` and nothing
    physical is left).
  - Exactly 4 gamma matrices after stripping `G5` is the only nonzero case
    implemented for the single-`G5`-remaining branch (0-3 correctly
    returns 0; 6+ needs the general identity with extra `g.eps` terms --
    not implemented).
  - **IMPLEMENTED:** `ReduceGeneralNumerator` handles `Eps(k, s2, s3, s4)`
    by introducing a dummy Lorentz index `d`, feeding `Mom(k, d)` as an
    extra open k-index into `TensorialDecomposition`, then calling
    `contract_with_eps(result, d, Eps(d, s2, s3, s4))` to substitute the
    reduced momentum back into the Eps slot. Pure-spectator `Eps` (no k in
    any slot) passes through unchanged. Self-energy chiral traces vanish
    correctly (Eps(p1,...,p1) = 0 after contraction).
  - Common call-site mistake (not a bug): `Gamma5` is a **function** --
    `g_A * Gamma5` multiplies by the function object itself and raises
    `TypeError: unsupported operand type(s) for *: 'Symbol' and 'function'`.
    Must call it: `g_A * Gamma5()`.

## `dirac.py` — user-facing Dirac algebra API

The public interface (all exported from `pvpy`):

```python
slash(p)          # γ·p = γ^a p_a  (p is a sympy Symbol)
Gamma(mu)         # γ^μ  with free index mu (sympy Symbol)
Gamma5()          # γ₅  — note: must be called, Gamma5 alone is the class
DiracTrace(expr)  # Tr[expr], auto-detects γ₅ presence
```

`slash`, `Gamma`, `Gamma5` return `DiracMatrix` objects.  Arithmetic `+`, `-`,
`*`, scalar multiplication all work.  Scalars (plain sympy expressions, ints,
floats) can be added or multiplied freely; they land in the scalar-coefficient
part of each term and are carried through to the trace output unchanged.

**Typical workflow:**

```python
expr = (slash(p1) - m) * Gamma(mu) * (g_V - g_A*Gamma5()) * (slash(p2) - m) * (g_V + g_A*Gamma5()) * Gamma(nu)
trace = DiracTrace(expr)        # returns sympy expr in g/Mom/Dot/Eps
result = contract(trace * propagator_numerator)   # Einstein sum over free indices
```

**Critical: take the trace BEFORE contracting.**  `contract` only understands
`g`/`Mom`/`Dot`/`Eps` sympy objects — it does not know about `DiracMatrix`
internals.  Calling `contract(dirac_matrix * something)` will not contract the
Dirac free indices with the external tensor; `DiracMatrix.__mul__` will absorb
the external expression as a *scalar coefficient* of each term, which is
wrong.  Always: `contract(DiracTrace(expr) * external_tensor)`.

**`Gamma5()` must be called (with parentheses).**  `g_A * Gamma5` silently
multiplies by the *class object* itself, not an instance, and raises
`TypeError` later.  This is already noted in the CLAUDE.md gamma_5 section but
is easy to hit again — mentioned here because it's the most common call-site
mistake in practice.

`DiracTrace` auto-detects γ₅: if any term contains a `G5` component, it
dispatches to `gamma5_trace`; otherwise to `gamma_trace + from_gamma_trace`.
Zero-trace guard: `gamma_trace` returns `sp.Integer(0)` for odd-count gamma
products; `DiracTrace` checks `if traced == sp.S.Zero: continue` before
calling `from_gamma_trace` (which would blow up on a plain integer, bug #8).

## `contract()` — Einstein summation (algebra.py)

```python
from pvpy import contract
result = contract(expr)          # auto-detects all repeated indices
result = contract(expr, d=sp.Symbol('d'))  # custom dimension symbol (default: sp.Symbol('d'))
```

Rules applied iteratively until stable (term by term after `sp.expand`):
- `g(a, a)  → d`
- `g(a,b) * Mom(p,a) → Mom(p,b)`
- `g(a,b) * g(a,c)   → g(b,c)`
- `Mom(p,a) * Mom(q,a) → Dot(p,q)`
- `g(a,b) * Eps(a,...) → Eps(b,...)`
- `Mom(p,a) * Eps(a,...) → Eps(p,...)`

Index detection convention: both args of `g` are indices; only the **second**
arg of `Mom(p, mu)` is an index (first is a momentum — never tracked); all
four args of `Eps` are treated as potential indices.  This prevents momentum
symbols appearing in the first slot of `Mom` from being mis-identified as
repeated indices.

`Eps*Eps` contractions are **not** implemented — they produce complex
combinations of metric determinants and are left as-is.

**Input guard:** `contract()` raises a clear `TypeError` with a hint message
if it receives a non-sympy object (e.g. a `DiracMatrix`).  Before the guard
was added, the failure manifested as an obscure `SympifyError` deep inside
`sp.expand` via sympy's deprecated string-fallback.  The guard is:
```python
if not isinstance(expr, sp.Basic):
    raise TypeError(
        f"contract() expects a sympy expression ... got {type(expr).__name__}.\n"
        f"Hint: take the trace first — contract(DiracTrace(expr) * other), ..."
    )
```

**`set_kinematics` works on any sympy expression**, not only on PV-function
expressions.  It is simply `sp.expand(expr.subs(substitutions))`, so it
handles `Dot`, `Mom`, `g`, `Eps` objects transparently.  The `d=4`
substitution is just another entry in the same dict:
```python
result = set_kinematics(result, {
    Dot(k, k): m_Z**2,
    Dot(p_1, p_2): (m_Z**2 - 2*m**2) / 2,
    sp.Symbol('d'): 4,          # set spacetime dimension to 4
})
```
**Caution:** for loop integrals, set `d=4` only *after* UV divergences have
cancelled (renormalization / finite quantity).  The `d` from trace prefactors
and the `d` implicit in PV function definitions are linked — premature
substitution hides the poles before they can cancel.

## Dirac string display (`dirac.py`, `DiracMatrix`)

`DiracMatrix` (the algebra object returned by `slash()`/`Gamma()`/`Gamma5()`
and their products, consumed by `DiracTrace()`) used to `repr()` as an
opaque `DiracMatrix(N terms)`. It now renders as real physics notation
(`repr()` -> plain text with `γ^{mu}`/slashed-p unicode, `_repr_latex_()` ->
`\gamma^{\mu}`, `\not{p_1}`, `\gamma_5` picked up automatically by Jupyter's
`display()`), purely for human inspection *before* tracing -- this is
display-only and untouched by `DiracTrace`/arithmetic, which still operate
on the underlying sympy tensor objects in `self.terms`.

Mechanism, in case it needs extending (e.g. for a new token kind):
- `_GammaTok`, `_SlashTok`, `_Gamma5Tok` are internal `sp.Function`
  subclasses marked `is_commutative = False`. That's the load-bearing
  trick -- ordinary sympy `Mul` reorders commutative factors however it
  likes, which would destroy the physical (non-commuting) gamma-matrix
  order. Noncommutative `Mul` factors keep their input order.
- `_term_factor_tokens(tensor, momentum_map)` walks `tensor.components` /
  `tensor.get_indices()` in the same style as `from_gamma_trace` (see
  above) to figure out, for each `GammaMatrix` slot in order, whether its
  index is a dummy paired with a momentum-head component (-> `_SlashTok`)
  or genuinely free (-> `_GammaTok`); `G5` components become
  `_Gamma5Tok()`. Momentum-head components themselves emit no token of
  their own -- they're absorbed into the preceding slash.
- `DiracMatrix.as_sympy()` sums `coeff * product-of-tokens` over
  `self.terms` (empty token list = identity = bare `1`, so scalar-only
  terms just show their coefficient); `__repr__`/`_repr_latex_` print that.
- Verified against `from_gamma_trace`'s exact index-pairing logic, so a
  free-index name here is guaranteed to be either a real user-facing free
  index (never contracted, since `Gamma(mu)` mints a fresh index object
  each call) or one of `slash()`'s internal `_pvpy_c{n}` dummies (always
  paired with its momentum head in the same term) -- no orphan dummy names
  should ever leak into the display.

## Design decisions raised and explicitly deferred (don't redo this debate)

- **`DiracMatrix` always stores a flat, fully-expanded list of `(coeff,
  tensor)` monomials (`self.terms`) -- there is no factored/tree form
  retained anywhere.** `__mul__` builds this list as the full cross
  product of the two operands' term lists, which is *why* multiplying
  several chiral/mass factors together (e.g. the V/A trace example above)
  necessarily explodes into every monomial before `display()` ever runs --
  it's not a printing choice, the expansion already happened at
  construction time. This is load-bearing: `DiracTrace` (and
  `gamma5_trace`/`from_gamma_trace`) work by iterating `expr.terms` and
  tracing each monomial independently.
  - Question raised: can we keep it factored for display and only expand
    when the user asks (`.expand()`)? Answer given and accepted: not
    without a real redesign -- `DiracMatrix` would need to become a lazy
    expression tree of un-multiplied sub-`DiracMatrix` objects, flattening
    into `self.terms` only when `DiracTrace()` actually needs to trace it.
    That's a change to the whole arithmetic layer for a display-only
    benefit, and was **explicitly deferred** (not rejected -- just not
    worth it right now). Workaround needing no code change: build named
    sub-pieces (e.g. `L = (slash(p1)-m)*(g_V-g_A*Gamma5())`) and `display()`
    those individually before combining them for the final trace -- each
    piece alone stays as compact as it can be.
- **Scope of the algebra layer (`g`/`Mom`/`Dot`/`Eps`/`DiracMatrix`) is
  considered sufficient for the package's actual goal (1-loop PV
  reduction) as of this session.** The user confirmed `Eps` usage
  (`Eps(mu,nu,rho,sigma) * Mom(p,mu) * Mom(p,nu)` etc., already added to
  `pvpy_tutorial.ipynb`) is enough and does not want further algebra
  elaboration (e.g. the lazy-tree `DiracMatrix` redesign above, or richer
  `Eps`/tensor features) unless a real need shows up later -- possibly
  after publication. Don't proactively add algebra features beyond what's
  asked; this is a considered decision, not an oversight.

## Bugs already found and fixed during development (don't reintroduce)

1. **`TensAdd` is NOT a subclass of `sp.Add`.** `isinstance(expr, sp.Add)`
   silently fails for tensor expressions. Must check
   `isinstance(expr, (sp.Add, TensAdd))`.
2. **Tensor products don't auto-distribute over a sum sitting inside a
   `Mul`.** `G(a)*k(-a)*(G(b)*k(-b)+G(b)*p(-b))*G(nu)` stays a single
   `TensMul` with an unexpanded `TensAdd` buried inside one factor, until
   you call `.expand()` on it. Always `.expand()` a tensor expression
   before trying to split it into additive terms.
3. **`sympy.Permutation` is at `sympy.combinatorics.Permutation`,** not
   top-level `sp.Permutation`.
4. **Differentiate w.r.t. mass BEFORE substituting a numeric mass value,
   not after.** If a propagator's mass is e.g. literally `0` (massless),
   building `B0(p**2, 0, m)` and then trying to differentiate "with
   respect to 0" is nonsensical -- there's no symbol left. Fix: build the
   PV call with a **fresh `sp.Dummy()` mass**, differentiate w.r.t. the
   dummy, and only substitute the real (possibly literal 0) value
   afterward, using `sp.limit` (not `.subs`) for any mass that was
   differentiated, since the intermediate chain-rule expression can have
   apparent (removable) `1/m` singularities that only cancel once a real
   closed-form PV function is plugged in -- confirmed by testing with a
   toy concrete `B0`.
5. **`Mom` needed explicit linearity added** (it doesn't come for free from
   being an `sp.Function`) -- without it, `Mom(k+p, mu)` is an opaque
   single object and the open/scalar k-detection logic can't tell it's
   k-dependent.
6. **Zero-index `TensorHead` (`G5`) preserves its position in `TensMul`**
   when multiplied with other tensors -- verified empirically, this is
   what makes the gamma_5 position-sign-tracking trick work at all.
7. **`from_gamma_trace` only handled a single free metric factor per term.**
   It pooled every free (uncontracted) metric index from a term into one
   flat list and only had branches for `len == 2` (emit one `g(mu,nu)`) or
   `len == 1` (raise `NotImplementedError`). A term with **two independent**
   free metric factors -- e.g. `tr[gamma^mu gamma^nu gamma^rho gamma^sigma]`
   produces terms like `metric(mu,rho)*metric(nu,sigma)` -- has 4 free
   indices belonging to 2 different metric components, hit neither branch,
   and silently dropped both `g(...)` factors, leaving only the numeric
   coefficient (`DiracTrace(Gamma(mu)*Gamma(nu)*Gamma(rho)*Gamma(sigma))`
   returned bare `4` instead of the correct
   `4*(g(mu,nu)*g(rho,sigma) - g(mu,rho)*g(nu,sigma) + g(mu,sigma)*g(nu,rho))`).
   Fix: tag each occurrence with the id of the tensor component it came
   from, and reconstruct one `g(...)` per component that has *both* its
   indices still free, instead of grouping by index-list-length alone.
   Caught via the tutorial notebook's `Gamma(mu)*Gamma(nu)*Gamma(rho)*Gamma(sigma)`
   smoke test -- worth keeping that exact case in mind as a regression
   check since it's the simplest input that exercises multiple independent
   metric factors in one term.
8. **`getattr(term, "coeff", term)` is not a safe "is this a tensor"
   check** -- every plain `sympy.Expr` (a bare `Integer`, `Symbol`, ...)
   already has an unrelated `.coeff` *method* (`Expr.coeff(x)`, for
   extracting the coefficient of `x`), so the `getattr` default never
   triggers for a non-tensor `term`; it silently returns a bound method
   object instead of `term` itself, and downstream `sp.sympify(...)` on
   that bound method blows up with a `SympifyError`/deprecation warning
   about `sympify()`'s string fallback. Surfaced in both
   `from_gamma_trace` and `gamma5_trace` once `_cancel_g5_pairs` (bug/
   feature #7's sibling, see the `gamma5_trace` entry above) started
   producing terms whose full `gamma_trace(term)` is identically
   `sp.Integer(0)` (an odd-gamma-count trace vanishes exactly). Fix: check
   `hasattr(term, "components")` first (a real distinguishing feature of
   tensor objects) and only then read `term.coeff`/`term.components`/
   `term.get_indices()` directly; for a plain scalar `term`, use it as-is.

9. **`_GammaTok._latex` / `_SlashTok._latex` / `_Gamma5Tok._latex` must
   accept `**kwargs` / `exp=None`.** Sympy's `_print_Pow` calls the base
   object's `_latex(printer, exp=<exponent_string>)` so the object can render
   itself with the exponent attached (e.g. `\left(\gamma_5\right)^{2}`).  Any
   display token class that only declares `_latex(self, printer)` will receive
   an unexpected `exp` keyword argument and raise `TypeError` when that token
   appears squared (e.g. `_Gamma5Tok()**2`).  This happens for `Gamma5()`
   whenever a product contains two γ₅ insertions that haven't been cancelled
   yet — `as_sympy()` / `DiracMatrix.__repr__` / `_repr_latex_()` all blow
   up.  Fix: add `def _latex(self, printer, exp=None): ...` (with `if exp is
   not None: return r'\left(...\right)^{%s}' % exp`) and `def _latex(self,
   printer, **kwargs): ...` for `_GammaTok`/`_SlashTok` where exponents don't
   arise in practice.

10. **`contract()` didn't handle `Pow(g/Mom/Eps, n)`.** Sympy collapses
    `g(mu,nu) * g(mu,nu)` into `g(mu,nu)**2` (a `Pow` object) before
    `contract` ever sees it.  The factor collection loop only looked for
    `isinstance(f, (g, Mom, Eps))` so `Pow` objects silently ended up in
    `scalar_part` and were never contracted — `g(mu,nu)**2` stayed as-is
    instead of reducing to `d`, `Mom(k,nu)**2` stayed instead of becoming
    `Dot(k,k)`.  Fix: also check `isinstance(f, sp.Pow) and isinstance(f.base,
    (g, Mom, Eps)) and f.exp.is_integer and f.exp.is_positive` in the factor
    loop and expand into `int(f.exp)` copies of `f.base`.

## Known limitations / honest TODO list

- `ReduceLoopIntegral`/`ReduceGeneralNumerator` do a **single** substitution
  pass. If a term's numerator-degree-in-k exceeds the combined denominator
  power for some propagator even after substitution, it raises
  `NotImplementedError` rather than recursing to re-expand the surplus.
  Unlikely at rank <=3 but possible for high-rank numerators against
  low-power propagators.
- `pv_name` only maps N=1..4 to letters A/B/C/D. Trivial to extend for
  5-point (E) etc. if ever needed.
- `decompose_in_basis` requires every momentum appearing in `Dot(k, V)` to
  be linear in the *same* elementary momentum symbols used in the
  propagator shifts. Fails loudly (clear `ValueError`) if not -- this is
  intentional, not a bug.
- `Eps` handling in `ReduceGeneralNumerator` is now implemented (see
  gamma_5 section above for the contract_with_eps approach).
- PV function mass argument convention is currently "mass" (`m`), not
  "mass-squared" (`m^2`). This works (verified) but produces intermediate
  chain-rule artifacts during mass-derivatives that only resolve once a
  real formula is substituted in. If the user's actual `A0`/`B0`/etc.
  signatures take mass-squared directly, `derivative_wrt_mass_squared`
  should be simplified to a direct `sp.diff` (no chain rule needed at
  all) -- ask before assuming either way.
- **SUPERSEDED, do not do this (2026-07-09):** an earlier version of this
  note suggested deriving `B1`/`B11`/`B00`/etc. by contracting the
  open-tensor ansatz and *solving the resulting linear system* for them in
  terms of `A0`/`B0`. **The user tried this direction and rejected it**:
  the current `B1.reduce()` in `functions/tensors.py` is exactly this kind
  of solved-system formula --
  `B1 = (A0(m0) - A0(m1) - (p2+m0**2-m1**2)*B0(p2,m0,m1)) / (2*p2)` -- and
  it is **ill-defined at `p²=0`** (division by `p²`; the docstring already
  flags `B1(0,...)` as needing a separate unimplemented limiting formula).
  This is the same class of problem as bug #4 (chain-rule/`1/m` artifacts
  from differentiating before substituting) -- solving a linear system
  algebraically introduces spurious poles at kinematic points where the
  *true* function is perfectly finite, because the solving step divides by
  a combination (here, essentially the system's determinant / a `p²`
  factor) that isn't actually present in the closed form.
  **Current plan instead:** the user has computed genuine closed-form
  formulas for `B0`, `B1`, `B11`, `B00` by hand (case-by-case over the
  different kinematic regions, same style as the existing `A0`/`B0`/`B00`
  in `scalar.py`) that are well-defined at `p²=0` and will implement them
  directly as primitives -- no linear-system solving. **Do not** re-solve
  for any tensor PV function from a linear system as a shortcut; only use
  a closed-form the user has actually derived.
  - Open question raised, not decided: whether to reorganize
    `functions/scalar.py` / `functions/tensors.py` by point-count instead
    (e.g. a "1-point" module for `A0`/`A00`, a "2-point" one for
    `B0`/`B1`/`B00`/`B11`, ...) rather than by scalar-vs-tensor. Don't
    assume this happened -- check the actual file layout.
  - `C0`, `C00`, `C001`, etc. (3-point) are still undefined placeholder
    `sympy.Function`s (via `get_pv_function`'s fallback) unless added to
    `PV_REGISTRY`; not addressed by the above, still open.
  - **`reduce_pv`/`simplify.py`'s intended role going forward:** apply
    known-good, *general* (never kinematically-singular) linear identities
    among PV functions as **forward, pattern-matched substitutions only**
    -- e.g. replace the literal combination `p²*B11 + d*B00` with
    `A0 + m0**2*B0` *only when that exact combination appears* in an
    expression. **Never** invert such an identity to solve for/isolate an
    individual tensor coefficient (e.g. don't use it to eliminate `B00`
    from an expression that doesn't already contain the matching
    combination) -- that inversion is exactly the ill-defined-at-`p²=0`
    trap above. Exact relations to implement are still being worked out
    with the user; ask for the precise identity before hard-coding one.

## `functions/scalar.py` is being fully rewritten from `articles/guide.tex` (in progress, 2026-07-09)

**Do not trust the current `functions/scalar.py` as a source of truth.**
The user explicitly said so after a confirmed bug was found in it (below):
`articles/guide.tex` (the paper the user is writing, which derives every PV
function from scratch and states the analytic formula for each kinematic
case) is the authoritative reference from now on. The user's stated plan:
replace everything currently in `scalar.py` with the formulas from
`guide.tex`, adopting the two-tier architecture worked out today (see next
subsection) instead of the current file's more complex multi-branch
relative-threshold masking. If you're asked to touch `scalar.py`, check
`guide.tex` first and expect the file to look quite different from what's
described in older parts of this document.

### Confirmed bug found in current `B0._eval_general` / `f_general`

Both `B0._eval_general` (scalar.py ~L191-205) and the numeric
`f_general` (scalar.py ~L241-251) have the term
```python
+ ((m1**2 - m2**2 + p2)/(2*p2)) * np.log(m1**2/m2**2)
```
with the **wrong sign** -- it should be **subtracted**, not added. This is
not merely a small-`p²` numerical-stability issue: it's wrong across the
*entire* domain (confirmed off by orders of magnitude even at `p²=0.2`,
`m1=2, m2=2.7`, nowhere near any degenerate point). Root cause: traces back
to the `m0/m1` vs `m1/m2` argument-labeling mismatch already flagged in
`guide.tex` (see next paragraph) -- under the correct mapping, the guide's
`(m1²−m0²+p²)` term becomes `(m2²−m1²+p²)` in the code's variable names,
opposite sign from what's coded as `(m1²−m2²+p²)`.

**How this was found/verified, worth reusing as a technique:** cross-checked
`f_general`'s output against direct numerical integration
(`scipy.integrate.quad`) of the Feynman-parameter integral
`B0_finite = -∫₀¹ dx ln(χ(x)/μ²)`, `χ(x) = x²p² - x(p²+m1²-m2²) + m1²`
(this integral form is first-principles, from `guide.tex` eq. 197, and
doesn't go through the Källén-function `Λ`/`R` machinery at all, so it's a
genuinely independent reference). With the sign flipped, the two methods
agree to machine precision (`~1e-14`) across a wide mass/`p²` range; as
currently coded they disagree by orders of magnitude. **This
Feynman-integral cross-check is the right way to validate any future
closed-form PV formula** -- prefer it over eyeballing Källén-function
algebra, which is extremely easy to get a sign wrong in (this session hit
that same `m0/m1` sign confusion repeatedly, in the guide's prose, in the
code, and almost in the small-`p²` derivation below).

**Not yet fixed in the file** -- the user is doing a full replacement per
above rather than patching this one line, so don't be surprised if this
exact bug is simply gone (superseded) rather than fixed-in-place next time
you look.

**Also worth checking when the rewrite happens:** `B00`, `dB0_dp2`,
`dB00_dp2` each have their *own* separately-coded `_eval_general`/
`f_general` (not calls into `B0`), which look like they may share the same
copy-pasted-and-mis-signed origin -- don't assume they're fine just because
`B0`'s was fixed.

### `guide.tex` proofreading note (not urgent, cosmetic)

Several equations declare the function signature as `B_0(p²,m_1,m_2)` /
`B_00(p²,m_1,m_2)` but then use `m_0, m_1` inside the body (e.g. guide.tex
lines 299-306, 331-342, 395-402, 411-413) -- an `m_1,m_2` -> `m_0,m_1`
argument-list labeling slip, consistent throughout, not a physics error
once you know to substitute. This is almost certainly what caused/
contributed to the sign confusion in the bug above -- worth cleaning up in
the paper since it's an easy trap for whoever implements from it next
(including future Claude sessions). Minor typos also noted: "inetgrals"
(§3.1 title), "Derivativ" used inconsistently vs "Derivative", "comports"
(probably meant "carries"/"has") -- low priority.

### Two-tier small-`p²` architecture, decided 2026-07-09 — **SUPERSEDED 2026-08-01**

> **Superseded by the Feynman-parameter numerical architecture below.**  
> The algebraic identity derived here remains correct and is recorded as
> a useful cross-check identity, but the two-tier closed-form approach has
> been abandoned in favour of direct numerical integration for all kinematic
> cases. Read the "Primary numerical architecture" section that follows.

Replaces the current file's per-mass-case `F_general`/`F_expanded`
relative-threshold masking (`f_p2_zero_m_non_zero` and its clones in
`B00`/`dB0_dp2`/`dB00_dp2`) with something much simpler: **just two
formulas per function** -- the fully general one, and one safe closed form
for small `p²` -- because the small-`p²` form turns out to already be safe
across *all* the mass-degenerate sub-cases at once, with no further
per-mass branching needed. Verified via sympy (see below); this is a real
result, not a hope.

**The exact identity** (derived and algebraically verified today, `m1,m2`
convention matching the code, not the guide's `m0,m1`):
```
B0(p²,m1,m2) = B0(0,m1,m2) - ∫₀¹ dx · ln( 1 - p²·x(1-x) / χ₀(x) )
```
where `χ₀(x) = χ(x,0) = m1²(1-x) + m2²·x` (a convex combination of the two
masses² along the Feynman parameter path). Derivation: `χ(x,p²) - χ(x,0) =
p²·x(x-1)` exactly (sympy-verified), so `χ(x,p²) = χ₀(x)·[1 -
p²x(1-x)/χ₀(x)]`, and the log of that ratio pulls straight out of the
integral defining `B0_finite`.

**Why this is safe at the mass degeneracies too, not just at generic
masses** (this was the open question raised in this session -- answered,
not just hoped): the ratio `x(1-x)/χ₀(x)` that could in principle cause
trouble is *algebraically* finite at both boundaries:
- `m1 -> m2`: `χ₀(x) -> m1²` (constant) -- no issue, simplest case.
- `m1 -> 0`: `x(1-x)/χ₀(x) -> (1-x)/m2²` -- the `x` cancels exactly against
  `χ₀(x) -> m2²·x`, finite at `x=0` too.
- `m2 -> 0`: symmetric, `(1-x)` cancels, finite at `x=1` too.

So a single small-`p²` numeric implementation of this integral covers the
general-mass, equal-mass, and one-mass-zero cases simultaneously -- no
separate `F_expanded`-style branch needed within the small-`p²` regime.

**Safety margin on `p²` itself:** `log(1-Y)` only breaks when `Y =
p²x(1-x)/χ₀(x) >= 1`. Since `x(1-x) <= 1/4` on `[0,1]` and `χ₀(x) >=
min(m1²,m2²)`, the danger threshold is `p² ≳ 4·min(m1²,m2²)` -- i.e. `p²`
comparable to the *smaller mass²*, not merely "nonzero". Genuine safety
margin for realistic small-`p²` phase-scan use, not a coincidence that
happens to work at machine-epsilon scale.

### Primary numerical architecture: Feynman-parameter integration (decided 2026-08-01)

**All B (and C) function numerical evaluation uses direct Feynman-parameter
quadrature as the primary path.** The closed-form analytical expressions
in `guide.tex §4` are kept as the **symbolic path** (`.reduce()`/`.doit()`)
and as **validation benchmarks** — not as production numerics.

#### Motivation (decided after reviewing the full formula set)

1. **Safety with no kinematic branching:** the Feynman-parameter integral
   is safe for all mass degeneracies (`m0=m1`, `m0=0`, `p²=0`) and for
   general kinematics, with no special-case branching needed. The only
   special treatment is `p² > (m0+m1)²` (above threshold), handled by
   the `iε` prescription.
2. **Consistency with C functions:** the analytic special-case approach
   would need to be repeated for C functions, which is essentially
   impossible (the kinematic degeneracy tree for 3-point functions is
   far more complex). Numerical integration unifies B and C under the
   same approach with no additional work.
3. **Error reduction:** the sign bug already found in the old `scalar.py`
   came from transcribing a complex closed-form. The integral form is
   three lines per function and trivially verified.

#### Architecture

The UV-divergent pole `Δ = 1/ε̄ − ln(m1²/μ²)` is **always kept symbolic**
— it cannot be integrated away and cancels in renormalized physical
quantities. Every function splits as:
```
Bi(p², m0, m1) = (UV pole in Δ) + finite_integral(p², m0, m1)
```
The finite part is computed numerically. UV-finite functions (all
`dBi/dp²`) are computed entirely via quadrature.

Core building block — shared across all functions:
```python
χ(x, p2, m0, m1) = m0²*(1-x) + m1²*x - x*(1-x)*p2
```

Feynman-parameter integrals (finite parts):
```
B0_finite  = -∫₀¹ ln(χ/μ²) dx
B1_finite  =  ∫₀¹ x·ln(χ/μ²) dx      (sign: check guide.tex §3.2)
B11_finite =  ∫₀¹ x²·ln(χ/μ²) dx
B00_finite = -½·∫₀¹ χ·ln(χ/μ²) dx    (from guide.tex §3.2)

dB0/dp²   = -∫₀¹ x(x-1)/χ dx         (UV-finite)
dB1/dp²   =  ∫₀¹ x²(x-1)/χ dx
dB11/dp²  =  ∫₀¹ x³(x-1)/χ dx
dB00/dp²  = -Δ/12 - ½·∫₀¹ x(x-1)·ln(χ/m1²) dx
```

Implementation plan:
- Pre-compute Gauss-Legendre nodes/weights (e.g. `np.polynomial.legendre.leggauss(50)`)
- Vectorize χ evaluation over all nodes simultaneously (`numpy`)
- **Below threshold** (`p² < (m0+m1)²`): χ > 0 on [0,1], pure real
- **Above threshold** (`p² > (m0+m1)²`): use `p2 = p2 + 1e-10j` (iε
  prescription); χ becomes complex, log/inverse handled in complex
  arithmetic; imaginary part of result = physical absorptive part
- All kinematic degeneracies (p²=0, m0=m1, m0=0) work automatically —
  χ(x) remains smooth everywhere, no branching needed

#### Two-tier role going forward

The two-tier architecture (general closed form + small-p² fallback) is
**no longer the production numeric path.** It survives as:
- The symbolic `.reduce()` return value (analytical closed form in
  `g.tex §4` variables — useful for computer algebra, `.doit()`, etc.)
- Cross-check benchmarks for validating the numerical integrals

#### Validation status (updated 2026-08-02)

| Function | Validated | Method | Accuracy |
|---|---|---|---|
| B0 | ✓ done | scipy.quad + analytical ref | ≲10⁻¹² below thr, ~3×10⁻⁸ above |
| B1 | ✓ done | reduction formula + swap identity | ≲10⁻¹⁵ all regimes |
| B11 | ✗ not started | — | — |
| B00 | ✗ not started | — | — |
| dB0/dp² | ✓ done | analytical formula + fin.diff. | ≲10⁻¹⁴ below thr, ≲10⁻¹⁴ above |

The cross-check pattern that found the B0 sign bug: compare GL quadrature
against `scipy.integrate.quad` on the raw Feynman-parameter integral
(first-principles, independent of the closed-form machinery). Use this for
all future function validations before trusting the result.

### Complete formula reference from `guide.tex §4` (verified 2026-07-21)

**`guide.tex` is the authoritative source; read it for full derivations. This section
records the implementation-critical formula map and any correctness notes.**

Conventions: `f = p²+m0²-m1²`, `Δ = 1/ε̄ − ln(m1²/μ²)`, `Λ² = λ(p²,m0²,m1²)`.

#### Non-zero momentum (p²≠0), general case

B0: already documented in the confirmed-bug subsection above (two equivalent forms,
equal-mass β form, one-zero-mass form).

B1 inversion formula and mass-swap identity:
```
B1(p²,m0,m1) = [A0(m1) − A0(m0) + f·B0(p²,m0,m1)] / (2p²)
B1(p²,m1,m0) = B0(p²,m0,m1) − B1(p²,m0,m1)
```

B00 and B11 from solving the 2×2 contraction system (using `d=4` to close):
```
B00 = (1/3)·[p²/6 + (m0²+m1²)/2 + A0(m1)/2 + m0²·B0 − f·B1]
B11 = 1/(3p²)·[−p²/6 − (m0²+m1²)/2 − A0(m1) + m0²·B0 − 2f·B1]
B11(p²,m1,m0) = B11(p²,m0,m1) − 2·B1(p²,m0,m1) + B0(p²,m0,m1)
```

Derivative formulas at non-zero p² (cascade: everything from `dB0/dp²`):
```
dB0/dp²   = (1/p²)·[1 − ((m0²−m1²)²−2p²(m0²+m1²)) / (p²·Λ)]   ← has 1/Λ
dB1/dp²   = (1/p²)·[(m0²−m1²)·B0/2 + f·(dB0/dp²)/2]
dB11/dp²  = (1/p²)·[−5B11/3 − 1/18 + m0²·(dB0/dp²) − 2f·(dB1/dp²)]
dB00/dp²  = (1/3)·[1/6 + m0²·(dB0/dp²) − B1 − f·(dB1/dp²)]
```

**⚠ Threshold issue for `dB0/dp²`:** the `1/Λ` factor diverges at both Källén zeros:
- `p²=(m0+m1)²` (physical threshold): genuine singularity — needs a stable formula.
- `p²=(m0−m1)²` (pseudo-threshold): numerator also→0 for real kinematics, resolves.

`B0` itself and `B1`, `B11`, `B00` are all well-behaved at both thresholds (`R→0`
cleanly at the physical threshold). Only `dB0/dp²` needs a threshold-stable formula;
`dB1/dp²`, `dB11/dp²`, `dB00/dp²` inherit the singularity from `dB0/dp²` algebraically
but introduce no additional independent ones.

Feynman-parameter integral forms (UV-finite pieces; use as independent cross-check,
same technique as the `B0` sign-bug verification above):
```
dB0/dp²   = −∫₀¹ x(x−1)/χ dx
dB1/dp²   =  ∫₀¹ x²(x−1)/χ dx
dB11/dp²  =  ∫₀¹ x³(x−1)/χ dx
dB00/dp²  = −Δ/12 − (1/2)∫₀¹ x(x−1)·ln(χ/m1²) dx

d²B0/d(p²)²   =  ∫₀¹ x²(x−1)²/χ² dx
d²B1/d(p²)²   = −∫₀¹ x³(x−1)²/χ² dx
d²B11/d(p²)²  = −∫₀¹ x⁴(x−1)²/χ² dx
d²B00/d(p²)²  = −(1/2)∫₀¹ x²(x−1)²/χ dx
```
(Note: `guide.tex` labels all four second-order forms as "d²B11/d(p²)²" — likely a
LaTeX copy-paste slip; the integrands follow the expected x^n pattern for B0,B1,B11,B00.)

#### Zero momentum (p²=0): three sub-cases

All values at p²=0 are stable (no 1/p² poles). The derivative `dBi/dp²` at p²=0 can
still carry UV-divergent `Δ` pieces because the p²-dependent `ln`-terms in `Bi` contribute
a `Δ`-like constant when differentiated and then set p²=0.

**Sub-case m0=m1=m:**
```
B0=Δ,  B1=Δ/2,  B11=Δ/3,  B00=(m²/2)(1+Δ)

dB0=1/(6m²),       dB1=Δ/2−1/(12m²),    dB11=Δ/3−1/(20m²),   dB00=−Δ/12
d²B0=1/(30m⁴),    d²B1=1/(60m⁴),        d²B11=1/(105m⁴),      d²B00=−1/(60m⁴)
```

**Sub-case m0=0, m1=m (one zero mass):**
```
B0=Δ+1,  B1=Δ/2+1/4,  B11=Δ/3+1/9,  B00=(m²/2)(3/2+Δ)  ← guide.tex lines 380-384

dB0=1/(2m²),       dB1=Δ/2−1/(6m²),     dB11=Δ/3−1/(12m²),   dB00=−Δ/12−5/72
d²B0=1/(3m⁴),     d²B1=1/(12m⁴),        d²B11=1/(30m⁴),       d²B00=−1/(24m⁴)
```

**Sub-case general m0≠m1:** longer polynomial/log expressions — read `guide.tex §4.2`
and `§4.3` directly. Do not retype here to avoid transcription errors. The general forms
for B0(0), all first derivatives dBi/dp²(0), and all second derivatives d²Bi/d(p²)²(0)
are given there (three separate formula blocks).

#### Taylor expansion switching conditions (`guide.tex §5`)

Switch from the general p²≠0 formula to Taylor expansion when any of:
```
(1) p² < a·max(m0,m1)²                      [small p²]
(2) |p²−(m0+m1)²| < a·max(m0,m1)²          [near physical threshold]
    |p²−(m0−m1)²| < a·max(m0,m1)²          [near pseudo-threshold]
(3) |m0−m1| < a·max(m0,m1)                  [nearly equal masses]
(4) min(m0,m1) < a·max(m0,m1)               [one mass ≪ other]
```
where `a` is a user-adjustable `scale_tolerance` parameter.

**Dimensional note:** `guide.tex §5` writes `a·max(m0,m1)` (units: mass) on the rhs of
conditions (1) and (2) instead of `a·max(m0,m1)²` (units: mass²). This is dimensionally
inconsistent with the lhs (which is mass²); corrected above. Fix in `guide.tex` before
publication.

**Note:** with the Feynman-parameter integration architecture (see above),
conditions (1)–(4) are no longer needed for the *numerical* path — the
integral handles all these cases automatically. These conditions remain
relevant only if/when someone wants a fast analytical approximation for
a specific kinematic region, or for the symbolic `.reduce()` path.

## `A_functions.py` and `B_functions.py` — implementation status (2026-08-02)

### Naming convention (decided 2026-08-02, enforced)

All new PV function files use the **guide.tex convention**: arguments are
`(p2, m0, m1)` where `m0` is the mass on propagator D0 and `m1` on D1.
This makes every formula directly copy-pasteable from guide.tex without
mental translation. `scalar.py` / `tensors.py` use the old `m1/m2` naming;
this is one more reason to retire them.

### `A_functions.py` — complete

| Function | Status | Notes |
|---|---|---|
| `A0(m)` | ✓ complete | symbolic + numeric kernel; massless limit `A0(0)=0` |
| `A00(m)` | ✓ complete | symbolic only (algebraic reduction to A0); `_derivative` returns 0 |
| `A0000(m)` | ✓ complete | symbolic only; not in old `scalar.py` / `tensors.py` |

All three have `_derivative(self, _)` returning `sp.S.Zero`: only p²
derivatives are relevant for PV functions; mass derivatives are not implemented.

### `B_functions.py` — B0 complete, rest to do

#### B0 — fully implemented and validated (2026-08-02)

**Numerical kernel** (`_numeric_kernel`): Feynman-parameter GL quadrature,
N=100 nodes. Two paths:

- **Below threshold** (`p² < (m0+m1)²`): vectorised single GL pass over [0,1]
  with `x = sin²(πt/2)` substitution. Jacobian vanishes at both endpoints,
  regularising log singularities when m0→0 or m1→0. Accuracy: machine
  precision (~1e-15).

- **Above threshold** (`p² > (m0+m1)²`): χ(x) has two zeros x1 < x2 in
  (0,1). Integration is split into [0,x1] + [x1,x2] + [x2,1], each piece
  using the same sin²(πt/2) substitution mapped to the subinterval (so the
  log singularities at x1,x2 are regularised). The imaginary (absorptive)
  part is computed analytically as `π(x2−x1)`. Accuracy: ~3×10⁻⁸.

The helper `_b0_scalar(p2, m0, m1, mu_val, is_above)` handles a single
kinematic point; `_gl_interval(p2, m0, m1, mu_val, a, b)` runs GL on [a,b].

**Symbolic kernel** (`_eval` / `_eval_*`): closed-form expressions from
guide.tex §4, dispatching via `sp.Piecewise` over six kinematic cases:
`(p²=0,m0=0)`, `(p²=0,m1=0)`, `(p²=0,m0=m1)`, `(p²=0,general)`,
`(m0=m1,general)`, `(general)`.

**Validation result** (cross-checked against scipy.integrate.quad):
- Below threshold, all mass configs: agreement to ≲10⁻¹²
- Above threshold, real part: ~3×10⁻⁸ (limited by N=100 GL)
- Above threshold, imaginary part: exact (analytical formula)
- Zero-mass limit: correct via substitution regularisation
- Equal-mass limit: correct
- Small p²: correct to ~10⁻¹²

#### Known bugs in `_eval_general` — found and fixed here, still broken in `scalar.py`

The bug documented in the "Confirmed bug" section below is present in
`scalar.py._eval_general` and `scalar.py.f_general`. In `B_functions.py`
it is **fixed**: the correct formula uses the symmetric form from guide.tex
eq. 318:

```
B0 = 1/ε̄ - ln(m0·m1/μ²) + 2 + (m1²-m0²)/(2p²)·ln(m0²/m1²) - R
```

**Note on guide.tex eq. 311 typo:** eq. 311 writes `(m1²-m0²+p²)/(2p²)` but
the correct coefficient is `(m1²-m0²-p²)/(2p²)` (off by `p²/p² = 1`). Eq.
318 (the symmetric form) is correct and was used as the implementation
reference. This typo in guide.tex should be fixed before publication.

#### `dB0_dp2` — fully implemented and validated (2026-08-02)

UV-finite (no 1/ε̄ pole). `part="pole"` always returns 0.

**Numerical kernel:**

- **Below threshold**: vectorised GL of `∫₀¹ x(1−x)/χ dx` via `x=sin²(πt/2)`.
  Integrand is smooth everywhere (no endpoint singularities even for m=0).
  Accuracy: ~10⁻¹⁴.

- **Above threshold**: χ(x) has simple poles (not log singularities) at x1, x2.
  The iε approach fails (peak height ~1/ε, width ~ε, unresolvable with N=100).
  Instead: **Cauchy-PV pole subtraction** for the real part + exact imaginary part.

  *Real part*: subtract residues from integrand before GL, then add back analytic
  Cauchy-PV terms:
  ```
  Re = GL[x(1-x)/χ - r₁/(x-x₁) - r₂/(x-x₂)] + r₁·ln((1-x₁)/x₁) + r₂·ln((1-x₂)/x₂)
  where rᵢ = xᵢ(1-xᵢ) / χ'(xᵢ),  χ'(x₁)=-√K, χ'(x₂)=+√K
  ```
  The subtracted integrand is smooth on [0,1]. Accuracy: ~10⁻¹⁴.

  *Imaginary part* (exact):
  ```
  Im = π·[(m₀²+m₁²)·p² − (m₀²−m₁²)²] / (p²²·√K)
  ```
  where K = Kallen(p²,m₀²,m₁²). Derived as d/dp²[π(x₂−x₁)]. Accuracy: exact.

**Symbolic kernel** (`_eval` / `_eval_*`): dispatches via `sp.Piecewise` over
six cases (same structure as B0). The p²=0 formulas are UV-finite constants;
the p²≠0 formulas are derived by differentiating the guide.tex eq. 318 B0 form:
```
∂B0/∂p² = −(m₁²−m₀²)/(2p²²)·ln(m₀²/m₁²)
         + [(m₀²+m₁²)p² − (m₀²−m₁²)²]/(p²²·Λ) · ln((m₀²+m₁²−p²+Λ)/(2m₀m₁))
         − 1/p²
```
For equal masses: simplifies to `(2m²/(p²Λ))·ln((2m²−p²+Λ)/(2m²)) − 1/p²`.

**Note on guide.tex dB0/dp² formula:** guide.tex eq. ~425 lists
`(1/p²)[1 − ((m₀²−m₁²)²−2p²(m₀²+m₁²))/(p²·Λ)]` — this is algebraically
**wrong** (neither matches the GL below threshold nor the finite-difference
above threshold). The correct formula is the one derived above by explicit
differentiation of B0 eq. 318. This error should be fixed in guide.tex before
publication.

#### B1 — fully implemented and validated (2026-08-02)

**NOT symmetric under m0↔m1.**  Swap identity (guide.tex eq. 339):
```
B1(p²,m1,m0) = B0(p²,m0,m1) − B1(p²,m0,m1)
```

**Numerical kernel:** `_bfn_scalar(p2, m0, m1, mu_val, is_above, n=1)` —
same split-GL scheme as B0 but integrand weight `x`:
- Below threshold: single GL pass of `-∫₀¹ x·ln(χ/μ²) dx`.
- Above threshold: split at x1,x2; Im = +π(x2²−x1²)/2 added analytically.

The helper `_bfn_scalar(n)` is now general: `n=0`→B0, `n=1`→B1, `n=2`→B11.
`_b0_scalar` now delegates to `_bfn_scalar(n=0)`.
`_gl_interval(p2, m0, m1, mu_val, a, b, n=0)` accepts `n` for weight `x^n`.

**Symbolic kernel (`_eval`):** dispatches over five kinematic cases:
1. `(p²=0, m0=0)`: B1 = Δ(m1)/2 + 1/4
2. `(p²=0, m1=0)`: B1 = Δ(m0)/2 + 3/4  ← via swap identity
3. `(p²=0, m0=m1)`: B1 = Δ(m)/2
4. `(p²=0, general)`: see below
5. `(general, p²≠0)`: reduction formula (guide.tex eq. 333)

**Pole:** 1/2 for all cases.

**p²=0 general formula** (guide.tex eq. 397-398, with Δ = 1/ε̄ − ln(m1²/μ²)):
```
B1(0,m0,m1) = ½/ε̄ − ½ ln(m1²/μ²) + ¼ + m0²/[2(m0²−m1²)] − m0⁴ ln(m0²/m1²)/[2(m0²−m1²)²]
```
This is identical to guide.tex eq. 397-398: the guide writes `−(3m0²−m1²)/(4(m1²−m0²))`
which equals `1/4 + m0²/[2(m0²−m1²)]` (same expression, different form).
Reference mass in Δ is m1, matching guide line 269.

**p²≠0 symbolic** (reduction formula, guide.tex eq. 333):
```
B1(p²,m0,m1) = [A0(m1) − A0(m0) + (p²+m0²−m1²)·B0(p²,m0,m1)] / (2p²)
```

**⚠ guide.tex eq. 397-398 is wrong:** B1(0,m0,m1) formula in the paper
is missing terms (1/2)ln(m0²/m1²) + (3m0²−m1²)/(2(m0²−m1²)).  Verified
numerically (off by ~0.5 for m0=1, m1=2) and via the swap identity.  Fix
in guide.tex before publication.

**⚠ guide.tex line 386 is wrong:** states "B1 and B11 are symmetric under
exchange of internal masses" — they are NOT symmetric (swap identity gives
B1(p²,m1,m0) = B0 − B1, which equals B1(p²,m0,m1) only when B0=2B1).

**Validation:**
- Swap identity `B0(p²,m0,m1) − B1(p²,m0,m1) − B1(p²,m1,m0) = 0`: to ~10⁻¹⁷
- Reduction formula cross-check (GL vs analytical): ≲10⁻¹⁵
- Im = f·Im(B0)/(2p²) above threshold: to ~10⁻¹⁷

#### B11, B00 — not yet started

Next implementation targets. Their Feynman-parameter integrals are:
```
B11_finite = -∫₀¹ x²·ln(χ/μ²) dx    (use _bfn_scalar with n=2)
B00_finite = -(1/2)·∫₀¹ χ·ln(χ/μ²) dx   (pole part needs separate handling)
```
For B11: use `_bfn_scalar(n=2)`, same as B0/B1. Im above threshold = +π(x2³-x1³)/3.
For B00: different structure (weight χ, not x^n), requires its own helper.
Both are NOT symmetric under m0↔m1.

## Roadmap: 3-point functions (planned, not started -- decided 2026-07-09)

**Priority order, explicit:** finish the 2-point sector cleanly first
(implement the Feynman-parameter numerical path for `B0`/`B1`/`B11`/`B00`,
validate against closed forms) *before* starting on 3-point functions.
Don't jump ahead to any of the below until the user says the 2-point work
is done.

**Architecture consistency note (2026-08-01):** the switch to Feynman-parameter
numerical integration for B functions was motivated partly *by* C functions —
having one uniform approach for both is cleaner than a closed-form B path and
a numerical C path. The C tensor coefficient strategy below (Feynman-parameter
direct integration, avoiding Gram determinants) is now the *same philosophy* as
the B function implementation, not a special fallback.

### `C0` (scalar triangle)

User already has the formula (derived by hand, "under my eye"), not yet
implemented. **Revised strategy (2026-08-01):** use direct Feynman-parameter
numerical integration (2D integral over the simplex) as the primary numerical
path, consistent with the B-function approach. The dilogarithm closed form
is still worth implementing as the symbolic `.doit()` path and as a
validation cross-check, but it is no longer the primary numerical route.
Original strategy note preserved: one general closed-form expression built
from dilogarithms, valid across the entire complex kinematic plane via
`+iε` prescription baked into the dilog arguments, **not** case-split by
kinematic region.
This mirrors what Package-X / LoopTools / OneLOop do for `C0`. References
if ever needed (not verified against the user's exact convention, just
pointers): 't Hooft & Veltman (1979, "Scalar One Loop Integrals"), Denner
(2005 review, "Techniques for the calculation of electroweak radiative
corrections"), Ellis & Zanderighi (arXiv:0712.1851, "OneLOop"). Key
implementation caveat: getting the complex branch / `+iε` handling right
in the dilog arguments is what makes one formula valid everywhere --
don't fall back to real-only piecewise formulas the way it doesn't need
to for this function.

### Tensor coefficients (`C1, C2, C11, C12, C22, C001, C002, C111, C112, C122, C222, ...`)

**Two-tier strategy, decided 2026-07-09, do not deviate without asking:**

1. **Fast/readable path, away from the Gram-determinant singularity:**
   standard algebraic Passarino-Veltman reduction to `A0`/`B0`/`B1`/`C0`
   (rational coefficients in the kinematic invariants, involving
   `1/det(Gram)`, where Gram is the 2x2 matrix of dot products of the two
   independent external momenta `q1, q2`). This is the same *kind* of
   "solve a linear system" reduction that was rejected for the 2-point
   tensor functions (see the superseded note above) -- but for 3-point it
   is unavoidable: unlike `B1`/`B11`/`B00`, there is no known
   Gram-determinant-free closed form for the triangle tensor
   coefficients. Accept the `1/det(Gram)` here; it's inherent to the
   physics, not a design mistake.
2. **Robust numeric fallback, near/at small Gram determinant:** do
   **not** attempt the full Denner-Dittmaier small-Gram-determinant
   analytic *expansion* (Nucl. Phys. B658 (2003) 175; also the COLLIER
   paper, Denner-Dittmaier-Hofer 2016) -- that's a research-grade
   undertaking and was explicitly ruled out as out of scope for now.
   Instead: evaluate the tensor integral **directly via the same
   Feynman-parameter construction already used for `C0`, carrying the
   numerator through instead of dropping it**, rather than going through
   the algebraic reduction at all for that kinematic point. This has *no*
   Gram determinant anywhere in its construction, which is exactly why it
   stays robust exactly where the algebraic path blows up:
   - Feynman-parametrize the propagators into `(k+shift(x))² − Δ(x)`,
     where `shift(x) = -(x2*q1 + x3*q2)` is *linear* in the Feynman
     parameters `x_i` and the external momenta `q_i` -- same `Δ(x)` as
     already used for `C0`.
   - Shift `k → k' = k + shift(x)`; expand the numerator
     `(k'-shift(x))^{μ1}...(k'-shift(x))^{μR}` binomially in `k'`. Odd
     powers of `k'` vanish under symmetric `d^dk'` integration; even
     powers give a pure Gamma-function factor depending only on `d`, the
     power of `k'`, and the propagator power -- **no Gram determinant**.
   - What survives is `∫dx1 dx2 dx3 δ(1-x1-x2-x3) [monomial in x2, x3] /
     Δ(x)^power` -- the same `Δ(x)` and integration domain/simplex as
     `C0`, just with an extra polynomial weight (`x2`, `x3`, `x2²`,
     `x2*x3`, etc., depending on which tensor structure) in the
     numerator. A well-defined, always-finite 2D numeric integral,
     structurally identical to the existing `C0` numeric integration --
     just weighted.
   - General-technique reference: Davydychev's papers on N-point tensor
     integral reduction via Feynman-parameter integrals -- an alternative
     to Gram-determinant-based PV algebra specifically because it avoids
     the Gram determinant entirely.
   - The exact monomial-in-`x2,x3` weight for each specific `C_ijk` needs
     to be worked out by hand (mechanical binomial expansion, but
     error-prone) -- the user is doing this themselves, same as the
     2-point closed forms. Don't guess/derive a specific one unprompted.
   - Bonus implication, noted but not yet acted on: since this
     construction has *no* Gram-determinant issue at all, it isn't
     strictly limited to being a "near-singular fallback" -- it's a
     generally-robust numeric method for any tensor coefficient. Keep the
     algebraic reduction as the fast/readable/symbolic path used away
     from the singular region; this Feynman-parameter numeric path is the
     always-correct backstop, decided to be invoked only near/at small
     `det(Gram)` for now (not universally) to keep the fast symbolic path
     as the default.

### Why `.doit()` on tensor `C`-functions is not meant to be human-readable

Fully expanding a tensor coefficient (e.g. `C11`) via the algebraic
reduction above substitutes real `C0`/`B0` closed forms (each already
several dilogarithms) into a linear combination weighted by
`1/det(Gram)` -- the result is a wall of `Li2` terms, not something
anyone reads by eye. This is expected, not a bug: keep `.reduce()`
(stopping at `C0`/`B0`/`A0` as opaque symbols, per the existing
`PVFunction.reduce()` vs `._eval()` split) as the human-facing,
algebra-friendly form; reserve full `.doit()`/numeric evaluation for when
actual numeric kinematics are being plugged in, not for display.

## Quick smoke test

```
python3 -m pvpy.tensorial_decomposition
```

Run from the project root (`PVpy_project/`, the parent of `pvpy/`) -- the
module uses relative imports (`from .algebra import ...`), so invoking it
directly as a script (`python3 pvpy/tensorial_decomposition.py`) fails with
`ImportError: attempted relative import with no known parent package`;
it must be run as a package module with `-m`.

Runs the `__main__` block: A0/A00 (tadpole), B0/B1/B00/B11 (bubble),
k.p1 and k^2 via contraction helpers, C0 and full rank-3 triangle
decomposition (check it has exactly 6 terms: C001,C002,C111,C112,C122,
C222 -- this count is a correctness check, matches standard PV tables),
and the gamma_5 chiral self-energy demo (should print `4*I*eps^{...}`
with the k.k piece having vanished).

For `dirac.py` specifically, `pvpy_tutorial.ipynb` (project root) is the
live regression check -- in particular:
- `DiracTrace(Gamma(mu)*Gamma(nu)*Gamma(rho)*Gamma(sigma))` should give
  `4*g(mu,nu)*g(rho,sigma) - 4*g(mu,rho)*g(nu,sigma) + 4*g(mu,sigma)*g(nu,rho)`
  (bug #7 above).
- `display()`-ing any `DiracMatrix` before tracing it, **including expressions
  containing `Gamma5()`**, should render as readable LaTeX without crashing
  (bug #9 above — previously exploded when γ₅ appeared squared before tracing).
- The V/A trace
  `DiracTrace((slash(p1)+m)*Gamma(mu)*(g_V-g_A*Gamma5())*(slash(p2)-m)*(g_V+g_A*Gamma5())*Gamma(nu))`
  should give only `g_V**2` and `g_A**2` terms (no `g_V*g_A` cross term),
  with the chiral `8i*g_A*g_V*Eps(mu,nu,p1,p2)` term matching Package-X sign
  exactly (GAMMA5_TRACE_COEFF = +4i, settled in convention section above).
- `contract(DiracTrace(expr) * g(mu,nu))` should produce a fully-contracted
  scalar with no residual `g^{munu}**2` or `k^{nu}**2` factors (bug #10).
  Correct order: trace first, then contract.  Passing a `DiracMatrix` directly
  to `contract` silently does the wrong thing (see `dirac.py` API section).
