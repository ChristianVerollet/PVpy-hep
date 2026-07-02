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
    __init__.py            — A0, B0, B00, A00, B1
    scalar.py              — A0, B0, B00 (primitives)
    tensors.py             — A00, B1 (derived)
```

`algebra.py` was split out of `tensorial_decomposition.py` to give a clean
import layer: `from pvpy.algebra import g, Mom, Dot, Eps, contract`. Old
imports like `from pvpy.tensorial_decomposition import g, Mom` still work
via the re-export.

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
- `B1`, `B11`, `C00`, `C001`, etc. are currently undefined placeholder
  `sympy.Function`s (via `get_pv_function`'s fallback) unless added to
  `PV_REGISTRY`. The user is implementing these themselves in a separate
  module. A nicer long-term approach discussed but not built: derive them
  automatically by contracting the open-tensor ansatz with momenta/metric
  (turning each into a Method-A-computable scalar) and solving the
  resulting linear system -- i.e. `B1`/`B11`/etc. don't need independent
  closed-form derivations, they can be solved for directly from `A0`/`B0`.
  Not implemented; flagged as a good next milestone if the user doesn't
  want to derive them by hand.

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
