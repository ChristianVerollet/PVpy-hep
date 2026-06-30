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
  - Exactly one `G5` per term is handled; an even number should cancel via
    `g5^2=1` but this is NOT auto-detected (raises `NotImplementedError`,
    not a silent wrong answer -- intentional, don't "fix" by guessing).
  - Exactly 4 gamma matrices after stripping `G5` is the only nonzero case
    implemented (0-3 correctly returns 0; 6+ needs the general identity
    with extra `g.eps` terms -- not implemented).
  - **NOT YET WIRED IN:** `ReduceGeneralNumerator` does not know what to do
    with an `Eps(...)` factor in a numerator term yet. This is the
    natural next piece if/when full automation through to PV functions is
    wanted for chiral diagrams.

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
- `Eps` is not yet consumed by `ReduceGeneralNumerator` (see gamma_5
  section above).
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
python3 tensorial_decomposition.py
```

Runs the `__main__` block: A0/A00 (tadpole), B0/B1/B00/B11 (bubble),
k.p1 and k^2 via contraction helpers, C0 and full rank-3 triangle
decomposition (check it has exactly 6 terms: C001,C002,C111,C112,C122,
C222 -- this count is a correctness check, matches standard PV tables),
and the gamma_5 chiral self-energy demo (should print `4*I*eps^{...}`
with the k.k piece having vanished).
