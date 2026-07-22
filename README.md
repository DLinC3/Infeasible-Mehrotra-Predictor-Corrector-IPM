# barrierQP

**One KKT factorization, two Newton questions.**

barrierQP is a tiny JAX primal-dual interior-point method for dense, strictly
convex QPs. Each iteration factors one reduced KKT matrix, asks it for an
affine predictor and a centered corrector, then takes the largest safe step.
I wanted Mehrotra's loop to be visible end to end, including its residual
ledger and centering controller.

## feel the magic

```bash
uv sync
uv run python central_path.py
```

A representative CUDA run prints:

```text
iter   primal      dual        mu         mu_aff      sigma      alpha_pri  alpha_dual  linear_res
   0  0.000e+00  2.075e+00  1.000e+00  2.115e-01  9.467e-03   1.000000   0.982182  4.441e-16
   1  0.000e+00  4.264e-03  1.003e-01  1.555e-02  3.728e-03   0.846850   0.989182  1.110e-16
   2  1.110e-16  3.698e-02  6.855e-03  2.609e-04  5.516e-05   1.000000   0.980775  3.886e-16
   3  0.000e+00  8.427e-04  1.496e-04  8.865e-07  2.079e-07   0.990003   0.989977  4.441e-16
   4  0.000e+00  8.466e-06  1.503e-06  9.223e-11  2.310e-13   0.990000   0.990000  4.441e-16
   5  0.000e+00  8.466e-08  1.503e-08  9.224e-15  2.312e-19   0.990000   0.990000  2.220e-16
6 iterations, 6 factorizations, 12 Newton solves
max full/reduced direction difference: 2.42e-15
final KKT residual: 8.47e-10
```

The demo asserts every claim before printing it — the factor/solve counts, the
agreement of the reduced directions with a full-system reference, strict
positivity of every recorded slack-dual pair, and the final KKT residual.
The last few floating-point digits can vary across devices or separate XLA
compilations; the assertions use tolerances, never bitwise equality.

## the files that matter

Read them in this order:

1. [`reference.py`](reference.py) (140 lines) — the full four-block delta-form
   Newton system, eager JAX, one fresh dense solve per direction;
2. [`barrierqp.py`](barrierqp.py) (275 lines) — the same iteration after
   eliminating `(ds, dz)`: one LU factorization, two right-hand sides, visible
   centering, and a compiled fixed-shape loop;
3. [`central_path.py`](central_path.py) (77 lines) — the fixed 2D experiment
   and its chronological predictor-corrector ledger;
4. [`test.py`](test.py) (158 lines) — an exact-rational first-iteration audit,
   direction and trajectory parity, and the OSQP oracle.

## how it works

The problem is the standard-form strictly convex QP

```text
minimize    1/2 x.T P x + q.T x
subject to  A x = b
            G x <= h        with slack s = h - G x > 0 and dual z > 0.
```

An iterate `(x, y, z, s)` is scored by four KKT residuals:

```text
r_dual = P x + q + A.T y + G.T z
r_eq   = A x - b
r_ineq = G x + s - h
mu     = s.T z / m_in
```

Newton's method for the perturbed KKT conditions solves the four-block system

```text
[P  A.T  G.T  0] [dx]   [-r_dual]
[A   0    0   0] [dy] = [-r_eq  ]
[G   0    0   I] [dz]   [-r_ineq]
[0   0    S   Z] [ds]   [   c   ]     c = -s*z + target - correction
```

which `reference.py` solves literally, afresh, for every direction. The last
two block rows are diagonal, so `barrierqp.py` eliminates `ds` and `dz` by
elementwise division and keeps only the reduced matrix

```text
D = z / s
K = [P + G.T diag(D) G   A.T]
    [A                    0 ]
```

The elimination stays explicit in `solve_direction`: `w = (c + z*r_ineq)/s`,
solve `K [dx; dy] = [-r_dual - G.T w; -r_eq]`, then `dz = D*(G dx) + w` and
`ds = -r_ineq - G dx`. Each iteration of `step` runs in execution order:

```text
residuals -> factor K once -> affine RHS   (target 0,        correction 0)
          -> mu_aff -> sigma = (mu_aff/mu)^3
          -> corrector RHS (target sigma*mu, correction ds_aff*dz_aff)
          -> fraction to boundary -> next strictly interior iterate
```

The same LU factors answer both right-hand sides; that is the entire point.
The affine solve asks "where would pure Newton go?", `mu_aff` measures how
much complementarity that step would keep, `sigma` converts the answer into a
centering weight, and the corrector solve asks the same factorization for the
direction actually taken.

**JAX in this file.** `jnp` arrays are immutable float64 (enabled at import,
[`barrierqp.py:20`](barrierqp.py#L20)), so a step returns a new `State` instead
of mutating one. The step is a pure function; `jax.jit` stages it once at
module scope ([`barrierqp.py:232`](barrierqp.py#L232)), and `solve` wraps it in
a `lax.while_loop` ([`barrierqp.py:227`](barrierqp.py#L227)) whose carried
arrays keep one fixed shape and dtype from first iteration to last, so the
whole solve stays on device (CPU or GPU, same program). The explanatory
`solve_trace` drives the identical jitted step from a Python loop; converting
`state.converged` to a Python `bool`
([`barrierqp.py:271`](barrierqp.py#L271)) is the explicit device-to-host
synchronization point. Printing happens only after that boundary. JAX is the
trusted ground here, not the subject: it does not make this IPM differentiable,
and no gradient of the solver is defined or claimed.

## correctness

```bash
JAX_PLATFORMS=cpu uv run python test.py
uv run python test.py
```

```text
audit okay (first iteration matches the exact fraction arithmetic)
directions okay (max full/reduced difference 1.93e-12)
trajectory okay (iterations {'tiny': 5, 'big': 7, 'noeq': 6}, jit/loop/counts agree)
OSQP okay (objectives, solutions, and KKT residuals agree)
all okay
```

The default-device run prints the same ladder (direction difference
`2.64e-12` on this machine's GPU). The first rung compares the first
iteration against exact `fractions.Fraction` arithmetic, hard-coded from an
independent derivation. The middle rungs are the strongest evidence: the
reduced one-factor/two-RHS path must reproduce the full four-block reference
*direction by direction, iteration by iteration* — a solver that reached a
similar final `x` through a sign error and a compensating mistake would fail
them. OSQP only certifies the final KKT point; the factor and solve counts
(`N` and `2N`) are checked as explicit bookkeeping.

## research roots

- [Mehrotra, *On the Implementation of a Primal-Dual Interior Point
  Method*](https://epubs.siam.org/doi/10.1137/0802028) — affine predictor,
  centering heuristic, and second-order corrector;
- [Frison and Diehl, *HPIPM*](https://arxiv.org/abs/2003.02547) — the reliable
  delta formulation, slack/dual elimination, the residual ledger, and the
  one-factor/multiple-RHS implementation perspective;
- [Google DeepMind's QTQP implementation at the inspected
  commit](https://github.com/google-deepmind/qtqp/tree/2d28ec9b019448c3621527a16930dbaccb5ddc8b)
  — the chronological pure-Python loop and per-iteration complementarity
  diagnostics, not its homogeneous embedding, backend surface, scaling, or
  certificates;
- [OSQP](https://arxiv.org/abs/1711.08013) — external final-point oracle only;
- [JAX documentation](https://docs.jax.dev/) and
  [QPAX explicit PDIP at the inspected
  commit](https://github.com/qpax-solver/qpax/tree/a014149ffc632d5ea0b1090f98d53e16e874c5c4)
  — public factor/solve primitives and fixed-shape compiled-loop patterns, not
  QPAX's alternate backend, elastic problems, f32 rescue logic, differentiable
  layer, or package surface.

## this is not HPIPM

The scope is deliberately narrow: dense, well-scaled, strictly convex `P`,
independent equality rows, and a strict interior are assumed, and violated
assumptions surface as assertion or linear-algebra errors, never as a solver
status other than `solved` or `max_iter`. There is no Phase I, no
infeasibility certificate, no homogeneous embedding, no scaling, no
regularization or iterative refinement, no sparse or OCP structure, no warm
start, no cone API, no autodiff, no custom GPU kernels, no multi-device
execution, and no production guarantee. CPU/GPU portability comes from the
same JAX program, not from two solver backends.

## license

No license file yet — choose one before publishing.
