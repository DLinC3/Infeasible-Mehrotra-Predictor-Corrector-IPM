# barrierQP

A tiny JAX solver for dense, strictly convex quadratic programs with equality
and inequality constraints. It implements Mehrotra's primal-dual
predictor-corrector method: each iteration reduces the four-block Newton
system, factors one KKT matrix, and reuses it for the affine and corrector
directions.

The core solver is 275 lines. A 140-line full-system implementation checks the
eliminated directions, while the central-path example reaches a known boundary
optimum in 6 iterations with 6 factorizations and 12 Newton solves. The same
tests pass on CPU and CUDA.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From the
repository root:

```bash
uv run python central_path.py
```

The example prints one predictor-corrector ledger row per iteration and ends
with:

```text
6 iterations, 6 factorizations, 12 Newton solves
max full/reduced direction difference: 2.09e-15
final KKT residual: 8.47e-10
```

The example asserts the direction agreement, factor/solve counts, strict
positivity, known optimum, and final KKT residual. The final floating-point
digits may vary across devices or separate XLA compilations. On a compatible
NVIDIA system, install the optional CUDA backend with `uv sync --extra cuda`.

## Problem

barrierQP accepts the standard-form quadratic program

```math
\begin{aligned}
\min_x \quad & \frac{1}{2}x^\top P x + q^\top x \\
\text{subject to} \quad & A x = b, \\
                         & G x \leq h.
\end{aligned}
```

where $P \succ 0$. Inequalities use positive slacks $s=h-Gx$, equality duals
$y$, and positive inequality duals $z$. The implementation permits zero
equality rows but requires at least one inequality.

## Main features

- infeasible-start Mehrotra predictor-corrector iterations;
- a full four-block Newton reference and an algebraically reduced solver;
- one LU factorization and two Newton solves per iteration;
- explicit affine prediction, $\mu_{\mathrm{aff}}$, centering parameter
  $\sigma$, corrector, and fraction-to-boundary steps;
- one pure step shared by a compiled `jax.lax.while_loop` and a host-side trace;
- direction-by-direction and trajectory-by-trajectory comparisons with the
  full system, followed by an OSQP final-point comparison.

## Project structure

Read the files in this order:

1. [`reference.py`](reference.py) (140 lines) writes the full four-block Newton
   system directly and performs a fresh dense solve for each direction.
2. [`barrierqp.py`](barrierqp.py) (275 lines) eliminates the slack and
   inequality-dual directions, reuses one LU factorization for two right-hand
   sides, and supplies the compiled and traced solve paths.
3. [`central_path.py`](central_path.py) (77 lines) runs a fixed two-dimensional
   problem with a known boundary optimum and prints the chronological ledger.
4. [`test.py`](test.py) (158 lines) contains the rational first-step
   derivation, direction and trajectory comparisons, JAX checks, positivity
   checks, count checks, and OSQP comparison.

## Design

The design follows Sections 3.1–3.3, “Primal-dual IPM” through “KKT system
solution” (PDF pp. 3–4), of [*HPIPM: a high-performance quadratic programming
framework for model predictive control*](https://arxiv.org/pdf/2003.02547).

## Validation

Run the comparisons on CPU and on the default JAX device:

```bash
JAX_PLATFORMS=cpu uv run python test.py
uv run python test.py
```

The CPU run prints:

```text
audit okay (first iteration matches the exact fraction arithmetic)
directions okay (max full/reduced difference 1.93e-12)
trajectory okay (iterations {'tiny': 5, 'big': 7, 'noeq': 6}, jit/loop/counts agree)
OSQP okay (objectives, solutions, and KKT residuals agree)
all okay
```

The first predictor-corrector iteration is checked against independently
derived rational values. Every predictor and corrector direction and the full
state trajectory are then compared with the four-block reference on three
fixtures, including one with no equalities. The suite also compares eager and
jitted steps, compiled and traced solves, factor/solve counts, strict
positivity, and the final objective, solution, and KKT residuals with OSQP.

## Limitations

- dense, well-scaled QPs with $P\succ0$, independent equality rows, and a
  strict interior only;
- no Phase I method, infeasibility or unboundedness certificate, homogeneous
  embedding, scaling, regularization, or iterative refinement;
- no sparse or OCP structure, cone API, warm start, or alternative
  initialization;
- no autodiff layer, custom GPU kernel, multi-device execution, or sparse
  performance path;
- factorization failures propagate from JAX; the only solver outcomes are
  `solved` and `max_iter`.

## References

- [Mehrotra, *On the Implementation of a Primal-Dual Interior Point
  Method*](https://epubs.siam.org/doi/10.1137/0802028) for the affine predictor,
  centering heuristic, and second-order corrector.
- [Frison and Diehl, *HPIPM*](https://arxiv.org/abs/2003.02547) for the delta
  formulation, slack/dual elimination, residual ledger, and factor reuse.
- [Google DeepMind's QTQP](https://github.com/google-deepmind/qtqp)
  for chronological iteration structure and complementarity diagnostics.
- [OSQP](https://arxiv.org/abs/1711.08013) for the final-point comparison.
- [JAX documentation](https://docs.jax.dev/) and
  [QPAX](https://github.com/qpax-solver/qpax) for factor/solve and compiled-loop
  patterns in JAX.
