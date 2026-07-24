# barrierQP

A tiny JAX solver for dense, strictly convex quadratic programs with equality and
inequality constraints. It implements Mehrotra's primal-dual predictor-corrector
interior-point method, and its one idea is factorization reuse: each iteration
factors a single KKT matrix and reuses it for both Newton directions. This is a
compact educational research implementation, not a production solver.

## Problem

```math
\min_x \ \tfrac{1}{2}x^\top P x + q^\top x \quad\text{s.t.}\quad A x = b,\ \ G x \leq h,
```

with $P \succ 0$. Equality and inequality blocks may have zero rows; the
unconstrained and equality-only cases are handled by a direct KKT solve.

## Method

Each barrier iteration reduces the four-block Newton system, factors **one** KKT
matrix, and reuses that factorization for the affine predictor and the centered
corrector. The centering $\sigma = (\mu_\mathrm{aff}/\mu)^3$ is measured by the
affine step between the two solves. An $N$-iteration solve therefore does exactly
$N$ factorizations and $2N$ Newton solves, and the whole iteration runs inside one
compiled `jax.lax.while_loop`.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import barrierqp; print(barrierqp.Solver)"
```

`uv sync --extra cuda` installs the optional CUDA backend; CPU float64 is the
canonical numerical setting.

## Usage

```python
import numpy as np, barrierqp

solver = barrierqp.Solver(P, q, A, b, G, h)   # arrays; A / G may have zero rows
result = solver.solve()

result.status        # "solved" | "max_iter" | "numerical_error" | "invalid_problem"
result.x, result.y, result.z, result.s
result.iterations, result.factorizations, result.newton_solves
```

## Notebook

[`barrier.ipynb`](barrier.ipynb) solves a small 2-D pentagon, shows the feasible
region, objective contours, and optimum, then re-solves with a different linear
term. A small optional local timing script is included in `bench.py`.

## Limitations

- dense, strictly convex QPs with $P \succ 0$ only;
- no Phase I, infeasibility/unboundedness certificate, homogeneous embedding, or
  warm start; no scaling, regularization, or iterative refinement;
- no sparse or OCP structure, cone API, autodiff, custom GPU kernel, or
  multi-device execution;
- non-finite iterates surface as `numerical_error` and invalid or non-convex data
  as `invalid_problem`; neither is ever returned as a successful solve.

## Status

This is a compact educational research implementation, not a production solver.
The implementation is feature-complete for its intended scope.

## References

- [Mehrotra, *On the Implementation of a Primal-Dual Interior Point Method*](https://epubs.siam.org/doi/10.1137/0802028)
- [Frison and Diehl, *HPIPM*](https://arxiv.org/abs/2003.02547)
- [JAX documentation](https://docs.jax.dev/)
- [qpbenchmark](https://github.com/qpsolvers/qpbenchmark), a broader benchmark
  framework for quadratic-programming solvers.
