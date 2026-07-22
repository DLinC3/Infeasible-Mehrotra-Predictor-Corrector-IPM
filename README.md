# barrierQP

barrierQP solves dense, strictly convex quadratic programs with equality and
inequality constraints using Mehrotra's primal-dual predictor-corrector method.
It is intended for readers who want a compact, modifiable implementation of
interior-point iterations, KKT elimination, and factorization reuse in JAX.
The solver, full-system reference, correctness tests, and central-path example
are complete and have been checked on CPU and CUDA; this is an educational
research implementation, not a production solver.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python central_path.py
```

The example prints one predictor-corrector ledger row per iteration. A
representative CPU run ends with:

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

$$
\begin{aligned}
\underset{x}{\operatorname{minimize}} \quad
    & \frac{1}{2}x^\top P x + q^\top x \\
\operatorname{subject\ to} \quad
    & A x = b, \\
    & G x \leq h,
\end{aligned}
$$

where $P \succ 0$. Inequalities use slacks $s=h-Gx>0$, equality duals $y$,
and inequality duals $z>0$. The implementation permits zero equality rows but
requires at least one inequality.

## Main features

- infeasible-start Mehrotra predictor-corrector iterations;
- a full four-block Newton reference and an algebraically reduced solver;
- one LU factorization and two Newton solves per iteration;
- explicit affine prediction, $\mu_{\mathrm{aff}}$, centering parameter
  $\sigma$, corrector, and fraction-to-boundary steps;
- one pure step shared by a compiled `jax.lax.while_loop` and a host-side trace;
- direction-by-direction and trajectory-by-trajectory validation against the
  reference, followed by an OSQP final-point check.

## Project structure

Read the files in this order:

1. [`reference.py`](reference.py) (140 lines) writes the full four-block Newton
   system directly and performs a fresh dense solve for each direction.
2. [`barrierqp.py`](barrierqp.py) (275 lines) eliminates the slack and
   inequality-dual directions, reuses one LU factorization for two right-hand
   sides, and supplies the compiled and traced solve paths.
3. [`central_path.py`](central_path.py) (77 lines) runs a fixed two-dimensional
   problem with a known boundary optimum and prints the chronological ledger.
4. [`test.py`](test.py) (158 lines) contains the rational first-step audit,
   direction and trajectory comparisons, JAX checks, positivity checks, count
   checks, and OSQP oracle.

## Design

At an iterate $(x,y,z,s)$, the KKT residuals and average complementarity are

$$
\begin{aligned}
r_{\mathrm{dual}} &= Px+q+A^\top y+G^\top z, \\
r_{\mathrm{eq}} &= Ax-b, \\
r_{\mathrm{ineq}} &= Gx+s-h, \\
\mu &= \frac{s^\top z}{m_{\mathrm{ineq}}}.
\end{aligned}
$$

For a complementarity right-hand side $c$, the reference solves

$$
\begin{bmatrix}
P & A^\top & G^\top & 0 \\
A & 0 & 0 & 0 \\
G & 0 & 0 & I \\
0 & 0 & S & Z
\end{bmatrix}
\begin{bmatrix}
\Delta x \\ \Delta y \\ \Delta z \\ \Delta s
\end{bmatrix}
=
\begin{bmatrix}
-r_{\mathrm{dual}} \\
-r_{\mathrm{eq}} \\
-r_{\mathrm{ineq}} \\
c
\end{bmatrix},
$$

where $S=\operatorname{diag}(s)$ and $Z=\operatorname{diag}(z)$. The optimized
path eliminates $\Delta s$ and $\Delta z$. Define

$$
D=\operatorname{diag}(z\oslash s),
\qquad
K=
\begin{bmatrix}
P+G^\top D G & A^\top \\
A & 0
\end{bmatrix}.
$$

For each Newton right-hand side, it computes

$$
\begin{aligned}
w &= (c+z\odot r_{\mathrm{ineq}})\oslash s, \\
K
\begin{bmatrix}\Delta x \\ \Delta y\end{bmatrix}
&=
\begin{bmatrix}
-r_{\mathrm{dual}}-G^\top w \\
-r_{\mathrm{eq}}
\end{bmatrix}, \\
\Delta z &= D G\Delta x+w, \\
\Delta s &= -r_{\mathrm{ineq}}-G\Delta x.
\end{aligned}
$$

Here $\odot$ and $\oslash$ denote elementwise multiplication and division. The
same LU factors of $K$ answer two questions in chronological order:

$$
\begin{aligned}
c_{\mathrm{aff}} &= -s\odot z, \\
\mu_{\mathrm{aff}}
    &= \frac{(s+\alpha_{\mathrm{aff}}^{p}\Delta s_{\mathrm{aff}})^\top
              (z+\alpha_{\mathrm{aff}}^{d}\Delta z_{\mathrm{aff}})}
             {m_{\mathrm{ineq}}}, \\
\sigma &= \operatorname{clip}
          \left(\left(\frac{\mu_{\mathrm{aff}}}{\mu}\right)^3,0,1\right), \\
c_{\mathrm{corr}}
    &= -s\odot z+\sigma\mu\mathbf{1}
       -\Delta s_{\mathrm{aff}}\odot\Delta z_{\mathrm{aff}}.
\end{aligned}
$$

The accepted primal and dual steps use the fraction-to-boundary rule

$$
\alpha(v,\Delta v;\tau)
=\min\!\left(1,
  \tau\min_{i:\,\Delta v_i<0}\frac{-v_i}{\Delta v_i}\right),
\qquad \tau=0.99,
$$

applied separately to $(s,\Delta s)$ and $(z,\Delta z)$. This preserves strict
positivity of every accepted slack and inequality dual.

Termination checks original-coordinate absolute-plus-relative tolerances for
$\max(\lVert r_{\mathrm{eq}}\rVert_\infty,
\lVert r_{\mathrm{ineq}}\rVert_\infty)$,
$\lVert r_{\mathrm{dual}}\rVert_\infty$, and $s^\top z$.

### JAX implementation

The module enables float64 before creating arrays
([`barrierqp.py:20`](barrierqp.py#L20)). `step` is a pure function: it forms
one reduced KKT matrix, calls `lu_factor` once, and calls `lu_solve` twice with
the same factor/pivot tuple. The transform is created once at module scope
([`barrierqp.py:232`](barrierqp.py#L232)).

The normal solve runs the step in a fixed-shape `lax.while_loop`
([`barrierqp.py:227`](barrierqp.py#L227)). `solve_trace` calls the identical
jitted step from a Python loop and converts the stop flag to a Python `bool` at
the explicit device-to-host boundary ([`barrierqp.py:272`](barrierqp.py#L272)).
No differentiation rule for the solver is defined or claimed.

## Validation

Run the compact correctness suite on CPU and on the default JAX device:

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
- no autodiff layer, custom GPU kernel, multi-device execution, or production
  performance guarantee;
- factorization failures propagate from JAX; the only solver outcomes are
  `solved` and `max_iter`.

No correctness issue is known for the documented problem class. Behavior near
degenerate solutions or outside the stated rank and scaling assumptions is not
characterized as a production solver status.

## References

- [Mehrotra, *On the Implementation of a Primal-Dual Interior Point
  Method*](https://epubs.siam.org/doi/10.1137/0802028) for the affine predictor,
  centering heuristic, and second-order corrector.
- [Frison and Diehl, *HPIPM*](https://arxiv.org/abs/2003.02547) for the delta
  formulation, slack/dual elimination, residual ledger, and factor reuse.
- [Google DeepMind's QTQP implementation at the inspected
  commit](https://github.com/google-deepmind/qtqp/tree/2d28ec9b019448c3621527a16930dbaccb5ddc8b)
  for chronological iteration structure and complementarity diagnostics.
- [OSQP](https://arxiv.org/abs/1711.08013) as the external final-point oracle.
- [JAX documentation](https://docs.jax.dev/) and
  [QPAX explicit PDIP at the inspected commit](https://github.com/qpax-solver/qpax/tree/a014149ffc632d5ea0b1090f98d53e16e874c5c4)
  for public factor/solve and compiled-loop patterns.

## License

No license file is currently included. Choose a license before redistribution
or publication.
