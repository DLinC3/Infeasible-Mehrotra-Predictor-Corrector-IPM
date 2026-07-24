"""A tiny JAX implementation of Mehrotra's predictor-corrector QP solver."""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # Must precede any array creation: f32 tolerances would mask real errors.

import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

TAU = 0.99  # fixed fraction-to-boundary damping for accepted steps


class Problem(NamedTuple):  # Fixed arrays form a pytree that jit can traverse.
    P: jax.Array  # (n, n) symmetric positive definite
    q: jax.Array  # (n,)
    A: jax.Array  # (m_eq, n), m_eq may be 0
    b: jax.Array  # (m_eq,)
    G: jax.Array  # (m_in, n), m_in >= 1
    h: jax.Array  # (m_in,)


class State(NamedTuple):  # Primal-dual pytree carried by lax.while_loop.
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    converged: jax.Array  # bool scalar, judged only after a step
    iteration: jax.Array  # int scalar; also the factorization count


class Result(NamedTuple):  # Final solution and factor/solve counts.
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    status: str  # solved | max_iter | numerical_error | invalid_problem
    iterations: int
    factorizations: int  # = iterations on the barrier path; 1 for a direct solve
    newton_solves: int   # = 2 * iterations on the barrier path; 1 for a direct solve


def inf_norm(v):  # Infinity norm that also accepts an empty equality residual.
    return jnp.max(jnp.abs(v), initial=0.0)


def init_state(problem):  # Fixed start with positive slack and inequality dual.
    n, m_eq, m_in = problem.q.shape[0], problem.b.shape[0], problem.h.shape[0]
    return State(x=jnp.zeros(n), y=jnp.zeros(m_eq),
                 z=jnp.ones(m_in), s=jnp.ones(m_in),
                 converged=jnp.asarray(False), iteration=jnp.asarray(0))


def residuals(problem, x, y, z, s):  # KKT residuals and complementarity.
    P, q, A, b, G, h = problem
    r_dual = P @ x + q + A.T @ y + G.T @ z
    r_eq = A @ x - b
    r_ineq = G @ x + s - h
    mu = s @ z / s.shape[0]
    return r_dual, r_eq, r_ineq, mu


def tolerances(problem, x, y, z, s, eps_abs, eps_rel):  # Scaled stop thresholds.
    P, q, A, b, G, h = problem
    eps_p = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(A @ x), inf_norm(b), inf_norm(G @ x), inf_norm(s), inf_norm(h)]))
    eps_d = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(P @ x), inf_norm(q), inf_norm(A.T @ y), inf_norm(G.T @ z)]))
    eps_g = eps_abs + eps_rel * jnp.abs(0.5 * x @ P @ x + q @ x)
    return eps_p, eps_d, eps_g


def stop_test(problem, x, y, z, s, eps_abs, eps_rel):  # Primal, dual, and gap test.
    r_dual, r_eq, r_ineq, _ = residuals(problem, x, y, z, s)
    eps_p, eps_d, eps_g = tolerances(problem, x, y, z, s, eps_abs, eps_rel)
    pres = jnp.maximum(inf_norm(r_eq), inf_norm(r_ineq))
    return (pres <= eps_p) & (inf_norm(r_dual) <= eps_d) & (s @ z <= eps_g)


def form_kkt(problem, D):  # Reduced system after eliminating ds and dz.
    P, _, A, _, G, _ = problem
    m_eq = A.shape[0]
    top = jnp.concatenate([P + G.T @ (D[:, None] * G), A.T], axis=1)
    bottom = jnp.concatenate([A, jnp.zeros((m_eq, m_eq))], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def solve_direction(K, lu, G, z, s, D, r_dual, r_eq, r_ineq, c):  # Shared-LU Newton direction.
    n = r_dual.shape[0]
    w = (c + z * r_ineq) / s
    rhs = jnp.concatenate([-r_dual - G.T @ w, -r_eq])
    sol = lu_solve(lu, rhs)
    dx, dy = sol[:n], sol[n:]
    dz = D * (G @ dx) + w
    ds = -r_ineq - G @ dx
    lin_res = inf_norm(K @ sol - rhs)  # true residual of the trusted LU solve
    return dx, dy, dz, ds, lin_res


def fraction_to_boundary(v, dv, tau):  # Keep v + alpha*dv strictly positive.
    ratios = jnp.where(dv < 0, -v / dv, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def step(problem, state, eps_abs, eps_rel):  # Pure Mehrotra step ready for jit.
    x, y, z, s = state.x, state.y, state.z, state.s
    m_in = problem.h.shape[0]

    r_dual, r_eq, r_ineq, mu = residuals(problem, x, y, z, s)

    D = z / s
    K = form_kkt(problem, D)
    lu = lu_factor(K)  # the one factorization; both Newton questions below reuse it

    # Question 1, affine predictor: target = 0, correction = 0.
    dx_aff, dy_aff, dz_aff, ds_aff, _ = solve_direction(
        K, lu, problem.G, z, s, D, r_dual, r_eq, r_ineq, -(s * z))
    alpha_aff_primal = fraction_to_boundary(s, ds_aff, 1.0)
    alpha_aff_dual = fraction_to_boundary(z, dz_aff, 1.0)

    # Mehrotra centering: predicted complementarity decides sigma, between the two solves.
    mu_aff = (s + alpha_aff_primal * ds_aff) @ (z + alpha_aff_dual * dz_aff) / m_in
    sigma = jnp.clip((mu_aff / mu) ** 3, 0.0, 1.0)

    # Question 2, centered corrector: target = sigma*mu, correction = ds_aff*dz_aff.
    c_corr = -(s * z) + sigma * mu - ds_aff * dz_aff
    dx, dy, dz, ds, _ = solve_direction(
        K, lu, problem.G, z, s, D, r_dual, r_eq, r_ineq, c_corr)
    alpha_primal = fraction_to_boundary(s, ds, TAU)
    alpha_dual = fraction_to_boundary(z, dz, TAU)

    x_new = x + alpha_primal * dx
    s_new = s + alpha_primal * ds
    y_new = y + alpha_dual * dy
    z_new = z + alpha_dual * dz

    # Stopping is judged only on the newly accepted iterate, never the raw start.
    done = stop_test(problem, x_new, y_new, z_new, s_new, eps_abs, eps_rel)
    return State(x_new, y_new, z_new, s_new, done, state.iteration + 1)


def _solve_loop(problem, state, eps_abs, eps_rel, max_iter):  # Fixed-shape device loop.
    def body(st):
        return step(problem, st, eps_abs, eps_rel)

    def cond(st):
        return jnp.logical_and(~st.converged, st.iteration < max_iter)

    # while_loop keeps the whole iteration on device; every carried array must
    # hold one fixed shape and dtype from first iteration to last.
    return jax.lax.while_loop(cond, body, state)


# The pure loop is staged once at import; re-tracing on every call would defeat
# compilation. The whole IPM iteration runs inside this single compiled program.
_solve_loop_jit = jax.jit(_solve_loop)


def _validate(problem, eps_abs, eps_rel, max_iter):  # Host-boundary problem check.
    P, q, A, b, G, h = problem
    if q.ndim != 1:
        return "q must be a vector"
    n = q.shape[0]
    if P.shape != (n, n):
        return "P must be square of side len(q)"
    if A.ndim != 2 or A.shape[1] != n:
        return "A must be (m_eq, n)"
    if b.shape != (A.shape[0],):
        return "b must have one entry per equality row"
    if G.ndim != 2 or G.shape[1] != n:
        return "G must be (m_in, n)"
    if h.shape != (G.shape[0],):
        return "h must have one entry per inequality row"
    if not all(bool(jnp.all(jnp.isfinite(v))) for v in problem):
        return "problem data must be finite"
    # Scale-aware symmetry: |P - P.T| <= atol + rtol*|P.T| tolerates float64
    # roundoff asymmetry while rejecting a materially non-symmetric matrix.
    if not bool(jnp.allclose(P, P.T, rtol=1e-8, atol=1e-12)):
        return "P must be symmetric"
    # Strict convexity is a precondition, not something the iteration detects:
    # a failed Cholesky (NaN factor) means P is not meaningfully positive definite
    # (this rejects indefinite and numerically singular P). Its O(n^3) cost is
    # part of construction and is included in the benchmark's setup timing.
    if not bool(jnp.all(jnp.isfinite(jnp.linalg.cholesky(P)))):
        return "P must be positive definite"
    if not (eps_abs > 0 and eps_rel >= 0 and max_iter >= 1):
        return "eps_abs>0, eps_rel>=0, max_iter>=1 required"
    return None


def _direct_solve(problem):  # Unconstrained / equality-only: one dense KKT solve.
    # With no inequalities there is no barrier, no slack, and no dual z; the
    # optimum is the single stationary point of [[P, A^T],[A, 0]] [x; y] = [-q; b].
    P, q, A, b, G, h = problem
    n, m_eq = q.shape[0], b.shape[0]
    K = jnp.block([[P, A.T], [A, jnp.zeros((m_eq, m_eq))]])
    rhs = jnp.concatenate([-q, b])
    sol = lu_solve(lu_factor(K), rhs)  # the single factorization of this path
    x, y = sol[:n], sol[n:]
    rel_res = inf_norm(K @ sol - rhs) / (1.0 + inf_norm(rhs))
    return x, y, jnp.zeros(0), jnp.zeros(0), rel_res


def _result(state):  # Convert device state of the barrier path to a host result.
    iterations = int(state.iteration)  # host conversion; also synchronizes the device
    # A blown-up KKT factor propagates NaN/Inf silently; never call that solved.
    finite = bool(jnp.all(jnp.isfinite(state.x)) & jnp.all(jnp.isfinite(state.y))
                  & jnp.all(jnp.isfinite(state.z)) & jnp.all(jnp.isfinite(state.s)))
    if not finite:
        status = "numerical_error"
    elif bool(state.converged):
        status = "solved"
    else:
        status = "max_iter"
    return Result(x=state.x, y=state.y, z=state.z, s=state.s, status=status,
                  iterations=iterations, factorizations=iterations,
                  newton_solves=2 * iterations)


class Solver:  # Host API around the pure JAX functions above.

    def __init__(self, P, q, A, b, G, h, eps_abs=1e-8, eps_rel=1e-8,
                 max_iter=50):  # Convert and validate fixed problem data.
        self.problem = Problem(*(jnp.asarray(v, dtype=jnp.float64)
                                 for v in (P, q, A, b, G, h)))
        # Plain host floats/ints: passed to jit as dynamic operands, so
        # changing a tolerance or the cap never triggers recompilation.
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)
        # Invalid data is surfaced as a returned status, not an exception, so
        # callers always get a Result to inspect.
        self._invalid_reason = _validate(self.problem, self.eps_abs,
                                         self.eps_rel, self.max_iter)

    def _invalid_result(self):  # Explicit failure carrying no fake solution.
        p = self.problem
        n, m_eq, m_in = p.q.shape[0], p.b.shape[0], p.h.shape[0]
        nan = jnp.nan
        return Result(x=jnp.full(n, nan), y=jnp.full(m_eq, nan),
                      z=jnp.full(m_in, nan), s=jnp.full(m_in, nan),
                      status="invalid_problem", iterations=0,
                      factorizations=0, newton_solves=0)

    def _direct_result(self):  # No-inequality case: report the direct solve.
        x, y, z, s, rel_res = _direct_solve(self.problem)
        ok = (bool(jnp.all(jnp.isfinite(x))) and bool(jnp.all(jnp.isfinite(y)))
              and float(rel_res) < 1e-6)  # a singular KKT leaves a large residual
        return Result(x=x, y=y, z=z, s=s,
                      status="solved" if ok else "numerical_error",
                      iterations=1, factorizations=1, newton_solves=1)

    def solve(self):  # Run the compiled lax.while_loop (or the direct solve).
        if self._invalid_reason is not None:
            return self._invalid_result()
        if self.problem.h.shape[0] == 0:  # no inequalities: no barrier to iterate
            return self._direct_result()
        state = init_state(self.problem)
        state = _solve_loop_jit(self.problem, state, self.eps_abs,
                                self.eps_rel, self.max_iter)
        return _result(state)
