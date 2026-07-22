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


class StepTrace(NamedTuple):  # Named values exposed from one iteration.
    iteration: jax.Array
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    r_eq: jax.Array
    r_ineq: jax.Array
    r_dual: jax.Array
    mu: jax.Array
    eps_primal: jax.Array
    eps_dual: jax.Array
    eps_gap: jax.Array
    dx_aff: jax.Array
    dy_aff: jax.Array
    dz_aff: jax.Array
    ds_aff: jax.Array
    alpha_aff_primal: jax.Array
    alpha_aff_dual: jax.Array
    mu_aff: jax.Array
    sigma: jax.Array
    dx: jax.Array
    dy: jax.Array
    dz: jax.Array
    ds: jax.Array
    alpha_primal: jax.Array
    alpha_dual: jax.Array
    linear_residual: jax.Array


class Trace(NamedTuple):  # Step records stacked along the iteration axis.
    iteration: jax.Array
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    r_eq: jax.Array
    r_ineq: jax.Array
    r_dual: jax.Array
    mu: jax.Array
    eps_primal: jax.Array
    eps_dual: jax.Array
    eps_gap: jax.Array
    dx_aff: jax.Array
    dy_aff: jax.Array
    dz_aff: jax.Array
    ds_aff: jax.Array
    alpha_aff_primal: jax.Array
    alpha_aff_dual: jax.Array
    mu_aff: jax.Array
    sigma: jax.Array
    dx: jax.Array
    dy: jax.Array
    dz: jax.Array
    ds: jax.Array
    alpha_primal: jax.Array
    alpha_dual: jax.Array
    linear_residual: jax.Array


class Result(NamedTuple):  # Final solution and factor/solve counts.
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    status: str  # "solved" or "max_iter", nothing else
    iterations: int
    factorizations: int  # = iterations: the step factors exactly once
    newton_solves: int   # = 2 * iterations: affine + corrector per factor


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
    eps_p, eps_d, eps_g = tolerances(problem, x, y, z, s, eps_abs, eps_rel)

    D = z / s
    K = form_kkt(problem, D)
    lu = lu_factor(K)  # the one factorization; both Newton questions below reuse it

    # Question 1, affine predictor: target = 0, correction = 0.
    c_aff = -(s * z)
    dx_aff, dy_aff, dz_aff, ds_aff, _ = solve_direction(
        K, lu, problem.G, z, s, D, r_dual, r_eq, r_ineq, c_aff)
    alpha_aff_primal = fraction_to_boundary(s, ds_aff, 1.0)
    alpha_aff_dual = fraction_to_boundary(z, dz_aff, 1.0)

    # Mehrotra centering: predicted complementarity decides sigma, visibly
    # between the two solves.
    mu_aff = (s + alpha_aff_primal * ds_aff) @ (z + alpha_aff_dual * dz_aff) / m_in
    sigma = jnp.clip((mu_aff / mu) ** 3, 0.0, 1.0)

    # Question 2, centered corrector: target = sigma*mu, correction = ds_aff*dz_aff.
    c_corr = -(s * z) + sigma * mu - ds_aff * dz_aff
    dx, dy, dz, ds, lin_res = solve_direction(
        K, lu, problem.G, z, s, D, r_dual, r_eq, r_ineq, c_corr)
    alpha_primal = fraction_to_boundary(s, ds, TAU)
    alpha_dual = fraction_to_boundary(z, dz, TAU)

    x_new = x + alpha_primal * dx
    s_new = s + alpha_primal * ds
    y_new = y + alpha_dual * dy
    z_new = z + alpha_dual * dz

    # Stopping is judged only on the newly accepted iterate, never the raw start.
    done = stop_test(problem, x_new, y_new, z_new, s_new, eps_abs, eps_rel)
    new_state = State(x_new, y_new, z_new, s_new, done, state.iteration + 1)
    trace = StepTrace(
        iteration=state.iteration, x=x, y=y, z=z, s=s,
        r_eq=r_eq, r_ineq=r_ineq, r_dual=r_dual, mu=mu,
        eps_primal=eps_p, eps_dual=eps_d, eps_gap=eps_g,
        dx_aff=dx_aff, dy_aff=dy_aff, dz_aff=dz_aff, ds_aff=ds_aff,
        alpha_aff_primal=alpha_aff_primal, alpha_aff_dual=alpha_aff_dual,
        mu_aff=mu_aff, sigma=sigma,
        dx=dx, dy=dy, dz=dz, ds=ds,
        alpha_primal=alpha_primal, alpha_dual=alpha_dual,
        linear_residual=lin_res)
    return new_state, trace


def _solve_loop(problem, state, eps_abs, eps_rel, max_iter):  # Fixed-shape device loop.
    def body(st):
        new_st, _ = step(problem, st, eps_abs, eps_rel)  # same step; the trace is unused here
        return new_st

    def cond(st):
        return jnp.logical_and(~st.converged, st.iteration < max_iter)

    # while_loop keeps the whole iteration on device; every carried array must
    # hold one fixed shape and dtype from first iteration to last.
    return jax.lax.while_loop(cond, body, state)


# Transforms are created once at import, not per solve: jit(step) stages the pure
# step for the given shapes, and re-tracing on every call would defeat compilation.
_step = jax.jit(step)
_solve_loop_jit = jax.jit(_solve_loop)


def _check_problem(problem, eps_abs, eps_rel, max_iter):  # Validate the supported QP form.
    P, q, A, b, G, h = problem
    n, m_eq, m_in = q.shape[0], b.shape[0], h.shape[0]
    assert P.shape == (n, n) and A.shape == (m_eq, n) and G.shape == (m_in, n)
    assert m_in >= 1, "barrierQP needs at least one inequality"
    assert bool(jnp.allclose(P, P.T)), "P must be symmetric"
    assert eps_abs > 0 and eps_rel >= 0 and max_iter >= 1


def _result(state):  # Convert device state to the host-facing result.
    iterations = int(state.iteration)  # host conversion; also synchronizes the device
    return Result(x=state.x, y=state.y, z=state.z, s=state.s,
                  status="solved" if bool(state.converged) else "max_iter",
                  iterations=iterations, factorizations=iterations,
                  newton_solves=2 * iterations)


class Solver:  # Host API around the pure JAX functions above.

    def __init__(self, P, q, A, b, G, h, eps_abs=1e-8, eps_rel=1e-8,
                 max_iter=50):  # Convert and validate fixed problem data.
        problem = Problem(*(jnp.asarray(v, dtype=jnp.float64)
                            for v in (P, q, A, b, G, h)))
        _check_problem(problem, eps_abs, eps_rel, max_iter)
        self.problem = problem
        # Plain host floats/ints: passed to jit as dynamic operands, so
        # changing a tolerance or the cap never triggers recompilation.
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.max_iter = int(max_iter)

    def solve(self):  # Run the compiled lax.while_loop.
        state = init_state(self.problem)
        state = _solve_loop_jit(self.problem, state, self.eps_abs,
                                self.eps_rel, self.max_iter)
        return _result(state)

    def trace(self):  # Record the same jitted step from a Python loop.
        state = init_state(self.problem)
        steps = []
        # bool() forces device-to-host sync: loop control needs a concrete value.
        while not bool(state.converged) and int(state.iteration) < self.max_iter:
            state, step_trace = _step(self.problem, state, self.eps_abs,
                                      self.eps_rel)
            steps.append(step_trace)
        # Stacking each StepTrace leaf over iterations turns the list of
        # per-step pytrees into one Trace of (N, ...) arrays.
        trace = Trace(*jax.tree.map(lambda *leaves: jnp.stack(leaves), *steps))
        return _result(state), trace
