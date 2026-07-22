"""barrierQP: one KKT factorization, two Newton questions.

Infeasible-start Mehrotra predictor-corrector for the dense strictly convex QP

    minimize    1/2 x.T P x + q.T x
    subject to  A x = b,  G x <= h        (slack s = h - G x > 0, dual z > 0)

Each iteration eliminates (ds, dz) from the four-block Newton system, factors
the reduced KKT matrix once with LU, and asks that single factorization two
questions: the affine predictor (where would pure Newton go?) and the centered
corrector (where should we actually go, given how much complementarity the
predictor would keep?). reference.py answers both questions with fresh full
four-block solves; this file must match it while doing one factor per iteration.
"""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # Must precede any array creation: f32 tolerances would mask real errors.

import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

TAU = 0.99  # fixed fraction-to-boundary damping for accepted steps


class Problem(NamedTuple):
    """Fixed QP data. A NamedTuple of arrays is a 'pytree': JAX transforms
    see through it, so a whole problem can cross the jit boundary as one value."""
    P: jax.Array  # (n, n) symmetric positive definite
    q: jax.Array  # (n,)
    A: jax.Array  # (m_eq, n), m_eq may be 0
    b: jax.Array  # (m_eq,)
    G: jax.Array  # (m_in, n), m_in >= 1
    h: jax.Array  # (m_in,)


class State(NamedTuple):
    """One iterate. JAX arrays are immutable, so a step returns a new State
    instead of mutating; every leaf keeps one fixed shape/dtype for the loop."""
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    converged: jax.Array  # bool scalar, judged only after a step
    iteration: jax.Array  # int scalar; also the factorization count


class Trace(NamedTuple):
    """Everything one iteration computed, in causal order, for tests and plots."""
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


class Result(NamedTuple):
    x: jax.Array
    y: jax.Array
    z: jax.Array
    s: jax.Array
    status: str  # "solved" or "max_iter", nothing else
    iterations: int
    factorizations: int  # = iterations: the step factors exactly once
    newton_solves: int   # = 2 * iterations: affine + corrector per factor


def inf_norm(v):
    return jnp.max(jnp.abs(v), initial=0.0)  # initial=0 keeps empty r_eq legal when m_eq == 0


def init_state(problem):
    """Deterministic interior start: x=0, y=0, unit positive s and z."""
    n, m_eq, m_in = problem.q.shape[0], problem.b.shape[0], problem.h.shape[0]
    return State(x=jnp.zeros(n), y=jnp.zeros(m_eq),
                 z=jnp.ones(m_in), s=jnp.ones(m_in),
                 converged=jnp.asarray(False), iteration=jnp.asarray(0))


def residuals(problem, x, y, z, s):
    P, q, A, b, G, h = problem
    r_dual = P @ x + q + A.T @ y + G.T @ z
    r_eq = A @ x - b
    r_ineq = G @ x + s - h
    mu = s @ z / s.shape[0]
    return r_dual, r_eq, r_ineq, mu


def tolerances(problem, x, y, z, s, eps_abs, eps_rel):
    """Absolute-plus-relative scales built from the terms forming each residual."""
    P, q, A, b, G, h = problem
    eps_p = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(A @ x), inf_norm(b), inf_norm(G @ x), inf_norm(s), inf_norm(h)]))
    eps_d = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(P @ x), inf_norm(q), inf_norm(A.T @ y), inf_norm(G.T @ z)]))
    eps_g = eps_abs + eps_rel * jnp.abs(0.5 * x @ P @ x + q @ x)
    return eps_p, eps_d, eps_g


def stop_test(problem, x, y, z, s, eps_abs, eps_rel):
    r_dual, r_eq, r_ineq, _ = residuals(problem, x, y, z, s)
    eps_p, eps_d, eps_g = tolerances(problem, x, y, z, s, eps_abs, eps_rel)
    pres = jnp.maximum(inf_norm(r_eq), inf_norm(r_ineq))
    return (pres <= eps_p) & (inf_norm(r_dual) <= eps_d) & (s @ z <= eps_g)


def form_kkt(problem, D):
    """Reduced KKT matrix after eliminating (ds, dz) from the four-block system."""
    P, _, A, _, G, _ = problem
    m_eq = A.shape[0]
    top = jnp.concatenate([P + G.T @ (D[:, None] * G), A.T], axis=1)
    bottom = jnp.concatenate([A, jnp.zeros((m_eq, m_eq))], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def solve_direction(K, lu, G, z, s, D, r_dual, r_eq, r_ineq, c):
    """Recover a full Newton direction from one reduced solve.

    c = -s*z + target - correction is the complementarity row's RHS. The
    elimination is elementwise division by s and z (diagonal matrices), never
    a formed matrix inverse; the only linear solve reuses the LU factor.
    """
    n = r_dual.shape[0]
    w = (c + z * r_ineq) / s
    rhs = jnp.concatenate([-r_dual - G.T @ w, -r_eq])
    sol = lu_solve(lu, rhs)
    dx, dy = sol[:n], sol[n:]
    dz = D * (G @ dx) + w
    ds = -r_ineq - G @ dx
    lin_res = inf_norm(K @ sol - rhs)  # true residual of the trusted LU solve
    return dx, dy, dz, ds, lin_res


def fraction_to_boundary(v, dv, tau):
    """Largest alpha in [0,1] with v + alpha*dv staying strictly positive
    (up to fraction tau); this is what preserves s > 0 and z > 0 forever."""
    ratios = jnp.where(dv < 0, -v / dv, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def step(problem, state, eps_abs, eps_rel):
    """One chronological Mehrotra iteration; pure, so jit can stage it."""
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
    trace = Trace(
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


def _solve_loop(problem, state, eps_abs, eps_rel, max_iter):
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


def _check_problem(problem, eps_abs, eps_rel, max_iter):
    P, q, A, b, G, h = problem
    n, m_eq, m_in = q.shape[0], b.shape[0], h.shape[0]
    assert P.shape == (n, n) and A.shape == (m_eq, n) and G.shape == (m_in, n)
    assert m_in >= 1, "barrierQP needs at least one inequality"
    assert bool(jnp.allclose(P, P.T)), "P must be symmetric"
    assert eps_abs > 0 and eps_rel >= 0 and max_iter >= 1


def _result(problem, state):
    iterations = int(state.iteration)  # host conversion; also synchronizes the device
    return Result(x=state.x, y=state.y, z=state.z, s=state.s,
                  status="solved" if bool(state.converged) else "max_iter",
                  iterations=iterations, factorizations=iterations,
                  newton_solves=2 * iterations)


def solve(P, q, A, b, G, h, eps_abs=1e-8, eps_rel=1e-8, max_iter=50):
    """Compiled solve: the fixed-shape while_loop around the one pure step."""
    problem = Problem(*map(jnp.asarray, (P, q, A, b, G, h)))
    _check_problem(problem, eps_abs, eps_rel, max_iter)
    state = init_state(problem)
    assert bool(jnp.all(state.s > 0) and jnp.all(state.z > 0))
    state = _solve_loop_jit(problem, state, eps_abs, eps_rel, max_iter)
    return _result(problem, state)


def solve_trace(P, q, A, b, G, h, eps_abs=1e-8, eps_rel=1e-8, max_iter=50):
    """Explanatory solve: the same jitted step driven from a Python loop so each
    iteration's Trace can be kept. Snapshots are appended only after the
    transformed call returns concrete arrays."""
    problem = Problem(*map(jnp.asarray, (P, q, A, b, G, h)))
    _check_problem(problem, eps_abs, eps_rel, max_iter)
    state = init_state(problem)
    traces = []
    # bool() forces device-to-host sync: loop control needs a concrete value.
    while not bool(state.converged) and int(state.iteration) < max_iter:
        state, trace = _step(problem, state, eps_abs, eps_rel)
        traces.append(trace)
    return _result(problem, state), traces
