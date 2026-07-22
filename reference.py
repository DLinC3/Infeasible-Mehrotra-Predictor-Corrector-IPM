"""The slow transparent oracle: Mehrotra predictor-corrector where every Newton
direction comes from a fresh dense solve of the full four-block KKT system.

This file is eager JAX on purpose: every array is concrete, every equation is
written once in display order, and nothing is compiled or cached. barrierqp.py
must reproduce these directions and iterates exactly (to float64 accuracy)
while factoring a reduced system once and reusing it for both solves.
"""

import jax

jax.config.update("jax_enable_x64", True)  # x64 first: float32 would hide real disagreement.

import jax.numpy as jnp

TAU = 0.99  # fixed fraction-to-boundary damping for accepted steps


def inf_norm(v):
    return jnp.max(jnp.abs(v), initial=0.0)  # initial=0 keeps empty r_eq legal when there are no equalities


def init_state(q, b, h):
    """Deterministic interior start: x=0, y=0, and unit positive s, z."""
    n, m_eq, m_in = q.shape[0], b.shape[0], h.shape[0]
    return jnp.zeros(n), jnp.zeros(m_eq), jnp.ones(m_in), jnp.ones(m_in)


def residuals(P, q, A, b, G, h, x, y, z, s):
    r_dual = P @ x + q + A.T @ y + G.T @ z
    r_eq = A @ x - b
    r_ineq = G @ x + s - h
    mu = s @ z / s.shape[0]
    return r_dual, r_eq, r_ineq, mu


def ledger(P, q, A, b, G, h, x, y, z, s, eps_abs, eps_rel):
    """Residual norms and their absolute-plus-relative stopping tolerances."""
    r_dual, r_eq, r_ineq, _ = residuals(P, q, A, b, G, h, x, y, z, s)
    pres = jnp.maximum(inf_norm(r_eq), inf_norm(r_ineq))
    dres = inf_norm(r_dual)
    gap = s @ z
    eps_p = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(A @ x), inf_norm(b), inf_norm(G @ x), inf_norm(s), inf_norm(h)]))
    eps_d = eps_abs + eps_rel * jnp.max(jnp.array(
        [inf_norm(P @ x), inf_norm(q), inf_norm(A.T @ y), inf_norm(G.T @ z)]))
    eps_g = eps_abs + eps_rel * jnp.abs(0.5 * x @ P @ x + q @ x)
    return pres, dres, gap, eps_p, eps_d, eps_g


def newton_full(P, A, G, z, s, r_dual, r_eq, r_ineq, c):
    """Solve the displayed four-block delta system afresh for one direction.

    [P  A.T  G.T  0 ] [dx]   [-r_dual]
    [A   0    0   0 ] [dy] = [-r_eq  ]
    [G   0    0   I ] [dz]   [-r_ineq]
    [0   0    S    Z] [ds]   [   c   ]      c = -s*z + target - correction
    """
    n, m_eq, m_in = P.shape[0], A.shape[0], G.shape[0]
    O = jnp.zeros
    kkt = jnp.concatenate([
        jnp.concatenate([P, A.T, G.T, O((n, m_in))], axis=1),
        jnp.concatenate([A, O((m_eq, m_eq)), O((m_eq, m_in)), O((m_eq, m_in))], axis=1),
        jnp.concatenate([G, O((m_in, m_eq)), O((m_in, m_in)), jnp.eye(m_in)], axis=1),
        jnp.concatenate([O((m_in, n)), O((m_in, m_eq)), jnp.diag(s), jnp.diag(z)], axis=1),
    ], axis=0)
    rhs = jnp.concatenate([-r_dual, -r_eq, -r_ineq, c])
    d = jnp.linalg.solve(kkt, rhs)
    dx, dy = d[:n], d[n:n + m_eq]
    dz, ds = d[n + m_eq:n + m_eq + m_in], d[n + m_eq + m_in:]
    lin_res = inf_norm(kkt @ d - rhs)
    return dx, dy, dz, ds, lin_res


def fraction_to_boundary(v, dv, tau):
    """Largest step in [0,1] keeping v + alpha*dv >= (1-tau) of the boundary distance."""
    ratios = jnp.where(dv < 0, -v / dv, jnp.inf)
    return jnp.minimum(1.0, tau * jnp.min(ratios))


def step(P, q, A, b, G, h, x, y, z, s, eps_abs, eps_rel, iteration):
    """One Mehrotra iteration with two independent full-system solves."""
    m_in = h.shape[0]
    r_dual, r_eq, r_ineq, mu = residuals(P, q, A, b, G, h, x, y, z, s)
    pres, dres, gap, eps_p, eps_d, eps_g = ledger(
        P, q, A, b, G, h, x, y, z, s, eps_abs, eps_rel)

    # Affine predictor: pure Newton toward complementarity zero.
    c_aff = -(s * z)
    dx_aff, dy_aff, dz_aff, ds_aff, _ = newton_full(
        P, A, G, z, s, r_dual, r_eq, r_ineq, c_aff)
    alpha_aff_primal = fraction_to_boundary(s, ds_aff, 1.0)
    alpha_aff_dual = fraction_to_boundary(z, dz_aff, 1.0)

    # Mehrotra centering: how much complementarity the pure step would keep.
    mu_aff = (s + alpha_aff_primal * ds_aff) @ (z + alpha_aff_dual * dz_aff) / m_in
    sigma = jnp.clip((mu_aff / mu) ** 3, 0.0, 1.0)

    # Centered corrector: target sigma*mu minus the predictor's cross term.
    c_corr = -(s * z) + sigma * mu - ds_aff * dz_aff
    dx, dy, dz, ds, lin_res = newton_full(
        P, A, G, z, s, r_dual, r_eq, r_ineq, c_corr)
    alpha_primal = fraction_to_boundary(s, ds, TAU)
    alpha_dual = fraction_to_boundary(z, dz, TAU)

    x_new = x + alpha_primal * dx
    s_new = s + alpha_primal * ds
    y_new = y + alpha_dual * dy
    z_new = z + alpha_dual * dz

    snapshot = {
        "iteration": iteration, "x": x, "y": y, "z": z, "s": s,
        "r_eq": r_eq, "r_ineq": r_ineq, "r_dual": r_dual, "mu": mu,
        "eps_primal": eps_p, "eps_dual": eps_d, "eps_gap": eps_g,
        "dx_aff": dx_aff, "dy_aff": dy_aff, "dz_aff": dz_aff, "ds_aff": ds_aff,
        "alpha_aff_primal": alpha_aff_primal, "alpha_aff_dual": alpha_aff_dual,
        "mu_aff": mu_aff, "sigma": sigma,
        "dx": dx, "dy": dy, "dz": dz, "ds": ds,
        "alpha_primal": alpha_primal, "alpha_dual": alpha_dual,
        "linear_residual": lin_res,
    }
    return (x_new, y_new, z_new, s_new), snapshot


def solve(P, q, A, b, G, h, eps_abs=1e-8, eps_rel=1e-8, max_iter=50):
    """Eager reference loop; returns the final iterate, status, and all snapshots."""
    x, y, z, s = init_state(q, b, h)
    snapshots = []
    status = "max_iter"
    for iteration in range(max_iter):
        (x, y, z, s), snap = step(
            P, q, A, b, G, h, x, y, z, s, eps_abs, eps_rel, iteration)
        snapshots.append(snap)
        # Stopping is tested only after a step, on the newly accepted iterate.
        pres, dres, gap, eps_p, eps_d, eps_g = ledger(
            P, q, A, b, G, h, x, y, z, s, eps_abs, eps_rel)
        if bool((pres <= eps_p) & (dres <= eps_d) & (gap <= eps_g)):
            status = "solved"
            break
    return (x, y, z, s), status, snapshots
