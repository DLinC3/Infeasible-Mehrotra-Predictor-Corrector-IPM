"""The barrierQP correctness ladder, in one linear pass:

1. an exact-rational audit of the first predictor/centering/corrector iteration;
2. full four-block reference directions vs the reduced one-factor/two-RHS path,
   for every iteration of every fixture (this is the project's strongest claim:
   agreeing final answers could hide compensating errors, agreeing directions
   cannot);
3. complete trajectory, eager-vs-jit, and compiled-loop-vs-trace parity;
4. OSQP as the external final-point oracle, plus the N factor / 2N solve counts.

Run it twice: `JAX_PLATFORMS=cpu uv run python test.py` and `uv run python test.py`.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
import osqp

import barrierqp
import reference
import jax
import jax.numpy as jnp

assert jnp.ones(1).dtype == jnp.float64, "x64 must be active before any array math"

# ------------------------------------------------------------------ fixtures
# tiny: hand-derived optimum x=(0,1), y=-2, z=(0,1), s=(2,0); the second
# inequality is active there, the first is slack.
tiny = (jnp.array([[4.0, 1.0], [1.0, 3.0]]), jnp.array([1.0, -2.0]),
        jnp.array([[1.0, 1.0]]), jnp.array([1.0]),
        jnp.array([[1.0, 0.0], [0.0, 1.0]]), jnp.array([2.0, 1.0]))

# big: seeded strictly convex QP built around an attainable interior point, so
# the equalities are consistent and Slater holds by construction.
rng = np.random.default_rng(0)
n, m_eq, m_in = 6, 2, 8
M = rng.normal(size=(n, n))
A_big = rng.normal(size=(m_eq, n))
x_feas = rng.normal(size=n)
G_big = rng.normal(size=(m_in, n))
big = tuple(jnp.asarray(v) for v in (
    M.T @ M + np.eye(n), rng.normal(size=n),
    A_big, A_big @ x_feas,
    G_big, G_big @ x_feas + rng.uniform(0.1, 1.0, size=m_in)))

# noeq: no equality rows at all (m_eq = 0), the demo's problem shape.
noeq = (jnp.array([[2.0, 0.5], [0.5, 1.0]]), jnp.array([-2.0, -2.0]),
        jnp.zeros((0, 2)), jnp.zeros((0,)),
        jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), jnp.array([1.0, 1.0, 1.0]))

fixtures = {"tiny": tiny, "big": big, "noeq": noeq}

# ------------------------------------------- rung 1: exact first iteration
# Every quantity in the first iteration on `tiny` is rational, so it was
# computed independently with fractions.Fraction and is hard-coded here.
_, audit = reference.step(*tiny, *reference.init_state(tiny[1], tiny[3], tiny[5]),
                          1e-8, 1e-8, 0)
exact = {
    "dx_aff": [1 / 7, 6 / 7], "dy_aff": [-11 / 7],
    "dz_aff": [-13 / 7, -1 / 7], "ds_aff": [6 / 7, -6 / 7],
    "alpha_aff_primal": 1.0, "alpha_aff_dual": 7 / 13,
    "mu_aff": 6 / 91, "sigma": 216 / 753571,
    "dx": [-5 / 49, 54 / 49], "dy": [-1645769 / 753571],
    "dz": [-384259 / 753571, -15163 / 753571], "ds": [54 / 49, -54 / 49],
    "alpha_primal": 539 / 600, "alpha_dual": 1.0,
}
for key, value in exact.items():
    assert_allclose(np.asarray(audit[key]), value, rtol=1e-13, atol=1e-15,
                    err_msg=key)
print("audit okay (first iteration matches the exact fraction arithmetic)")

# ------------------- rungs 2+3: directions, trajectories, jit, loop, counts
DIRECTION_FIELDS = ["dx_aff", "dy_aff", "dz_aff", "ds_aff", "dx", "dy", "dz", "ds"]
SCALAR_FIELDS = ["alpha_aff_primal", "alpha_aff_dual", "mu_aff", "sigma",
                 "alpha_primal", "alpha_dual", "mu"]
STATE_FIELDS = ["x", "y", "z", "s"]

worst_direction = 0.0
iteration_counts = {}
for name, prob in fixtures.items():
    ref_final, ref_status, ref_snaps = reference.solve(*prob)
    result, traces = barrierqp.solve_trace(*prob)
    assert result.status == ref_status == "solved"
    assert result.iterations == len(ref_snaps) == len(traces)
    iteration_counts[name] = result.iterations

    for snap, tr in zip(ref_snaps, traces):
        for field in DIRECTION_FIELDS + SCALAR_FIELDS + STATE_FIELDS:
            ref_v, red_v = np.asarray(snap[field]), np.asarray(getattr(tr, field))
            if ref_v.size:
                assert_allclose(red_v, ref_v, rtol=1e-9, atol=1e-11, err_msg=field)
                if field in DIRECTION_FIELDS:
                    worst_direction = max(worst_direction,
                                          float(np.max(np.abs(red_v - ref_v))))
        # strict interiority of every recorded iterate
        assert np.all(np.asarray(tr.s) > 0) and np.all(np.asarray(tr.z) > 0)
        assert float(tr.linear_residual) < 1e-10

    # The compiled while_loop must land where the traced loop landed. The two
    # are differently compiled XLA programs, so allow device-level rounding
    # noise but nothing an extra or missing iteration could survive.
    compiled = barrierqp.solve(*prob)
    assert compiled.status == result.status
    assert compiled.iterations == result.iterations
    for field in STATE_FIELDS:
        assert_allclose(np.asarray(getattr(compiled, field)),
                        np.asarray(getattr(result, field)), rtol=1e-9, atol=1e-11)

    # the jitted step is the eager step (same function, staged once)
    problem = barrierqp.Problem(*prob)
    st0 = barrierqp.init_state(problem)
    st_eager, tr_eager = barrierqp.step(problem, st0, 1e-8, 1e-8)
    st_jit, tr_jit = barrierqp._step(problem, st0, 1e-8, 1e-8)
    for field in DIRECTION_FIELDS + SCALAR_FIELDS:
        eager_v = np.asarray(getattr(tr_eager, field))
        if eager_v.size:
            assert_allclose(np.asarray(getattr(tr_jit, field)), eager_v,
                            rtol=1e-12, atol=1e-14, err_msg=field)
    # the loop carry keeps fixed shapes and dtypes, iteration after iteration
    assert jax.tree.map(lambda a: (a.shape, a.dtype), st0) == \
           jax.tree.map(lambda a: (a.shape, a.dtype), st_jit)

    # count bookkeeping: one factorization and two Newton solves per iteration
    assert result.factorizations == result.iterations
    assert result.newton_solves == 2 * result.iterations

print(f"directions okay (max full/reduced difference {worst_direction:.2e})")
print(f"trajectory okay (iterations {iteration_counts}, jit/loop/counts agree)")

# ---------------------------------------------------------- rung 4: OSQP
# OSQP sees the same QP as  l <= [A; G] x <= u  with l = [b, -inf], u = [b, h].
for name, prob in fixtures.items():
    P, q, A, b, G, h = (np.asarray(v) for v in prob)
    stacked = np.vstack([A, G])
    lower = np.hstack([b, np.full(h.shape, -np.inf)])
    upper = np.hstack([b, h])
    oracle = osqp.OSQP()
    oracle.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(stacked), lower,
                 upper, eps_abs=1e-9, eps_rel=1e-9, verbose=False,
                 max_iter=200000)
    osqp_res = oracle.solve()
    assert osqp_res.info.status == "solved"

    ours = barrierqp.solve(*prob)
    x = np.asarray(ours.x)
    objective = 0.5 * x @ P @ x + q @ x
    assert_allclose(objective, osqp_res.info.obj_val, rtol=1e-7, atol=1e-7)
    assert_allclose(x, osqp_res.x, rtol=1e-5, atol=1e-5)

    # our final point satisfies the original KKT system on its own terms
    r_dual, r_eq, r_ineq, mu = (np.asarray(v) for v in barrierqp.residuals(
        barrierqp.Problem(*prob), ours.x, ours.y, ours.z, ours.s))
    kkt = max(np.max(np.abs(r_dual)), np.max(np.abs(r_ineq)),
              np.max(np.abs(r_eq), initial=0.0), float(ours.s @ ours.z))
    assert kkt < 1e-7, kkt
print("OSQP okay (objectives, solutions, and KKT residuals agree)")

print("all okay")
