"""The canonical barrierQP experiment: watch Mehrotra walk the central path.

One fixed two-dimensional QP over a pentagon (five inequalities, no
equalities, boundary optimum). Because every h_i = 1, the solver's default
start x=0, s=1 is exactly feasible. The demo re-derives every printed claim
from the solver trace and checks the reduced directions against the full
four-block reference.
"""

import numpy as np

import barrierqp
import reference

# ------------------------------------------------------------- the fixture
# Pentagon g_i . x <= 1: the square [-1,1]^2 cut by 0.625*(x1+x2) <= 1.
G = np.array([[1.0, 0.0], [0.0, 1.0], [0.625, 0.625], [-1.0, 0.0], [0.0, -1.0]])
h = np.ones(5)
P = np.array([[2.0, 0.5], [0.5, 1.5]])
# q is chosen so the KKT conditions hold exactly at x* = (0.85, 0.75) with
# z* = (0,0,1,0,0): only the diagonal cut is active, strictly (z*_2 = 1).
# Check: r_dual = P x* + q + G.T z* = [2.075,1.55] - [2.7,2.175] + [.625,.625] = 0.
q = np.array([-2.7, -2.175])
x_star = np.array([0.85, 0.75])
A = np.zeros((0, 2))
b = np.zeros(0)

result, traces = barrierqp.solve_trace(P, q, A, b, G, h)
_, ref_status, ref_snaps = reference.solve(P, q, A, b, G, h)

# --------------------------------------------------------- executable claims
assert result.status == ref_status == "solved"
N = result.iterations
assert result.factorizations == N and result.newton_solves == 2 * N
assert len(traces) == len(ref_snaps) == N

max_dir_diff = 0.0
for snap, tr in zip(ref_snaps, traces):
    for field in ["dx_aff", "dz_aff", "ds_aff", "dx", "dz", "ds"]:
        diff = np.max(np.abs(np.asarray(snap[field]) - np.asarray(getattr(tr, field))))
        max_dir_diff = max(max_dir_diff, float(diff))
assert max_dir_diff < 1e-8, max_dir_diff

# One host crossing: stack every per-iteration trace field into NumPy arrays.
T = {f: np.stack([np.asarray(getattr(tr, f)) for tr in traces])
     for f in barrierqp.Trace._fields}
x_path = np.vstack([T["x"], np.asarray(result.x)])
s_path = np.vstack([T["s"], np.asarray(result.s)])
z_path = np.vstack([T["z"], np.asarray(result.z)])
assert np.all(s_path > 0) and np.all(z_path > 0)

problem = barrierqp.Problem(*map(np.asarray, (P, q, A, b, G, h)))
r_dual_f, _, r_ineq_f, _ = (np.asarray(v) for v in barrierqp.residuals(
    problem, result.x, result.y, result.z, result.s))
final_kkt = max(np.max(np.abs(r_dual_f)), np.max(np.abs(r_ineq_f)),
                float(result.s @ result.z))
assert final_kkt < 1e-7, final_kkt
# The boundary optimum is known in closed form: the diagonal cut is active.
assert np.max(np.abs(x_path[-1] - x_star)) < 1e-6
assert np.min(s_path[-1]) < 1e-6

# Per-iteration residuals, including the accepted final iterate as row N.
pres = np.append(np.max(np.abs(T["r_ineq"]), axis=1), np.max(np.abs(r_ineq_f)))
dres = np.append(np.max(np.abs(T["r_dual"]), axis=1), np.max(np.abs(r_dual_f)))

print("iter   primal      dual        mu         mu_aff      sigma      "
      "alpha_pri  alpha_dual  linear_res")
for k in range(N):
    print(f"{k:4d}  {pres[k]:.3e}  {dres[k]:.3e}  {T['mu'][k]:.3e}  "
          f"{T['mu_aff'][k]:.3e}  {T['sigma'][k]:.3e}  "
          f"{T['alpha_primal'][k]:9.6f}  {T['alpha_dual'][k]:9.6f}  "
          f"{T['linear_residual'][k]:.3e}")

print(f"{N} iterations, {result.factorizations} factorizations, "
      f"{result.newton_solves} Newton solves")
print(f"max full/reduced direction difference: {max_dir_diff:.2e}")
print(f"final KKT residual: {final_kkt:.2e}")
