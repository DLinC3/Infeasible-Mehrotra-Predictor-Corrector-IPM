"""The canonical barrierQP experiment: watch Mehrotra walk the central path.

One fixed two-dimensional QP over a pentagon (five inequalities, no
equalities, boundary optimum). Because every h_i = 1, the solver's default
start x=0, s=1 is exactly feasible, so the drawn path honestly lives inside
the polygon. The demo re-derives every printed claim from the solver trace,
checks the reduced directions against the full four-block reference, and only
then writes central_path.png.
"""

import matplotlib

matplotlib.use("Agg")  # never open a window; this script only writes one PNG
import matplotlib.pyplot as plt
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
ref_final, ref_status, ref_snaps = reference.solve(P, q, A, b, G, h)

# ----------------------------------------------- claims, before any plotting
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

# Per-iteration ledger, including the accepted final iterate as row N.
pres = np.append(np.max(np.abs(T["r_ineq"]), axis=1), np.max(np.abs(r_ineq_f)))
dres = np.append(np.max(np.abs(T["r_dual"]), axis=1), np.max(np.abs(r_dual_f)))
gap = np.append(np.sum(T["s"] * T["z"], axis=1), float(result.s @ result.z))
eps_p_f, eps_d_f, eps_g_f = (float(v) for v in barrierqp.tolerances(
    problem, result.x, result.y, result.z, result.s, 1e-8, 1e-8))
eps_p = np.append(T["eps_primal"], eps_p_f)
eps_d = np.append(T["eps_dual"], eps_d_f)
eps_g = np.append(T["eps_gap"], eps_g_f)

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

# ------------------------------------------------------------- the picture
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("barrierQP: one KKT factorization, two Newton questions", y=0.98)

# A. primal geometry: polygon, contours, iterates, and the two arrows of one
# iteration (both directions came from the same reduced KKT factorization).
ax = axes[0, 0]
pts = []
for i in range(len(h)):
    for j in range(i + 1, len(h)):
        Mij = G[[i, j]]
        if abs(np.linalg.det(Mij)) > 1e-12:
            v = np.linalg.solve(Mij, h[[i, j]])
            if np.all(G @ v <= h + 1e-9):
                pts.append(v)
pts = np.array(pts)
order = np.argsort(np.arctan2(pts[:, 1] - pts.mean(0)[1], pts[:, 0] - pts.mean(0)[0]))
polygon = pts[order]
ax.fill(polygon[:, 0], polygon[:, 1], color="0.92", zorder=0)
ax.plot(np.append(polygon[:, 0], polygon[0, 0]),
        np.append(polygon[:, 1], polygon[0, 1]), color="0.4", lw=1)
gx, gy = np.meshgrid(np.linspace(-1.4, 1.7, 200), np.linspace(-1.4, 1.7, 200))
f = 0.5 * (P[0, 0] * gx**2 + 2 * P[0, 1] * gx * gy + P[1, 1] * gy**2) \
    + q[0] * gx + q[1] * gy
ax.contour(gx, gy, f, levels=12, colors="tab:blue", linewidths=0.5, alpha=0.6)
ax.plot(x_path[:, 0], x_path[:, 1], "o-", color="tab:red", ms=4, lw=1.2,
        label="iterates")
ax.plot(*x_path[-1], "*", color="black", ms=13, label="optimum")
# Arrows at the most legible iteration: where the affine trial step and the
# accepted predictor-corrector step disagree the most.
trial = T["alpha_aff_primal"][:, None] * T["dx_aff"]
accepted = np.diff(x_path, axis=0)
k_arrow = int(np.argmax(np.linalg.norm(trial - accepted, axis=1)))
base = x_path[k_arrow]
ax.annotate("", base + accepted[k_arrow], base, arrowprops=dict(
    arrowstyle="->", color="tab:green", lw=1.8))
ax.annotate("", base + trial[k_arrow], base, arrowprops=dict(
    arrowstyle="->", linestyle="--", color="tab:purple", lw=1.6))
ax.plot([], [], "--", color="tab:purple", label=f"affine predictor (iter {k_arrow})")
ax.plot([], [], "-", color="tab:green", label="accepted corrector step")
ax.set_title("A. primal path: both arrows from one factorization")
ax.set_xlabel("x1"), ax.set_ylabel("x2")
ax.legend(fontsize=8, loc="lower left")
ax.set_aspect("equal")

# B. the KKT ledger against its moving tolerances.
ax = axes[0, 1]
its = np.arange(N + 1)
ax.semilogy(its, np.maximum(pres, 1e-17), "o-", label="primal ||r_ineq||", ms=3)
ax.semilogy(its, dres, "o-", label="dual ||r_dual||", ms=3)
ax.semilogy(its, gap, "o-", label="gap s'z", ms=3)
ax.semilogy(its, eps_p, ":", color="tab:blue", label="eps_primal")
ax.semilogy(its, eps_d, ":", color="tab:orange", label="eps_dual")
ax.semilogy(its, eps_g, ":", color="tab:green", label="eps_gap")
ax.axvline(N, color="0.5", lw=1, dashes=(4, 2))
ax.text(N, ax.get_ylim()[0], " stop", ha="left", va="bottom", fontsize=8)
ax.set_title("B. KKT ledger: residuals meet their tolerances")
ax.set_xlabel("iteration")
ax.legend(fontsize=8, loc="upper right", ncol=2)

# C. the controller: predicted complementarity decides centering and stepping.
ax = axes[1, 0]
its_k = np.arange(N)
ax.plot(its_k, T["mu_aff"] / T["mu"], "o-", label="mu_aff / mu", ms=3)
ax.plot(its_k, T["sigma"], "o-", label="sigma = (mu_aff/mu)^3", ms=3)
ax.plot(its_k, T["alpha_primal"], "o-", label="alpha_primal", ms=3)
ax.plot(its_k, T["alpha_dual"], "o-", label="alpha_dual", ms=3)
ax.set_ylim(-0.05, 1.05)
ax.set_title("C. controller: affine forecast -> centering -> safe step")
ax.set_xlabel("iteration")
ax.legend(fontsize=8, loc="center right")

# D. every slack-dual pair walks the s*z = mu hyperbolas toward its corner.
ax = axes[1, 1]
for i in range(len(h)):
    ax.loglog(s_path[:, i], z_path[:, i], "o-", ms=3, lw=1,
              label=f"inequality {i}")
s_line = np.logspace(-10, 1, 50)
for mu_level in [1e-2, 1e-5, 1e-8]:
    ax.loglog(s_line, mu_level / s_line, ":", color="0.6", lw=0.8)
    ax.annotate(f"s·z={mu_level:.0e}", (2.0, mu_level / 2.0), fontsize=7,
                color="0.45")
ax.set_xlim(1e-10, 1e2), ax.set_ylim(1e-11, 1e1)
ax.set_title("D. complementarity: active pairs go left, inactive go down")
ax.set_xlabel("slack s_i"), ax.set_ylabel("dual z_i")
ax.legend(fontsize=8, loc="lower left")

fig.tight_layout()
fig.savefig("central_path.png", dpi=130)

import os

assert os.path.getsize("central_path.png") > 0
print("wrote central_path.png")
