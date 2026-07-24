"""barrierQP local timing experiment (optional, prints only).

    uv run --group bench python bench.py          # n = 32, 64, 128
    uv run --group bench python bench.py --full    # n = 16, 32, 64, 128, 256

Compares barrierQP with PIQP, Clarabel, and OSQP through their native public APIs
on well-scaled dense strictly-convex QPs. It writes nothing and downloads nothing.
This is a local experiment, not a general solver ranking: barrierQP is a JAX
educational implementation, so its first (compiling) call is shown separately and
the steady-state numbers are fresh setup + solve. A missing comparator prints
``not installed``.
"""

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import barrierqp

TOL = 1e-6        # requested abs/rel tolerance for every solver
ACCEPT = 1e-5     # independent acceptance: primal and stationarity residual


# --------------------------------------------------------------------------- #
# Deterministic fixtures + independent metrics                                #
# --------------------------------------------------------------------------- #
def _spd(rng, n, cond=1e2):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = np.exp(np.linspace(0.0, np.log(cond), n))
    return (Q * d) @ Q.T


def fixture(n, seed=0):
    """A well-scaled dense strictly-convex QP with a known feasible point."""
    rng = np.random.default_rng(1000 * seed + n)
    P = 0.5 * (lambda M: M + M.T)(_spd(rng, n))
    A = rng.normal(size=(n // 4, n))
    G = rng.normal(size=(n, n))
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    x = rng.normal(size=n)
    return dict(n=n, P=P, q=rng.normal(size=n), A=A, b=A @ x,
                G=G, h=G @ x + rng.uniform(0.5, 1.5, size=n))


def metrics(fix, x, dual):
    """Objective, primal residual, and stationarity in original coordinates."""
    P, q, A, b, G, h = (fix[k] for k in ("P", "q", "A", "b", "G", "h"))
    x = np.asarray(x, float)
    obj = 0.5 * x @ P @ x + q @ x
    primal = max(np.max(np.abs(A @ x - b), initial=0.0),
                 np.max(np.maximum(G @ x - h, 0.0), initial=0.0))
    if dual is None:
        stat = np.inf
    else:
        stat = np.max(np.abs(P @ x + q + np.vstack([A, G]).T @ np.asarray(dual)), initial=0.0)
    return float(obj), float(primal), float(stat)


def accepted(fix, x, dual):
    if x is None or not np.all(np.isfinite(np.asarray(x))):
        return False
    obj, primal, stat = metrics(fix, x, dual)
    return np.isfinite(obj) and primal < ACCEPT and stat < ACCEPT


def _median_ms(times_ns):
    return float(np.median(times_ns)) / 1e6


# --------------------------------------------------------------------------- #
# Native solver runners: (times_ns, x, dual_stack, solved_flag) or None       #
# --------------------------------------------------------------------------- #
def run_barrierqp(fix, reps):
    P, q, A, b, G, h = (jax.numpy.asarray(np.asarray(fix[k], float))
                        for k in ("P", "q", "A", "b", "G", "h"))

    def once():
        r = barrierqp.Solver(P, q, A, b, G, h, eps_abs=TOL, eps_rel=TOL, max_iter=500).solve()
        jax.block_until_ready([r.x, r.y, r.z, r.s])
        return r

    r = once()  # warm-up (shape already compiled by the first-call measurement)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        r = once()
        times.append(time.perf_counter_ns() - t0)
    dual = np.concatenate([np.asarray(r.y), np.asarray(r.z)])
    return times, np.asarray(r.x), dual, r.status == "solved"


def barrierqp_first_call_ms(fix):
    P, q, A, b, G, h = (jax.numpy.asarray(np.asarray(fix[k], float))
                        for k in ("P", "q", "A", "b", "G", "h"))
    t0 = time.perf_counter_ns()
    r = barrierqp.Solver(P, q, A, b, G, h, eps_abs=TOL, eps_rel=TOL, max_iter=500).solve()
    jax.block_until_ready([r.x])
    return (time.perf_counter_ns() - t0) / 1e6


def run_piqp(fix, reps):
    try:
        import piqp
    except ImportError:
        return None
    P, q, A, b, G, h = (np.asarray(fix[k], float) for k in ("P", "q", "A", "b", "G", "h"))
    Pf, Af, Gf = np.asfortranarray(P), np.asfortranarray(A), np.asfortranarray(G)
    h_l = np.full(G.shape[0], -np.inf)

    def once():
        s = piqp.DenseSolver()
        s.settings.eps_abs = TOL; s.settings.eps_rel = TOL; s.settings.verbose = False
        s.setup(Pf, q, Af, b, Gf, h_l, h, None, None)
        s.solve()
        return s.result

    r = once()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); r = once(); times.append(time.perf_counter_ns() - t0)
    dual = np.concatenate([np.asarray(r.y), np.asarray(r.z_u) - np.asarray(r.z_l)])
    return times, np.asarray(r.x), dual, str(r.info.status) == str(piqp.PIQP_SOLVED)


def run_clarabel(fix, reps):
    try:
        import clarabel
        import scipy.sparse as sp
    except ImportError:
        return None
    P, q, A, b, G, h = (np.asarray(fix[k], float) for k in ("P", "q", "A", "b", "G", "h"))
    Pc, Ac = sp.csc_matrix(P), sp.csc_matrix(np.vstack([A, G]))
    bc = np.concatenate([b, h])
    cones = [clarabel.ZeroConeT(A.shape[0]), clarabel.NonnegativeConeT(G.shape[0])]
    st = clarabel.DefaultSettings()
    st.verbose = False; st.tol_gap_abs = TOL; st.tol_gap_rel = TOL; st.tol_feas = TOL

    def once():
        return clarabel.DefaultSolver(Pc, q, Ac, bc, cones, st).solve()

    sol = once()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); sol = once(); times.append(time.perf_counter_ns() - t0)
    return times, np.asarray(sol.x), np.asarray(sol.z), str(sol.status) == "Solved"


def run_osqp(fix, reps):
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError:
        return None
    P, q, A, b, G, h = (np.asarray(fix[k], float) for k in ("P", "q", "A", "b", "G", "h"))
    Po, Ao = sp.csc_matrix(P), sp.csc_matrix(np.vstack([A, G]))
    lo = np.concatenate([b, np.full(G.shape[0], -np.inf)])
    uo = np.concatenate([b, h])

    def once():
        o = osqp.OSQP()
        o.setup(Po, q, Ao, lo, uo, eps_abs=TOL, eps_rel=TOL, verbose=False, max_iter=20000)
        return o.solve()

    r = once()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); r = once(); times.append(time.perf_counter_ns() - t0)
    return times, np.asarray(r.x), np.asarray(r.y), r.info.status == "solved"


COMPARATORS = [("piqp", run_piqp), ("clarabel", run_clarabel), ("osqp", run_osqp)]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="barrierQP local timing experiment")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    sizes = [16, 32, 64, 128, 256] if args.full else [32, 64, 128]
    reps = 5

    print(f"barrierQP local timing  |  setup + solve (median ms)  |  CPU float64  |  "
          f"tol {TOL}, accept < {ACCEPT}")
    print(f"{'n':>4} {'m_eq':>5} {'m_in':>5} | {'barrierqp':>10} {'first-call':>10} "
          f"| {'piqp':>9} {'clarabel':>9} {'osqp':>9}")

    for n in sizes:
        fix = fixture(n)
        first = barrierqp_first_call_ms(fix)         # compiling call (fresh shape)
        times, x, dual, native_ok = run_barrierqp(fix, reps)
        # private sanity check: barrierQP must return an accurate result
        if not (native_ok and accepted(fix, x, dual)):
            raise SystemExit(f"barrierQP returned an inaccurate result at n={n}: "
                             f"{metrics(fix, x, dual)}")
        cells = []
        for _, runner in COMPARATORS:
            res = runner(fix, reps)
            if res is None:
                cells.append("not inst.")
            else:
                rts, rx, rdual, rok = res
                cells.append(f"{_median_ms(rts):.3f}" if (rok and accepted(fix, rx, rdual)) else "FAIL")
        print(f"{n:>4} {n // 4:>5} {n:>5} | {_median_ms(times):>10.3f} {first:>10.0f} "
              f"| {cells[0]:>9} {cells[1]:>9} {cells[2]:>9}")

    print("\nfirst-call includes one-time JAX compilation (excluded from the barrierqp "
          "column). Local experiment, not a general solver ranking; timings depend on "
          "hardware, versions, tolerance, and workload.")


if __name__ == "__main__":
    main()
