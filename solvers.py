import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class PDIPSolver:
    """
    Mehrotra-style primal-dual interior-point method for convex QPs:

        min  0.5 x^T Q x + q^T x
        s.t. A x = b,  G x + s = h,  s >= 0,  z >= 0.

    Uses affine-scaling + centering/corrector with sigma from predicted gap,
    and positivity step limit alpha_max (no residual-norm backtracking here).
    """
    def __init__(self, max_qp_iter: int = 50, tol: float = 1e-15):
        """Set iteration cap and convergence tolerance."""
        self.max_qp_iter = max_qp_iter
        self.tol = tol

        self.nx = None
        self.n_eq = None
        self.n_ineq = None

        self.x0 = None
        self.s0 = None
        self.z0 = None
        self.y0 = None

        self._Q = None
        self._q = None
        self._A = None
        self._b = None
        self._G = None
        self._h = None

    def init_problem(self, Q: jax.Array, q: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array) -> None:
        """Store QP data (symmetrize Q) and validate shapes."""
        self._Q = 0.5 * (Q + Q.T)
        self._q = q
        self.nx = q.size
        assert q.size**2 == Q.size

        self._A = A
        self._b = b
        self.n_eq = b.size
        assert A.shape == (self.n_eq, self.nx)

        self._G = G
        self._h = h
        self.n_ineq = h.size
        assert G.shape == (self.n_ineq, self.nx)

    def init_soln(self):
        """
        CVXGEN-style strictly-interior start:
          1) solve [Q G^T A^T; G -I 0; A 0 0][x;*;y]=[-q;h;b]
          2) r:=Gx-h, set s=-r and z=r, then shift both to make s>0,z>0.
        """
        nx, ne, ni = self.nx, self.n_eq, self.n_ineq
        Q, A, G = self._Q, self._A, self._G
        q, b, h = self._q, self._b, self._h

        K = jnp.block([
            [Q,                       G.T,                      A.T],
            [G,             -jnp.eye(ni),           jnp.zeros((ni, ne))],
            [A,              jnp.zeros((ne, ni)),   jnp.zeros((ne, ne))]
        ])
        rhs = jnp.concatenate([-q, h, b])

        sol = jsp.linalg.solve(K, rhs, assume_a="gen")
        x = sol[:nx]
        y = sol[nx + ni :]

        r = G @ x - h

        alpha_p = jnp.max(r)
        s = jnp.where(alpha_p < 0.0, -r, -r + (1.0 + alpha_p) * jnp.ones_like(r))

        alpha_d = jnp.max(-r)
        z = jnp.where(alpha_d < 0.0, r, r + (1.0 + alpha_d) * jnp.ones_like(r))

        self.x0, self.s0, self.z0, self.y0 = x, s, z, y
        return x, s, z, y

    def compute_residuals(self, xbar: jax.Array, sbar: jax.Array, zbar: jax.Array, ybar: jax.Array):
        """
        Residual RHS (already negated) for Newton/KKT solve:
          r1=-(Qx+q+A^Ty+G^Tz), r2=-(S z), r3=-(Gx+s-h), r4=-(Ax-b).
        """
        r1 = -(self._Q @ xbar + self._q + self._A.T @ ybar + self._G.T @ zbar)
        r2 = -(jnp.diag(sbar) @ zbar)
        r3 = -(self._G @ xbar + sbar - self._h)
        r4 = -(self._A @ xbar - self._b)
        return r1, r2, r3, r4

    def compute_centering_plus_corrector(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array) -> None:
        """
        Mehrotra parameters from affine step:
          mu=(s^T z)/m, alpha_aff=alpha_max(s,z;ds,dz),
          sigma=(( (s+αds)^T(z+αdz) )/(s^T z))^3.
        """
        m = self.n_ineq
        mu = (s0.T @ z0) / m
        alpha_aff, _ = self.compute_line_search(s0, ds, z0, dz, tol=1e-16)
        eta_aff = (s0 + alpha_aff * ds).T @ (z0 + alpha_aff * dz)
        eta = s0.T @ z0
        sigma = (eta_aff / eta) ** 3
        return mu, sigma

    def compute_line_search(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array, n_steps: int = 500, tol: float = 1e-15):
        """
        Positivity step limit:
          alpha_max = min(1, min_i{-s_i/ds_i|ds_i<0}, min_i{-z_i/dz_i|dz_i<0}).
        """
        def max_step(v, dv):
            mask = dv < 0.0
            return jnp.min(-v[mask] / dv[mask]) if jnp.any(mask) else jnp.inf

        step_s = max_step(s0, ds)
        step_z = max_step(z0, dz)
        alpha_sup = jnp.minimum(1.0, jnp.minimum(step_s, step_z))
        ok = jnp.isfinite(alpha_sup)
        return float(alpha_sup), bool(ok)

    def has_converged(self) -> bool:
        """Converged if max residual norm and avg complementarity (s^T z)/m are <= tol."""
        r1, r2, r3, r4 = self.compute_residuals(self.x0, self.s0, self.z0, self.y0)
        n1 = jnp.linalg.norm(r1, 2)
        n2 = jnp.linalg.norm(r2, 2)
        n3 = jnp.linalg.norm(r3, 2)
        n4 = jnp.linalg.norm(r4, 2)
        res_ok = bool(jnp.max(jnp.array([n1, n2, n3, n4])) <= self.tol)

        m = self.n_ineq if self.n_ineq > 0 else 1
        gap_ok = bool((self.s0 @ self.z0) / m <= self.tol)
        return res_ok and gap_ok

    def solve_kkt_system(self, r1: jax.Array, r2: jax.Array, r3: jax.Array, r4: jax.Array, sbar: jax.Array, zbar: jax.Array):
        """
        Solve Newton system with current (sbar,zbar):
          [Q 0 G^T A^T; 0 Z S 0; G I 0 0; A 0 0 0] [dx ds dz dy] = [r1 r2 r3 r4].
        """
        nx, ni, ne = self.nx, self.n_ineq, self.n_eq
        Q, A, G = self._Q, self._A, self._G

        S = jnp.diag(sbar)
        Z = jnp.diag(zbar)

        K = jnp.block([
            [Q,                   jnp.zeros((nx, ni)),    G.T,                 A.T],
            [jnp.zeros((ni, nx)),          Z,             S,                   jnp.zeros((ni, ne))],
            [G,                   jnp.eye(ni),            jnp.zeros((ni, ni)), jnp.zeros((ni, ne))],
            [A,                   jnp.zeros((ne, ni)),    jnp.zeros((ne, ni)), jnp.zeros((ne, ne))]
        ])
        rhs = jnp.concatenate([r1, r2, r3, r4])

        sol = jsp.linalg.solve(K, rhs, assume_a="gen")
        dx = sol[:nx]
        ds = sol[nx : nx + ni]
        dz = sol[nx + ni : nx + 2 * ni]
        dy = sol[nx + 2 * ni :]
        return dx, ds, dz, dy

    def solve_qp(self, verbose: bool = False):
        """
        Mehrotra loop:
          init -> affine solve -> (mu,sigma) -> corrector solve -> combine -> alpha -> update.
        Returns objective values per iteration.
        """
        x, s, z, y = self.init_soln()
        costs = [float(0.5 * (x @ (self._Q @ x)) + self._q @ x)]

        for k in range(1, self.max_qp_iter + 1):
            self.x0, self.s0, self.z0, self.y0 = x, s, z, y
            if self.has_converged():
                if verbose:
                    print(f"Converged at iter {k-1}")
                break

            # affine
            r1, r2, r3, r4 = self.compute_residuals(x, s, z, y)
            dx_aff, ds_aff, dz_aff, dy_aff = self.solve_kkt_system(r1, r2, r3, r4, s, z)

            # sigma, mu
            mu, sigma = self.compute_centering_plus_corrector(s, ds_aff, z, dz_aff)

            # corrector
            u1_cc = jnp.zeros(self.nx)
            u3_cc = jnp.zeros(self.n_ineq)
            u4_cc = jnp.zeros(self.n_eq)
            u2_cc = sigma * mu * jnp.ones(self.n_ineq) - (jnp.diag(ds_aff) @ dz_aff)
            dx_cc, ds_cc, dz_cc, dy_cc = self.solve_kkt_system(u1_cc, u2_cc, u3_cc, u4_cc, s, z)

            # combine
            dx = dx_aff + dx_cc
            ds = ds_aff + ds_cc
            dz = dz_aff + dz_cc
            dy = dy_aff + dy_cc

            # step (0.99 shrink)
            alpha_sup, ok = self.compute_line_search(s, ds, z, dz, tol=0.0)
            alpha = float(min(1.0, 0.99 * alpha_sup)) if ok else 1e-3

            # update
            x = x + alpha * dx
            s = s + alpha * ds
            z = z + alpha * dz
            y = y + alpha * dy

            cost = float(0.5 * (x @ (self._Q @ x)) + self._q @ x)
            costs.append(cost)

            if verbose:
                r1n, r2n, r3n, r4n = self.compute_residuals(x, s, z, y)
                rdual = float(jnp.linalg.norm(r1n, 2))
                rpri  = float(jnp.linalg.norm(jnp.concatenate([r3n, r4n]), 2))
                gap   = float(s @ z)
                mu_now = float(gap / max(self.n_ineq, 1))
                print(f"it={k:02d} alpha={alpha:.3e} cost={cost:.3e} mu={mu_now:.3e} r_feas={(rdual**2+rpri**2)**0.5:.3e} gap={gap:.3e}")

            self.x0, self.s0, self.z0, self.y0 = x, s, z, y

        return costs
