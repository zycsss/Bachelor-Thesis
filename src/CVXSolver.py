import numpy as np
import cvxpy as cp
import Constants as c


class CVXSolver:
    def __init__(self, K, B_total=None, C_DT_total=None, beta_max=None):
        self.K = K
        self.B_total = B_total if B_total is not None else c.total_bandwidth
        self.C_DT_total = (
            C_DT_total if C_DT_total is not None else c.total_compute_power
        )
        self.beta_max = beta_max if beta_max is not None else c.beta_max

    def solve(self, X, verbose=False):
        X = self._to_numpy_2d(X)

        N = X.shape[0]
        K = self.K

        D = X[:, :K]
        H = X[:, K : 2 * K]
        beta = X[:, 2 * K : 3 * K]
        comp_speed = X[:, 3 * K : 4 * K]
        tr_power = X[:, 4 * K : 5 * K]

        g = (H**2) * tr_power / c.N0

        eta = np.exp(beta * c.compression_constant) - np.exp(c.compression_constant)
        T_comp = D * eta / comp_speed

        T_DT = cp.Variable((N, K), nonneg=True)
        T_tr = cp.Variable((N, K), nonneg=True)
        T = cp.Variable(N, nonneg=True)

        b = cp.Variable((N, K), nonneg=True)
        f = cp.Variable((N, K), nonneg=True)

        rate = -cp.rel_entr(b, b + g) / np.log(2)

        constraints = [
            cp.sum(b, axis=1) <= self.B_total,
            cp.sum(f, axis=1) <= self.C_DT_total,
            f >= cp.multiply(c.dt_compute_complexity * D, cp.inv_pos(T_DT)),
            rate >= cp.multiply(D, cp.inv_pos(cp.multiply(beta, T_tr))),
            T[:, None] >= T_DT + T_tr + T_comp,
        ]

        problem = cp.Problem(cp.Minimize(cp.sum(T)), constraints)

        problem.solve(
            solver=cp.MOSEK,
            verbose=verbose,
        )

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"CVX leader optimization failed: {problem.status}")

        return np.asarray(b.value), np.asarray(f.value)

    def predict(self, X, verbose=False):
        return self.solve(X, verbose=verbose)

    def _to_numpy_2d(self, X):
        try:
            import torch

            if isinstance(X, torch.Tensor):
                X = X.detach().cpu().numpy()
        except ImportError:
            pass

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must have shape (5K,) or (N, 5K)")

        expected_cols = 5 * self.K
        if X.shape[1] != expected_cols:
            raise ValueError(f"X must have {expected_cols} columns, got {X.shape[1]}")

        return X
