import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp

import Constants as c


class CVXLeaderModel(nn.Module):
    """
    CVXPY traditional leader optimizer.

    Works like a neural network model:

        model = CVXPYLeaderModel(K)
        b, f = model(X)

    Input X shape:
        (N, 3K)

    X format:
        X[:, 0:K]      = D
        X[:, K:2K]     = H_mag
        X[:, 2K:3K]    = beta

    Output:
        b shape: (N, K)
        f shape: (N, K)
    """

    def __init__(
        self,
        K,
        B=None,
        C_dt=None,
        min_b=1e-9,
        min_f=1e-9,
        solver="CLARABEL",
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()

        self.K = K
        self.B = c.total_bandwidth if B is None else B
        self.C_dt = c.total_compute_speed if C_dt is None else C_dt

        self.min_b = min_b
        self.min_f = min_f
        self.solver = solver
        self.dtype = dtype
        self.device = device

        # Dummy parameter so next(model.parameters()).device works
        self.dummy = nn.Parameter(torch.empty(0, device=device, dtype=dtype))

    def solve_one(self, x_np):
        K = self.K

        D = x_np[:K]
        H = x_np[K : 2 * K]
        beta = x_np[2 * K : 3 * K]

        epsilon = c.compression_constant
        p = c.transmit_power
        N0 = c.N0
        c_k = c.dt_compute_complexity
        f_s = c.sensor_compression_speed

        b = cp.Variable(K, pos=True)
        f = cp.Variable(K, pos=True)
        T = cp.Variable()

        # Constant compression time
        eta = np.exp(beta * epsilon) - np.exp(epsilon)
        T_comp = (D * eta) / f_s

        constraints = [
            cp.sum(b) <= self.B,
            cp.sum(f) <= self.C_dt,
            b >= self.min_b,
            f >= self.min_f,
        ]

        for k in range(K):
            a_k = (H[k] ** 2 * p) / N0

            # rate = b_k * log2(1 + a_k / b_k)
            # CVXPY-safe form:
            # b * log(1 + a / b) = -rel_entr(b, b + a)
            rate_k = -cp.rel_entr(b[k], b[k] + a_k) / np.log(2)

            T_tr_k = (D[k] / beta[k]) * cp.inv_pos(rate_k)
            T_dt_k = D[k] * c_k * cp.inv_pos(f[k])

            constraints.append(T_comp[k] + T_tr_k + T_dt_k <= T)

        problem = cp.Problem(cp.Minimize(T), constraints)

        try:
            problem.solve(solver=self.solver)
        except Exception:
            problem.solve(solver="SCS")

        if b.value is None or f.value is None:
            raise RuntimeError("CVXPY failed to solve the leader optimization problem.")

        return b.value, f.value

    def forward(self, X):
        X_np = X.detach().cpu().numpy()

        b_list = []
        f_list = []

        for n in range(X_np.shape[0]):
            b_np, f_np = self.solve_one(X_np[n])
            b_list.append(b_np)
            f_list.append(f_np)

        b = torch.tensor(
            np.asarray(b_list),
            dtype=self.dtype,
            device=self.device,
        )

        f = torch.tensor(
            np.asarray(f_list),
            dtype=self.dtype,
            device=self.device,
        )

        return b, f
