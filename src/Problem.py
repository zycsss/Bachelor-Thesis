import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pathlib import Path
import ProcessTime
import AnalyzeUtil
import cvxpy as cp
import numpy as np
import Constants as c
from DataUtil import generate_data_loader
from TrainingUtil import unsupervised_loss, train, set_seed
from torch.optim import Adam, lr_scheduler
from EarlyStopping import EarlyStopping
from NNSolver import Net
import time


class Problem:
    def __init__(
        self,
        D: NDArray,
        H_mag: NDArray,
        comp_speed=None,
        tr_power=None,
        beta_max=c.beta_max,
        total_bandwidth=c.total_bandwidth,
    ):
        self.D = self._as_2d(D)
        self.H_mag = self._as_2d(H_mag)

        self.N, self.K = self.D.shape

        if self.H_mag.shape != (self.N, self.K):
            raise ValueError("H_mag must have the same shape as D")

        self.tr_power = self._init_param(
            tr_power,
            default_value=c.transmit_power,
            name="tr_power",
        )

        self.comp_speed = self._init_param(
            comp_speed,
            default_value=c.sensor_compression_speed,
            name="comp_speed",
        )

        self.beta_max = beta_max
        self.total_bandwidth = total_bandwidth

        # Decision variables, shape: (N, K)
        self.b = np.ones((self.N, self.K), dtype=float)
        self.f = np.ones((self.N, self.K), dtype=float)

        # Dual variables, shape: (N, K, 2)
        # lam[:, :, 0] = lambda_1
        # lam[:, :, 1] = lambda_2
        self.lam = np.full((self.N, self.K, 2), 0.1, dtype=float)

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # print(f"using: {self.device}")

        self.total_time_used = 0
        self.leader_time_used = []

    def _as_2d(self, x):
        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("Input must be shape (K,) or (N, K)")

        return x

    def _init_param(self, value, default_value, name):
        if value is None:
            return np.full((self.N, self.K), default_value, dtype=float)

        value = self._as_2d(value)

        if value.shape == (1, self.K) and self.N > 1:
            value = np.repeat(value, self.N, axis=0)

        if value.shape != (self.N, self.K):
            raise ValueError(f"{name} must have shape (K,) or (N, K)")

        return value

    def train_new_model(
        self,
        checkpoint_path="model/checkpoint.pth",
        print_losses=True,
        n_epoch=100,
        train_length=int(1e4),
    ):

        set_seed()

        model = Net(
            self.K, self.total_bandwidth, c.total_compute_power, self.beta_max
        ).to(self.device)

        checkpoint_path = str(checkpoint_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        train_loader, test_loader = generate_data_loader(
            train_length,
            self.K,
            self.K * 4e3,
            comp_speed_range=[
                c.sensor_compression_speed / 2,
                c.sensor_compression_speed * 2,
            ],
            tr_power_range=[self.tr_power.mean() / 2, self.tr_power.mean() * 2],
            beta_max=self.beta_max,
        )

        optimizer = Adam(model.parameters(), lr=1e-3)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        early_stopper = EarlyStopping(patience=4, min_delta=0.1, path=checkpoint_path)

        loss_fn = unsupervised_loss

        train(
            model,
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            early_stopper,
            loss_fn,
            self.device,
            n_epoch=n_epoch,
            path_to_weight=checkpoint_path,
            print_losses=print_losses,
        )

        return model

    def to_X(self):
        """
        Returns X with shape (N, 3K):
        [D_1...D_K, H_1...H_K, beta_1...beta_K, comp_speed_1...comp_speed_k]
        """
        return np.concatenate(
            [self.D, self.H_mag, self.beta(), self.comp_speed, self.tr_power],
            axis=1,
        )

    def beta(self):
        return self.beta_from_lam(self.lam)

    def beta_from_lam(self, lam):
        epsilon = c.compression_constant

        lam_sum = lam[:, :, 0] + lam[:, :, 1]

        beta_values = (1 / epsilon) * np.log(
            (self.comp_speed / (epsilon * (self.D + 1e-12))) * lam_sum
        )

        beta_values = np.nan_to_num(
            beta_values, nan=1.0, posinf=self.beta_max, neginf=1.0
        )

        return np.clip(beta_values, 1, self.beta_max)

    def r(self):
        return self.b * np.log2(1 + (self.H_mag**2 * self.tr_power) / (c.N0 * self.b))

    def mu(self):
        return self.mu_from_lam(self.lam)

    def mu_from_lam(self, lam):
        lam2 = lam[:, :, 1]
        r = self.r()

        return np.sqrt(lam2 * r / self.D)

    def dual_values(self, lam):
        epsilon = c.compression_constant

        beta = self.beta_from_lam(lam)
        mu = self.mu_from_lam(lam)
        r = self.r()

        objective = (self.D / self.comp_speed) * np.exp(beta * epsilon) + (
            self.D / r
        ) * mu

        g1 = 1 - beta
        g2 = 1 / (mu) - beta

        return objective + lam[:, :, 0] * g1 + lam[:, :, 1] * g2

    def leader_optimization(self, model):

        if model == "cvx":
            self.leader_optimization_cvx()
            return

        X = self.to_X()

        device = next(model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        model.eval()

        b, f = model(X_tensor)

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)

    def leader_optimization_cvx(self, verbose=False):
        eps = 1e-8

        beta_all = np.clip(self.beta(), 1.0, self.beta_max)

        D_all = np.maximum(self.D, 1e-9)
        H_all = np.maximum(self.H_mag, 1e-20)
        p_all = np.maximum(self.tr_power, 1e-12)
        comp_speed_all = np.maximum(self.comp_speed, 1e-9)

        beta_all = np.nan_to_num(beta_all, nan=1.0, posinf=self.beta_max, neginf=1.0)

        D_all = np.nan_to_num(D_all, nan=1e-9, posinf=1e9, neginf=1e-9)
        H_all = np.nan_to_num(H_all, nan=1e-20, posinf=1e2, neginf=1e-20)
        p_all = np.nan_to_num(p_all, nan=1e-12, posinf=1e2, neginf=1e-12)
        comp_speed_all = np.nan_to_num(
            comp_speed_all, nan=1e-9, posinf=1e12, neginf=1e-9
        )

        # Keep old values if a scenario fails
        b_result = self.b.copy()
        f_result = self.f.copy()

        failed_scenarios = []

        for n in range(self.N):
            beta = beta_all[n, :]
            D = D_all[n, :]
            H = H_all[n, :]
            p = p_all[n, :]
            comp_speed = comp_speed_all[n, :]

            g = (H**2) * p / c.N0
            g = np.maximum(g, 1e-20)

            eta = np.exp(np.clip(beta * c.compression_constant, None, 50.0)) - np.exp(
                c.compression_constant
            )

            T_comp = D * eta / comp_speed

            T_DT = cp.Variable(self.K)
            T_tr = cp.Variable(self.K)
            T = cp.Variable()

            b = cp.Variable(self.K)
            f = cp.Variable(self.K)

            constraints = [
                b >= eps,
                f >= eps,
                T_DT >= eps,
                T_tr >= eps,
                T >= eps,
                cp.sum(b) <= self.total_bandwidth,
                cp.sum(f) <= c.total_compute_power,
                f >= cp.multiply(c.dt_compute_complexity * D, cp.inv_pos(T_DT)),
                T >= T_DT + T_tr + T_comp,
            ]

            for k in range(self.K):
                constraints.append(
                    -cp.rel_entr(b[k], b[k] + g[k]) * cp.inv_pos(np.log(2))
                    >= D[k] * cp.inv_pos(beta[k] * T_tr[k])
                )

            problem = cp.Problem(cp.Minimize(T), constraints)

            try:
                problem.solve(
                    solver=cp.MOSEK,
                    verbose=verbose,
                    mosek_params={
                        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-6,
                        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-6,
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-6,
                        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
                    },
                )

                if (
                    problem.status in ["optimal", "optimal_inaccurate"]
                    and b.value is not None
                    and f.value is not None
                ):
                    b_result[n, :] = np.asarray(b.value).reshape(self.K)
                    f_result[n, :] = np.asarray(f.value).reshape(self.K)
                else:
                    failed_scenarios.append(n)
                    if verbose:
                        print(f"Skipping scenario {n}; MOSEK status = {problem.status}")

            except Exception as e:
                failed_scenarios.append(n)
                if verbose:
                    print(f"Skipping scenario {n}; MOSEK failed: {e}")

        self.b = b_result
        self.f = f_result

        if len(failed_scenarios) > 0:
            print(
                f"MOSEK skipped {len(failed_scenarios)} scenario(s): {failed_scenarios}"
            )

    def follower_optimization(
        self,
        num_iters=20,
        init_step=0.1,
        grow=2.0,
    ):
        for _ in range(num_iters):
            beta = self.beta()
            mu = self.mu()

            g = np.stack(
                [
                    1 - beta,
                    1 / (mu) - beta,
                ],
                axis=2,
            )

            original_lam = self.lam.copy()
            h = init_step

            dual = np.sum(self.dual_values(original_lam))

            new_lam = np.maximum(0, original_lam + h * g)
            new_dual = np.sum(self.dual_values(new_lam))

            while new_dual > dual:
                dual = new_dual
                h *= grow

                new_lam = np.maximum(0, original_lam + h * g)
                new_dual = np.sum(self.dual_values(new_lam))

            self.lam = np.maximum(0, original_lam + (h / grow) * g)

    def _torch_state(self, device=None):
        if device is None:
            device = "cpu"

        X = torch.tensor(self.to_X(), dtype=torch.float32, device=device)
        b = torch.tensor(self.b, dtype=torch.float32, device=device)
        f = torch.tensor(self.f, dtype=torch.float32, device=device)

        return X, b, f

    def max_completion_time(self, device=None):
        X, b, f = self._torch_state(device=device)
        return ProcessTime.t_max_completion(X, b, f, numpy=True)

    def optimize(
        self,
        model,
        max_iters=30,
        tol=1e-6,
        follower_iters=20,
    ):
        history = []

        prev_obj = None

        t_start_total = time.perf_counter()

        for it in range(max_iters):

            # 1. Leader optimization: update self.b and self.f
            t_start_leader = time.perf_counter()
            self.leader_optimization(model)
            t_end_leader = time.perf_counter()
            self.leader_time_used += [t_start_leader - t_end_leader]

            # 2. Follower optimization: update self.lam
            self.follower_optimization(num_iters=follower_iters)

            # 3. Compute current DT synchronization time
            max_time = self.max_completion_time(device=self.device)

            # Average over samples/scenarios
            obj = np.mean(max_time)

            if prev_obj is None:
                diff = np.inf
            else:
                diff = abs(obj - prev_obj)

            history.append(
                {
                    "iter": it,
                    "objective": obj,
                    "diff": diff,
                    "max_time_mean": np.mean(max_time),
                    "max_time_max": np.max(max_time),
                }
            )

            if diff <= tol:
                break

            prev_obj = obj

        t_end_total = time.perf_counter()

        self.total_time_used = t_end_total - t_start_total

        return pd.DataFrame(history)

    def to_df(self):
        X, b, f = self._torch_state()
        return AnalyzeUtil.to_df(X, b, f)

    def print_avg(self):
        X, b, f = self._torch_state()
        AnalyzeUtil.print_avg(X, b, f)

    def plot(self):
        X, b, f = self._torch_state()
        AnalyzeUtil.plot_result(X, b, f)

    def get_total_time_used(self) -> float:
        if self.total_time_used == 0:
            raise RuntimeError("The optimization is not performed yet")
        return self.total_time_used

    def get_leader_time_used(self) -> list:
        return self.leader_time_used

    def get_t_total_max(self):
        t_total = self.to_df()["$t_{total}$"]
        return t_total.max()
