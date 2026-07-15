import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
import ProcessTime
import AnalyzeUtil
import cvxpy as cp
import numpy as np
import Constants as c
from DataUtil import generate_data_loader
from TrainingUtil import unsupervised_loss, train, set_seed
from torch.optim import AdamW, lr_scheduler
from EarlyStopping import EarlyStopping
from NNSolver import Net, allocate_f_from_A, allocate_uniform_f_dt
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
        print_losses=True,
        n_epoch=100,
        train_length=int(1e4),
        uniform_f_dt=False,
    ):

        set_seed()

        model = Net(
            self.K,
            self.total_bandwidth,
            c.total_compute_power,
            self.beta_max,
            uniform_f_dt=uniform_f_dt,
        ).to(self.device)

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

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        warmup_epochs = max(1, n_epoch // 20)
        decay_epochs = max(1, n_epoch - warmup_epochs)
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=decay_epochs,
                    eta_min=1e-6,
                ),
            ],
            milestones=[warmup_epochs],
        )

        early_stopper = EarlyStopping(patience=30, min_delta=1e-9, relative=True)

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
            return self.leader_optimization_cvx()

        if model == "uniform":
            self.leader_optimization_uniform()
            return

        if model == "uniform_f_dt":
            self.leader_optimization_uniform_f_dt()
            return

        t_start_leader = time.perf_counter()
        
        X = self.to_X()

        device = next(model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        model.eval()

        b, f = model(X_tensor)

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)
        
        t_end_leader = time.perf_counter()
        leader_time_used = t_end_leader - t_start_leader
        
        return leader_time_used / self.N

    def leader_optimization_uniform(self):
        X = torch.tensor(self.to_X(), dtype=torch.float32, device=self.device)
        b = torch.full(
            (self.N, self.K),
            self.total_bandwidth / self.K,
            dtype=torch.float32,
            device=self.device,
        )

        A = ProcessTime.t_comp(X, b, None) + ProcessTime.t_tr(X, b, None)
        D_k = X[:, : self.K]
        f = allocate_f_from_A(
            A=A,
            D_k=D_k,
            C_DT_total=c.total_compute_power,
        )

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)

    def leader_optimization_uniform_f_dt(self):
        X = torch.tensor(self.to_X(), dtype=torch.float32, device=self.device)
        b = torch.full(
            (self.N, self.K),
            self.total_bandwidth / self.K,
            dtype=torch.float32,
            device=self.device,
        )
        D_k = X[:, : self.K]
        f = allocate_uniform_f_dt(
            D_k=D_k,
            C_DT_total=c.total_compute_power,
        )

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)

    def leader_optimization_cvx(self, verbose=False):
        eps = 1e-8
        solver_time_used = 0.0

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
                t_start_solve = time.perf_counter()
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
                solve_wall_time = time.perf_counter() - t_start_solve
                solve_time = problem.solver_stats.solve_time
                if solve_time is None:
                    solve_time = solve_wall_time
                solver_time_used += solve_time

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
        
        return solver_time_used / self.N

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
            
            leader_time_used_per_scenario = self.leader_optimization(model)
            if leader_time_used_per_scenario is not None:
                self.leader_time_used.append(leader_time_used_per_scenario)
            

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
        
    def print_one_sample(self, sample_idx):
        X, b, f = self._torch_state()
        AnalyzeUtil.print_avg(X[sample_idx, :].reshape(1, -1), b[sample_idx, :].reshape(1, -1), f[sample_idx, :].reshape(1, -1))

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
        return np.mean(self.max_completion_time(device=self.device))
