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
from TrainingUtil import unsupervised_loss, train
from torch.optim import Adam, lr_scheduler
from EarlyStopping import EarlyStopping
from NNSolver import Net
import time



class Algorithm:
    def __init__(
        self,
        D: NDArray,
        H_mag: NDArray,
        comp_speed: NDArray,
        tr_power: NDArray,
        beta_max=c.beta_max,
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

        # Decision variables, shape: (N, K)
        self.b = np.ones((self.N, self.K), dtype=float)
        self.f = np.ones((self.N, self.K), dtype=float)

        # Dual variables, shape: (N, K, 2)
        # lam[:, :, 0] = lambda_1
        # lam[:, :, 1] = lambda_2
        self.lam = np.full((self.N, self.K, 2), 0.1, dtype=float)
        
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.time_used = 0


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
    
    def train_new_model(self):
        
        train_loader, test_loader = generate_data_loader(
            int(1e5), 
            self.K, self.K * 4e3, 
            comp_speed_range=[c.sensor_compression_speed/2, c.sensor_compression_speed*2], 
            tr_power_range=[c.transmit_power/2, c.transmit_power*2],
            beta_max=self.beta_max
            )

        model = Net(self.K, c.total_bandwidth, c.total_compute_speed, self.beta_max)
        
        optimizer = Adam(model.parameters(), lr=0.01)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1e-2, patience=3)
        early_stopper = EarlyStopping(patience=5, min_delta=0.1)
        loss_fn = unsupervised_loss

        train(model, train_loader, test_loader, optimizer, scheduler, early_stopper, loss_fn, self.device, n_epoch=100)
        
        return model

    def to_X(self):
        """
        Returns X with shape (N, 3K):
        [D_1...D_K, H_1...H_K, beta_1...beta_K, comp_speed_1...comp_speed_k]
        """
        return np.concatenate(
            [
                self.D,
                self.H_mag,
                self.beta(),
                self.comp_speed,
                self.tr_power
            ],
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
        g2 = 1 / mu - beta

        return objective + lam[:, :, 0] * g1 + lam[:, :, 1] * g2

    def leader_optimization(self, model):
        
        if model == 'cvx':
            self.leader_optimization_cvx()
            return
        elif model == 'new':
            model = self.train_new_model()
            return
        
        X = self.to_X()

        device = next(model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        model.eval()
        
        b, f = model(X_tensor)

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)
        
        
    def leader_optimization_cvx(self, verbose=False):
        beta = self.beta()

        D = self.D
        H = self.H_mag
        p = self.tr_power
        comp_speed = self.comp_speed

        g = (H ** 2) * p / c.N0

        eta = np.exp(beta * c.compression_constant) - np.exp(c.compression_constant)
        T_comp = D * eta / comp_speed

        T_DT = cp.Variable((self.N, self.K), nonneg=True)
        T_tr = cp.Variable((self.N, self.K), nonneg=True)
        T = cp.Variable(self.N, nonneg=True)

        b = cp.Variable((self.N, self.K), nonneg=True)
        f = cp.Variable((self.N, self.K), nonneg=True)

        rate = -cp.rel_entr(b, b + g) / np.log(2)

        constraints = [
            cp.sum(b, axis=1) <= c.total_bandwidth,
            cp.sum(f, axis=1) <= c.total_compute_speed,

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
            raise RuntimeError(f"Leader optimization failed: {problem.status}")

        self.b = np.asarray(b.value)
        self.f = np.asarray(f.value)
    

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
                    1 / mu - beta,
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
        
        start = time.perf_counter()

        for it in range(max_iters):
            # 1. Leader optimization: update self.b and self.f
            self.leader_optimization(model)

            # 2. Follower optimization: update self.lam
            self.follower_optimization(num_iters=follower_iters)

            # 3. Compute current DT synchronization time
            device = next(model.parameters()).device if model is not None else 'cpu'
            max_time = self.max_completion_time(device=device)

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
        
        end = time.perf_counter()
        
        self.time_used = end - start
        
        return pd.DataFrame(history)
    
    def get_X_b_f(self):
        X = torch.tensor(self.to_X(), dtype=torch.float32)
        b = torch.tensor(self.b, dtype=torch.float32)
        f = torch.tensor(self.f, dtype=torch.float32)
        return X,b,f

    def to_df(self):
        X, b, f = self.get_X_b_f()
        return AnalyzeUtil.to_df(X, b, f)

    def print_avg(self):
        X, b, f = self.get_X_b_f()
        AnalyzeUtil.print_avg(X, b, f)

    def plot(self):
        X, b, f = self.get_X_b_f()
        AnalyzeUtil.plot_result(X, b, f)
        
    def get_time_used(self):
        if self.time_used == 0:
            raise RuntimeError('The optimization is not performed yet')
        return self.time_used