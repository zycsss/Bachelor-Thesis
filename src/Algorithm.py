import numpy as np
import pandas as pd
import torch
import Constants
from numpy.typing import NDArray
import ProcessTime
from IPython.core.display import display, Markdown
from AnalyzeUtil import plot_result, to_db


class Algorithm:
    def __init__(
        self,
        D: NDArray,
        H_mag: NDArray,
        beta_max=Constants.beta_max,
        p=None,
        comp_speed=None,
    ):
        self.D = self._as_2d(D)
        self.H_mag = self._as_2d(H_mag)

        self.N, self.K = self.D.shape

        if self.H_mag.shape != (self.N, self.K):
            raise ValueError("H_mag must have the same shape as D")

        self.p = self._init_param(
            p,
            default_value=Constants.transmit_power,
            name="p",
        )

        self.comp_speed = self._init_param(
            comp_speed,
            default_value=Constants.sensor_compression_speed,
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
                self.comp_speed
            ],
            axis=1,
        )

    def beta(self):
        return self.beta_from_lam(self.lam)

    def beta_from_lam(self, lam):
        epsilon = Constants.compression_constant

        lam_sum = lam[:, :, 0] + lam[:, :, 1]

        beta_values = (1 / epsilon) * np.log(
            (self.comp_speed / (epsilon * self.D)) * lam_sum
        )

        return np.clip(beta_values, 1, self.beta_max)

    def r(self):
        return self.b * np.log2(1 + (self.H_mag**2 * self.p) / (Constants.N0 * self.b))

    def mu(self):
        return self.mu_from_lam(self.lam)

    def mu_from_lam(self, lam):
        lam2 = lam[:, :, 1]
        r = self.r()

        return np.sqrt(lam2 * r / self.D)

    def dual_values(self, lam):
        epsilon = Constants.compression_constant

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
        X = self.to_X()

        device = next(model.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        b, f = model(X_tensor)

        self.b = b.detach().cpu().numpy().reshape(self.N, self.K)
        self.f = f.detach().cpu().numpy().reshape(self.N, self.K)

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
        verbose=True,
    ):
        history = []

        prev_obj = None

        for it in range(max_iters):
            # 1. Leader optimization: update self.b and self.f
            self.leader_optimization(model)

            # 2. Follower optimization: update self.lam
            self.follower_optimization(num_iters=follower_iters)

            # 3. Compute current DT synchronization time
            device = next(model.parameters()).device
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

        return pd.DataFrame(history)

    def to_df(self):
        X = torch.tensor(self.to_X(), dtype=torch.float32)
        b = torch.tensor(self.b, dtype=torch.float32)
        f = torch.tensor(self.f, dtype=torch.float32)

        beta = self.beta()

        return pd.DataFrame(
            {
                "D": np.mean(self.D, axis=0),
                "H_mag(dB)": to_db(np.mean(self.H_mag, axis=0)),
                "p": np.mean(self.p, axis=0),
                "comp_speed": np.mean(self.comp_speed, axis=0),
                "b": np.mean(self.b, axis=0),
                "f": np.mean(self.f, axis=0),
                "beta": np.mean(beta, axis=0),
                "lambda_1": np.mean(self.lam[:, :, 0], axis=0),
                "lambda_2": np.mean(self.lam[:, :, 1], axis=0),
                r"$t_{total}$": np.mean(
                    ProcessTime.t_total(X, b, f, numpy=True), axis=0
                ),
                r"$t_{comp}$": np.mean(ProcessTime.t_comp(X, b, f, numpy=True), axis=0),
                r"$t_{tr}$": np.mean(ProcessTime.t_tr(X, b, f, numpy=True), axis=0),
                r"$t_{dt}$": np.mean(ProcessTime.t_dt(X, b, f, numpy=True), axis=0),
            }
        )

    def display_df(self):
        display(Markdown(self.to_df().to_markdown(index=False, floatfmt=".4f")))

    def plot(self):
        X = torch.tensor(self.to_X(), dtype=torch.float32)
        b = torch.tensor(self.b, dtype=torch.float32)
        f = torch.tensor(self.f, dtype=torch.float32)
        plot_result(X, b, f)