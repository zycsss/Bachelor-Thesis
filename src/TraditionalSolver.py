import torch
import torch.nn as nn

import Constants as c
import ProcessTime as pt


class TraditionalLeaderOptimizer(nn.Module):
    """
    Traditional optimizer for the leader problem.

    Input X shape:
        [num_scenarios, 3K]

    Format:
        X[:, 0:K]      = D_1, ..., D_K
        X[:, K:2K]     = H_1, ..., H_K
        X[:, 2K:3K]    = beta_1, ..., beta_K

    Optimizes:
        b: bandwidth allocation
        f: DT compute allocation

    Objective:
        min_{b, f} max_k t_total_k(X, b, f)

    Constraints:
        sum_k b_k = total_bandwidth
        sum_k f_k = total_compute_speed
    """

    def __init__(
        self,
        K,
        B=None,
        C_dt=None,
        min_share=1e-4,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()

        self.K = K
        self.B = c.total_bandwidth if B is None else B
        self.C_dt = c.total_compute_speed if C_dt is None else C_dt

        self.min_share = min_share
        self.dtype = dtype
        self.device = device

    def _allocate(self, logits, total_resource):
        """
        Convert unconstrained logits into feasible resource allocation.

        Guarantees:
            allocation_k > 0
            sum_k allocation_k = total_resource
        """
        raw_share = torch.softmax(logits, dim=1)
        share = self.min_share + (1.0 - self.K * self.min_share) * raw_share
        allocation = total_resource * share

        return allocation, share

    def _objective(self, X, b_logits, f_logits, tau=None):
        b, _ = self._allocate(b_logits, self.B)
        f, _ = self._allocate(f_logits, self.C_dt)

        T = pt.t_total(X, b, f)

        if tau is None:
            return T.max(dim=1).values.mean()

        return (tau * torch.logsumexp(T / tau, dim=1)).mean()

    def solve(
        self,
        X,
        steps=1000,
        lr=1e-2,
        tau=None,
        optimizer_type="adam",
        verbose=False,
        return_history=True,
    ):
        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        b_logits = torch.zeros(
            batch_size,
            self.K,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        f_logits = torch.zeros(
            batch_size,
            self.K,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        params = [b_logits, f_logits]
        loss_history = []

        if optimizer_type.lower() == "adam":
            opt = torch.optim.Adam(params, lr=lr)

            for step in range(steps):
                opt.zero_grad()

                loss = self._objective(X, b_logits, f_logits, tau=tau)
                loss.backward()
                opt.step()

                if return_history:
                    loss_history.append(loss.item())

                if verbose and step % 100 == 0:
                    print(f"step={step}, loss={loss.item():.6e}")

        elif optimizer_type.lower() == "lbfgs":
            opt = torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=steps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                loss = self._objective(X, b_logits, f_logits, tau=tau)
                loss.backward()
                return loss

            loss = opt.step(closure)

            if return_history:
                loss_history.append(loss.item())

        else:
            raise ValueError("optimizer_type must be 'adam' or 'lbfgs'.")

        with torch.no_grad():
            b, b_share = self._allocate(b_logits, self.B)
            f, f_share = self._allocate(f_logits, self.C_dt)

            T_comp = pt.t_comp(X, b, f)
            T_tr = pt.t_tr(X, b, f)
            T_dt = pt.t_dt(X, b, f)
            T_total = pt.t_total(X, b, f)

            max_time = T_total.max(dim=1).values

        result = {
            "b": b,
            "f": f,
            "b_share": b_share,
            "f_share": f_share,
            "T_comp": T_comp,
            "T_tr": T_tr,
            "T_dt": T_dt,
            "T_total": T_total,
            "max_time": max_time,
        }

        if return_history:
            result["loss_history"] = loss_history

        return result


class TraditionalLeaderModel(nn.Module):
    """
    Wrapper so the traditional solver works like a neural network model.

    Usage:
        model = TraditionalLeaderModel(K)
        b, f = model(X)
    """

    def __init__(
        self,
        K,
        B=None,
        C_dt=None,
        min_share=1e-4,
        dtype=torch.float32,
        device="cpu",
        steps=1000,
        lr=1e-2,
        tau=None,
        optimizer_type="adam",
        verbose=False,
    ):
        super().__init__()

        self.solver = TraditionalLeaderOptimizer(
            K=K,
            B=B,
            C_dt=C_dt,
            min_share=min_share,
            dtype=dtype,
            device=device,
        )

        self.steps = steps
        self.lr = lr
        self.tau = tau
        self.optimizer_type = optimizer_type
        self.verbose = verbose

        # dummy parameter so next(model.parameters()).device works
        self.dummy = nn.Parameter(torch.empty(0, device=device, dtype=dtype))

    def forward(self, X):
        result = self.solver.solve(
            X,
            steps=self.steps,
            lr=self.lr,
            tau=self.tau,
            optimizer_type=self.optimizer_type,
            verbose=self.verbose,
            return_history=False,
        )

        return result["b"], result["f"]
