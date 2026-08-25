import torch
import torch.nn as nn

import Constants as c
import ProcessTime


class Net(nn.Module):
    def __init__(self, K, B_total, C_DT_total, beta_max, uniform_f_dt=False):
        super().__init__()
        self.K = K
        self.B_total = B_total
        self.C_DT_total = C_DT_total
        self.beta_max = beta_max
        self.uniform_f_dt = uniform_f_dt

        n_feat = 64
        combined_feat = n_feat * 2

        self.sensor_net = nn.Sequential(
            nn.Linear(5, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, n_feat),
            nn.ReLU(),
        )

        self.context_net = nn.Sequential(
            nn.Linear(n_feat * 3, n_feat),
            nn.ReLU(),
        )

        self.head_b_delta = nn.Linear(combined_feat, 1)
        
        nn.init.normal_(self.head_b_delta.weight, mean=0.0, std=1e-2)
        if self.head_b_delta.bias is not None:
            nn.init.constant_(self.head_b_delta.bias, 0.0)

    def forward(self, x):
        D_k = x[:, : self.K]
        H_k_mag = x[:, self.K : self.K * 2]
        beta = x[:, self.K * 2 : self.K * 3]
        comp_speed = x[:, self.K * 3 : self.K * 4]
        tr_power = x[:, self.K * 4 : self.K * 5]

        D_norm = D_k / 1e3
        H_log = torch.log10(H_k_mag + 1e-12)
        comp_speed_norm = comp_speed / 200e3
        tr_power_log = torch.log10(tr_power + 1e-12)

        sensor_inputs = torch.stack(
            [
                D_norm,
                H_log,
                beta,
                comp_speed_norm,
                tr_power_log,
            ],
            dim=2,
        )

        local_feats = self.sensor_net(sensor_inputs)

        global_max, _ = torch.max(local_feats, dim=1)
        global_mean = torch.mean(local_feats, dim=1)
        global_std = torch.std(local_feats, dim=1)

        global_summary = torch.cat(
            [global_max, global_mean, global_std],
            dim=1,
        )

        global_context = self.context_net(global_summary)
        global_context = global_context.unsqueeze(1).expand(-1, self.K, -1)

        combined = torch.cat([local_feats, global_context], dim=2)
        
        logits_b = self.head_b_delta(combined).squeeze(2)

        b_k = softmax_with_floor_and_temp(
            logits_b,
            self.B_total,
            min_share=1e-5,
            temp=0.3,
        )

        if self.uniform_f_dt:
            f_dt_k = allocate_uniform_f_dt(
                D_k=D_k,
                C_DT_total=self.C_DT_total,
            )
        else:
            A = ProcessTime.t_comp(x, b_k, None) + ProcessTime.t_tr(x, b_k, None)

            f_dt_k = allocate_f_from_A(
                A=A,
                D_k=D_k,
                C_DT_total=self.C_DT_total,
            )

        return b_k, f_dt_k




def allocate_f_from_A(A, D_k, C_DT_total, num_iter=20):
    w = D_k * c.dt_compute_complexity

    low = torch.max(A, dim=1, keepdim=True).values + 1e-6
    high = low + torch.sum(w, dim=1, keepdim=True) / C_DT_total

    for _ in range(num_iter):
        mid = 0.5 * (low + high)

        f_mid = w / torch.clamp(mid - A, min=1e-9)
        f_sum = torch.sum(f_mid, dim=1, keepdim=True)

        too_much_compute = f_sum > C_DT_total

        low = torch.where(too_much_compute, mid, low)
        high = torch.where(too_much_compute, high, mid)

    T_star = high

    f = w / torch.clamp(T_star - A, min=1e-9)

    f = C_DT_total * f / (torch.sum(f, dim=1, keepdim=True) + 1e-12)

    return f


def allocate_uniform_f_dt(D_k, C_DT_total):
    return torch.full_like(D_k, C_DT_total / D_k.shape[1])


def softmax_with_floor_and_temp(logits, total_budget, min_share=1e-6, temp=0.1):
    K = logits.shape[1]

    logits = torch.clamp(logits, -30.0, 30.0)
    raw = torch.softmax(logits / temp, dim=1)

    share = min_share + (1.0 - K * min_share) * raw

    return total_budget * share
