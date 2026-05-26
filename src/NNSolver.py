import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants as c
import ProcessTime


class Net(nn.Module):
    def __init__(self, K, B_total, C_DT_total, beta_max):
        super().__init__()
        self.K = K
        self.B_total = B_total
        self.C_DT_total = C_DT_total
        self.beta_max = beta_max

        # Shared Encoder 
        # We process each sensor identically first.
        # Input: 3 features (1 Normalized Data + 1 Log Channel gain + 1 beta)
        self.sensor_net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # Global Context
        # Summarizes the whole system (e.g., "Are we all struggling?")
        self.context_net = nn.Sequential(nn.Linear(64*3, 64), nn.ReLU())

        # Decoders
        # Input: 64 (Local Sensor State) + 64 (Global System State)
        self.head_b_delta = nn.Linear(128, 1)
        
        
        # f_dt residual decoder
        # combined has 128 features, and we add f_est_share as 1 extra feature
        self.f_decoder = nn.Sequential(
            nn.Linear(128 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        
        # Initialize b head
        nn.init.normal_(self.head_b_delta.weight, mean=0.0, std=1e-3)

        if self.head_b_delta.bias is not None:
            nn.init.constant_(self.head_b_delta.bias, 0.0)

        # Initialize f residual decoder close to zero
        last_f_layer = self.f_decoder[-1]

        if isinstance(last_f_layer, nn.Linear):
            nn.init.normal_(last_f_layer.weight, mean=0.0, std=1e-4)

            if last_f_layer.bias is not None:
                nn.init.constant_(last_f_layer.bias, 0.0)

    def forward(self, x):
        """
        x: [Batch, 2*K] concatenated (Data_1...Data_K, H_1...H_K)
        """
        # 1. Unpack and Preprocess Inputs (CRITICAL FIX)
        # Split input into Data and Channel parts
        D_k = x[:, : self.K]  # Shape: [Batch, K]
        H_k_mag = x[:, self.K : self.K * 2]  # Shape: [Batch, K]
        beta = x[:, self.K * 2 : self.K * 3]  # Shape: [Batch, K]
        comp_speed = x[:, self.K * 3 : self.K * 4]

        # A. Normalize Data: Scale 4000 -> 1.0 range
        D_norm = D_k / 1e3

        # B. Log-Scale Channel: Scale 1e-6 -> -6.0 range
        H_log = torch.log10(H_k_mag + 1e-12)
        
        comp_speed_norm = comp_speed / 200e3

        # Stack for Symmetric Processing: [Batch, K, 4]
        sensor_inputs = torch.stack([D_norm, H_log, beta, comp_speed_norm], dim=2)

        # 2. Encode Each Sensor (Shared Weights)
        # [Batch, K, 64]
        local_feats = self.sensor_net(sensor_inputs)

        # --- 3. Global Context (Distribution Aware) ---
        # A. Max Pooling (Identify Bottleneck)
        global_max, _ = torch.max(local_feats, dim=1)  # [Batch, 64]

        # B. Mean Pooling (Identify Average Load)
        global_mean = torch.mean(local_feats, dim=1)  # [Batch, 64]
        
        global_std = torch.std(local_feats, dim=1)  # [Batch, 64]

        # Concatenate to capture the full distribution shape
        # Shape: [Batch, 128]
        global_summary = torch.cat([global_max, global_mean, global_std], dim=1)

        # Process context
        global_context = self.context_net(global_summary)  # [Batch, 64]
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.K, -1)

        # 4. Combine & Decode
        combined = torch.cat([local_feats, global_context_expanded], dim=2)

        # 4. Predict Deviations
        # --- Bandwidth ---
        logits_b = self.head_b_delta(combined).squeeze(2)
        b_k = softmax_with_floor_and_temp(logits_b, self.B_total, min_share=1e-2, temp=0.1)

       
        # -------------------------
        # 2. Estimate f_dt share from predicted bandwidth
        # -------------------------
        f_est_share = estimate_f_share(
            x,
            b_k,
            self.K,
        )  # [Batch, K], sums to 1

        # -------------------------
        # 3. Let NN learn residual correction around f_est
        # -------------------------
        f_input = torch.cat(
            [
                combined,
                f_est_share.unsqueeze(2),
            ],
            dim=2,
        )  # [Batch, K, 129]

        delta_logits_f = self.f_decoder(f_input).squeeze(2)  # [Batch, K]

        # Physics estimate as baseline in logit space
        f_est_logits = torch.log(f_est_share + 1e-12)

        # Final f logits = estimated baseline + learned correction
        logits_f = f_est_logits + delta_logits_f

        f_dt_k = softmax_with_floor_and_temp(
            logits_f,
            self.C_DT_total,
            min_share=1e-5,
            temp=0.05,
        )

        return b_k, f_dt_k


def softmax_with_floor_and_temp(logits, total_budget, min_share=1e-2, temp=0.02):
    K = logits.shape[1]
    raw = torch.softmax(logits / temp, dim=1)

    share = min_share + (1.0 - K * min_share) * raw

    return total_budget * share

def estimate_f_share(X, b, K):

    D_k = X[:, :K]
    c_k = c.dt_compute_complexity

    
    T_comp = ProcessTime.t_comp(X, b, None)

    T_tr = ProcessTime.t_tr(X, b, None)

    # Non-DT delay
    A = T_comp + T_tr

    # Choose an estimated common finish time.
    # It must be larger than max(A), otherwise denominator becomes invalid.
    A_max = torch.max(A, dim=1, keepdim=True).values
    A_mean = torch.mean(A, dim=1, keepdim=True)
    A_std = torch.std(A, dim=1, keepdim=True)

    T_star_est = A_max + A_mean + 0.5 * A_std

    # Remaining time available for DT processing
    remaining = torch.clamp(T_star_est - A, min=1e-9)

    # Required compute score:
    # f_k ≈ D_k * c_k / (T_star - A_k)
    score = D_k * c_k / remaining
    
    score = torch.pow(score, 0.35)

    # Normalize to shares
    f_share = score / (score.sum(dim=1, keepdim=True) + 1e-12)

    return f_share