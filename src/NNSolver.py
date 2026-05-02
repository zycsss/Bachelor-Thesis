import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, K, B_total, C_DT_total, beta_max):
        super().__init__()
        self.K = K
        self.B_total = B_total
        self.C_DT_total = C_DT_total
        self.beta_max = beta_max

        # --- 1. Shared Encoder (The "Eyes" of the model) ---
        # We process each sensor identically first.
        # Input: 3 features (1 Normalized Data + 1 Log Channel gain + 1 beta)
        self.sensor_net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )

        # --- 2. Global Context (The "Brain") ---
        # Summarizes the whole system (e.g., "Are we all struggling?")
        self.context_net = nn.Sequential(nn.Linear(64*3, 64), nn.ReLU())

        # --- 3. Decoders (The "Hands") ---
        # Input: 64 (Local Sensor State) + 64 (Global System State)
        self.head_b_delta = nn.Linear(128, 1)
        self.head_f_delta = nn.Linear(128, 1)

        # Initialize to 0.0 -> Start exactly at uniform allocation
        for m in [self.head_b_delta, self.head_f_delta]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: [Batch, 2*K] concatenated (Data_1...Data_K, H_1...H_K)
        """
        # 1. Unpack and Preprocess Inputs (CRITICAL FIX)
        # Split input into Data and Channel parts
        D_k = x[:, : self.K]  # Shape: [Batch, K]
        H_k_mag = x[:, self.K : self.K * 2]  # Shape: [Batch, K]
        beta = x[:, self.K * 2 :]  # Shape: [Batch, K]

        # A. Normalize Data: Scale 4000 -> 1.0 range
        D_norm = D_k / 1e3

        # B. Log-Scale Channel: Scale 1e-6 -> -6.0 range
        H_log = torch.log10(H_k_mag + 1e-12)

        # Stack for Symmetric Processing: [Batch, K, 3]
        sensor_inputs = torch.stack([D_norm, H_log, beta], dim=2)

        # 2. Encode Each Sensor (Shared Weights)
        # [Batch, K, 64]
        local_feats = self.sensor_net(sensor_inputs)

        # --- 3. Global Context (Distribution Aware) ---
        # A. Max Pooling (Identify Bottleneck)
        global_max, _ = torch.max(local_feats, dim=1)  # [Batch, 64]

        # B. Mean Pooling (Identify Average Load)
        global_mean = torch.mean(local_feats, dim=1)  # [Batch, 64]
        
        global_std = 10 ** torch.std(local_feats, dim=1)  # [Batch, 64]

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
        b_k = softmax_with_floor_and_temp(logits_b, self.B_total)

        # --- Compute (Same Logic) ---
        logits_f = self.head_f_delta(combined).squeeze(2)
        f_dt_k = softmax_with_floor_and_temp(logits_f, self.C_DT_total)

        return b_k, f_dt_k


def softmax_with_floor_and_temp(logits, total_budget, min_share=1e-6, temp=0.5):
    """
    logits: [batch, K]
    total_budget: scalar or [batch, 1]
    min_share: each sensor gets at least this fraction
    """
    K = logits.shape[1]
    raw = torch.softmax(logits / temp, dim=1)

    # Each sensor gets at least min_share.
    # Remaining budget is distributed by softmax.
    share = min_share + (1.0 - K * min_share) * raw

    return total_budget * share
