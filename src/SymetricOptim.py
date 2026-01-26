import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricResidualNet(nn.Module):
    def __init__(self, K, B_total, C_DT_total, beta_max):
        super().__init__()
        self.K = K
        self.B_total = B_total
        self.C_DT_total = C_DT_total
        self.beta_max = beta_max

        # --- 1. Shared Encoder (The "Eyes" of the model) ---
        # We process each sensor identically first.
        # Input: 2 features (1 Normalized Data + 1 Log Channel gain)
        self.sensor_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # --- 2. Global Context (The "Brain") ---
        # Summarizes the whole system (e.g., "Are we all struggling?")
        self.context_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # --- 3. Decoders (The "Hands") ---
        # Input: 64 (Local Sensor State) + 64 (Global System State)
        self.head_b_delta = nn.Linear(128, 1)
        self.head_f_delta = nn.Linear(128, 1)
        self.head_beta = nn.Linear(128, 1)

        # Initialize to 0.0 -> Start exactly at uniform allocation
        for m in [self.head_b_delta, self.head_f_delta, self.head_beta]:
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: [Batch, 2*K] concatenated (Data_1...Data_K, H_1...H_K)
        """
        # 1. Unpack and Preprocess Inputs (CRITICAL FIX)
        # Split input into Data and Channel parts
        D_k = x[:, :self.K]      # Shape: [Batch, K]
        H_k_mag = x[:, self.K:]  # Shape: [Batch, K]

        # A. Normalize Data: Scale 4000 -> 1.0 range
        D_norm = D_k / 1e3

        # B. Log-Scale Channel: Scale 1e-6 -> -6.0 range
        H_log = torch.log10(H_k_mag + 1e-12)

        # Stack for Symmetric Processing: [Batch, K, 2]
        sensor_inputs = torch.stack([D_norm, H_log], dim=2)

        # 2. Encode Each Sensor (Shared Weights)
        # [Batch, K, 64]
        local_feats = self.sensor_net(sensor_inputs)

        # --- 3. Global Context (Distribution Aware) ---
        # A. Max Pooling (Identify Bottleneck)
        global_max, _ = torch.max(local_feats, dim=1) # [Batch, 64]
        
        # B. Mean Pooling (Identify Average Load)
        global_mean = torch.mean(local_feats, dim=1)  # [Batch, 64]
        
        # Concatenate to capture the full distribution shape
        # Shape: [Batch, 128]
        global_summary = torch.cat([global_max, global_mean], dim=1)
        
        # Process context
        global_context = self.context_net(global_summary) # [Batch, 64]
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.K, -1)

        # 4. Combine & Decode
        combined = torch.cat([local_feats, global_context_expanded], dim=2)

        # 4. Predict Deviations (Softmax Replacement)
        
        # --- Constants for sharpness ---
        # Temperature < 1.0 allows for extreme inequality (Rich get richer)
        # Temperature > 1.0 forces equality
        temp = 0.1  
        
        # Safety floor: Ensure no one gets 0.0 (prevent infinite loss)
        epsilon = 0.01 

        # --- Bandwidth ---
        # 1. Get raw scores (logits) from the head
        # Squeeze converts [Batch, K, 1] -> [Batch, K]
        logits_b = self.head_b_delta(combined).squeeze(2)
        
        # 2. Apply Softmax with Temperature
        # dividing by 0.2 makes the differences 5x bigger before softmax
        raw_dist_b = F.softmax(logits_b / temp, dim=1)
        
        # 3. Apply Safety Floor (Mix 99% logic with 1% uniform)
        safe_dist_b = (1 - epsilon) * raw_dist_b + (epsilon / self.K)
        
        # 4. Allocate
        b_k = safe_dist_b * self.B_total

        # --- Compute (Same Logic) ---
        logits_f = self.head_f_delta(combined).squeeze(2)
        raw_dist_f = F.softmax(logits_f / temp, dim=1)
        safe_dist_f = (1 - epsilon) * raw_dist_f + (epsilon / self.K)
        f_dt_k = safe_dist_f * self.C_DT_total

        # --- Compression (Keep Tanh) ---
        # Compression is an absolute value (1.0 to 3.0), not a shared resource.
        # So Tanh is still correct here!
        delta_beta = torch.tanh(self.head_beta(combined).squeeze(2))
        
        # Map to [1.0, beta_max]
        mid_beta = (1.0 + self.beta_max) / 2.0
        range_beta = (self.beta_max - 1.0) / 2.0
        beta_k = mid_beta + (delta_beta * range_beta)

        return b_k, f_dt_k, beta_k