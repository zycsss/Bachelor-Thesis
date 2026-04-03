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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # --- 2. Global Context (The "Brain") ---
        # Summarizes the whole system (e.g., "Are we all struggling?")
        self.context_net = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

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
        # --- 1. Unpack and Preprocess Inputs ---
        # Split input into Data and Channel parts
        D_k = x[:, : self.K]  # Shape: [Batch, K]
        H_k_mag = x[:, self.K :]  # Shape: [Batch, K]
        
        # A. Get Sorting Indices based on Data Size (Ascending)
        # We sort by D because Data Load is the main driver of inequality
        sort_vals, sort_idx = torch.sort(D_k, dim=1) 

        # B. Reorder D and H to match the sorted order
        # We use gather to rearrange the columns according to sort_idx
        D_sorted = torch.gather(D_k, 1, sort_idx)
        H_sorted = torch.gather(H_k_mag, 1, sort_idx)

        # A. Log-Scale Data
        D_log = torch.log10(D_sorted + 1e-12)

        # B. Log-Scale Channel
        H_log = torch.log10(H_sorted + 1e-12)

        # Stack for Symmetric Processing: [Batch, K, 2]
        sensor_inputs = torch.stack([D_log, H_log], dim=2)

        # --- 2. Encode Each Sensor (Shared Weights) ---
        # BatchNorm1d expects [N, Features].
        # We merge Batch and K into N -> [Batch * K, 2]
        batch_size = sensor_inputs.shape[0]
        flat_inputs = sensor_inputs.view(-1, 2)

        # Pass through Sensor Net
        flat_feats = self.sensor_net(flat_inputs)  # Output: [Batch*K, 64]

        # Un-flatten back to [Batch, K, 64]
        local_feats = flat_feats.view(batch_size, self.K, -1)

        # --- 3. Global Context (Distribution Aware) ---
        # A. Max Pooling (Identify Bottleneck)
        global_max, _ = torch.max(local_feats, dim=1)  # [Batch, 64]

        # B. Mean Pooling (Identify Average Load)
        global_mean = torch.mean(local_feats, dim=1)  # [Batch, 64]

        # Concatenate to capture the full distribution shape
        # Shape: [Batch, 128]
        global_summary = torch.cat([global_max, global_mean], dim=1)

        # Process context
        global_context = self.context_net(global_summary)  # [Batch, 64]
        global_context_expanded = global_context.unsqueeze(1).expand(-1, self.K, -1)

        # Combine
        combined = torch.cat([local_feats, global_context_expanded], dim=2)

        # --- 4. Predict Deviations ---

        temp = 1

        safety_floor = 1e-3

        # --- Bandwidth ---
        logits_b = self.head_b_delta(combined).squeeze(2)
        dist_b = F.softmax(logits_b / temp, dim=1)
        safe_dist_b = (1 - safety_floor) * dist_b + (safety_floor / self.K)
        b_sorted = safe_dist_b * self.B_total

        # --- Compute (Same Logic) ---
        logits_f = self.head_f_delta(combined).squeeze(2)
        dist_f = F.softmax(logits_f / temp, dim=1)
        safe_dist_f = (1 - safety_floor) * dist_f + (safety_floor / self.K)
        f_sorted = safe_dist_f * self.C_DT_total

        # --- Compression (Tanh) ---
        delta_beta = torch.tanh(self.head_beta(combined).squeeze(2))

        # Map to [1.0, beta_max]
        mid_beta = (1.0 + self.beta_max) / 2.0
        range_beta = (self.beta_max - 1.0) / 2.0
        beta_sorted = mid_beta + (delta_beta * range_beta)
        
        # --- STEP 4: UNSORT (Restore Original Order) ---
        
        # We need to map the sorted results back to where the original sensors were.
        # scatter_(dim, index, src) puts values from 'src' into 'self' at 'index'
        
        # Create empty placeholders
        b_final = torch.zeros_like(b_sorted)
        f_final = torch.zeros_like(f_sorted)
        beta_final = torch.zeros_like(beta_sorted)
        
        # Unshuffle
        b_final.scatter_(1, sort_idx, b_sorted)
        f_final.scatter_(1, sort_idx, f_sorted)
        beta_final.scatter_(1, sort_idx, beta_sorted)

        return b_final, f_final, beta_final
