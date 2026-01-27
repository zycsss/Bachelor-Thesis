import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import math
import Constants

def generate_channel_gains(length, K):
    
    shape = (length, K)
    
    theta_k = torch.randn(shape) * 2 * torch.pi
    real_part = torch.randn(shape) * Constants.sigma_k
    imag_part = torch.randn(shape) * Constants.sigma_k
    CN = torch.complex(real_part, imag_part)
    NLoS = math.sqrt(1 / (Constants.kappa + 1)) * CN
    LoS = math.sqrt(Constants.kappa / (Constants.kappa + 1)) * Constants.sigma_k * torch.exp(1j * theta_k)
    h_k = LoS + NLoS
    H_k = math.sqrt(Constants.A0) * Constants.d_k ** (- Constants.alpha / 2) * h_k
    H_k = H_k
    
    return torch.abs(H_k)


def generate_D_k_distribution(length, K):
    raw = torch.rand(length, K)
    return raw / raw.sum(dim=1, keepdim=True)


# def generate_data(D_k_length, H_k_length, K, total_data_bits):
#     d_k = generate_D_k_distribution(D_k_length, K) * total_data_bits
#     H_k = generate_channel_gains(H_k_length, K)
#     d_k_extended = d_k.repeat_interleave(H_k_length, dim=0)
#     H_k_extended = H_k.repeat(D_k_length, 1)
#     return torch.cat((d_k_extended, H_k_extended), dim=1)


def generate_data(length, K, total_data_bits):
    d_k = generate_D_k_distribution(length, K) * total_data_bits
    H_k = generate_channel_gains(length, K)
    return torch.cat((d_k, H_k), dim=1)


def generate_data_loader(length, K, total_data_bits, batch_size):
    loader = DataLoader(
        TensorDataset(generate_data(length, K, total_data_bits)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader