import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import math
import Constants


def generate_channel_gains(length, K):

    shape = (length, K)

    theta_k = torch.rand(shape) * 2 * torch.pi
    real_part = torch.randn(shape) * Constants.sigma_k
    imag_part = torch.randn(shape) * Constants.sigma_k
    CN = torch.complex(real_part, imag_part)
    NLoS = math.sqrt(1 / (Constants.kappa + 1)) * CN
    LoS = (
        math.sqrt(Constants.kappa / (Constants.kappa + 1))
        * Constants.sigma_k
        * torch.exp(1j * theta_k)
    )
    h_k = LoS + NLoS
    H_k = math.sqrt(Constants.A0) * Constants.d_k ** (-Constants.alpha / 2) * h_k
    H_k = H_k

    return torch.abs(H_k)


def generate_D_k_distribution(length, K):
    raw = torch.rand(length, K)
    return raw / raw.sum(dim=1, keepdim=True)


def generate_beta(length, K, beta_max):
    # beta_min = 1.0
    beta = torch.rand(length, K) * (beta_max - 1.0) + 1.0
    return beta


def generate_comp_speed(length, K, range):
    return torch.rand(length, K) * (range[1] - range[0]) + range[0]


def generate_data(length, K, total_data_bits, beta_max, comp_speed_range):
    d_k = generate_D_k_distribution(length, K) * total_data_bits
    H_k = generate_channel_gains(length, K)
    beta = generate_beta(length, K, beta_max)
    comp_speed = generate_comp_speed(length, K, comp_speed_range)
    return torch.cat((d_k, H_k, beta, comp_speed), dim=1)


def generate_data_loader(
    length, K, total_data_bits, beta_max, comp_speed_range, batch_size=64, train_size=0.7
):
    dataset = TensorDataset(generate_data(length, K, total_data_bits, beta_max, comp_speed_range))
    train_dataset, test_dataset = random_split(dataset, [train_size, 1 - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
