import Constants as c
import torch
import math


def t_tr(X, b, f, beta, numpy=False):
    # Unpack
    K = b.shape[1]
    D_k = X[:, :K]
    H_k_mag = X[:, K:]

    p_k = c.transmit_power
    N0 = c.N0

    # Rate (Shannon)
    snr = (H_k_mag**2 * p_k) / (N0 * b)
    r_k = b * torch.log2(1 + snr)

    T_tr = D_k / (beta * r_k)

    if numpy:
        return T_tr.cpu().detach().numpy()

    return T_tr


def t_comp(X, b, f, beta, numpy=False):
    # Unpack
    K = b.shape[1]
    D_k = X[:, :K]

    epsilon = c.compression_constant
    f_S = c.sensor_compression_speed

    eta = torch.exp(beta * epsilon) - math.exp(epsilon)
    T_comp = (D_k * eta) / f_S

    if numpy:
        return T_comp.cpu().detach().numpy()

    return T_comp


def t_dt(X, b, f, beta, numpy=False):
    # Unpack
    K = b.shape[1]
    D_k = X[:, :K]
    c_k = c.dt_compute_complexity

    T_DT = (D_k * c_k) / (f)

    if numpy:
        return T_DT.cpu().detach().numpy()

    return T_DT


def t_total(X, b, f, beta, numpy=False):
    T_total = t_comp(X, b, f, beta) + t_tr(X, b, f, beta) + t_dt(X, b, f, beta)

    if numpy:
        return T_total.cpu().detach().numpy()

    return T_total
