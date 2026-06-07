import Constants as c
import torch
import math


def t_tr(X, b, f, numpy=False):
    K = b.shape[1]
    D_k = X[:, :K]
    H_k_mag = X[:, K:2*K]
    beta = X[:, 2*K:3*K]
    tr_power = X[:, 4*K:5*K]

    b = torch.clamp(b, min=1e-9)
    beta = torch.clamp(beta, min=1.0)
    N0 = c.N0

    snr = (H_k_mag**2 * tr_power) / (N0 * b + 1e-16)
    snr = torch.clamp(snr, min=0.0, max=1e30)

    r_k = b * torch.log2(1.0 + snr)
    r_k = torch.clamp(r_k, min=1e-12)

    T_tr = D_k / (beta * r_k)

    return T_tr.cpu().detach().numpy() if numpy else T_tr
    

def t_comp(X, b, f, numpy=False):
    K = b.shape[1]
    D_k = X[:, :K]
    beta = X[:, 2*K:3*K]
    f_S = X[:, 3*K:4*K]

    beta = torch.clamp(beta, min=1.0, max=10.0)
    f_S = torch.clamp(f_S, min=1e-9)

    epsilon = c.compression_constant

    eta = torch.exp(torch.clamp(beta * epsilon, max=50.0)) - math.exp(epsilon)

    T_comp = (D_k * eta) / f_S

    return T_comp.cpu().detach().numpy() if numpy else T_comp


def t_dt(X, b, f, numpy=False):
    K = b.shape[1]
    D_k = X[:, :K]
    c_k = c.dt_compute_complexity

    f = torch.clamp(f, min=1e-9)

    T_DT = (D_k * c_k) / f

    return T_DT.cpu().detach().numpy() if numpy else T_DT


def t_total(X, b, f, numpy=False):
    T_total = t_comp(X, b, f) + t_tr(X, b, f) + t_dt(X, b, f)
    
    if numpy:
        return T_total.cpu().detach().numpy()
    
    return T_total

def t_max_completion(X, b, f, numpy=False):
    T_max_completion = t_total(X, b, f).max(1).values
    
    if numpy:
        return T_max_completion.cpu().detach().numpy()
    
    return T_max_completion