import pandas as pd
import numpy as np
import const


import numpy as np

def r(df):
    b = df["b"].to_numpy()
    H = df["H"].to_numpy()
    p = df["p"].to_numpy()
    return b * np.log2(1 + (np.abs(H) ** 2) * p / (const.N0 * b))


def beta(df, beta_max):
    D = df["D"].to_numpy()
    f_s = df["f_s"].to_numpy()

    lam = np.stack(df["lam"].to_numpy())   # shape: (n, 2)
    lam_sum = lam.sum(axis=1)

    beta_vals = (1 / const.epsilon) * np.log(
        (f_s / (const.epsilon * D)) * lam_sum
    )
    beta_vals = np.minimum(beta_vals, beta_max)
    beta_vals = np.maximum(beta_vals, 1.0)

    return beta_vals


def mu(df):
    D = df["D"].to_numpy()

    lam = np.stack(df["lam"].to_numpy())   # shape: (n, 2)

    mu_vals = np.where(
        lam[:, 1] <= 0,
        1.0,
        np.sqrt(lam[:, 1] * r(df) / D)
    )

    return mu_vals