from generate_sensor_list import generate_df
import const
import numpy as np
import math
import sensor_list


n = 4

shape = (n)

theta = np.random.randn(shape) * 2 * np.pi
real_part = np.random.randn(shape) * const.sigma
imag_part = np.random.randn(shape) * const.sigma

CN = real_part + 1j * imag_part
NLoS = math.sqrt(1 / (const.kappa + 1)) * CN
LoS = (
    math.sqrt(const.kappa / (const.kappa + 1))
    * const.sigma
    * np.exp(1j * theta)
)

h = LoS + NLoS
H = (
    math.sqrt(const.A0)
    * const.d ** (-const.alpha / 2)
    * h
)

df = generate_df(n, const.D, const.p, const.f_s, const.d, H, const.B_total / n, const.C_DT / n)

print(sensor_list.beta(df, 3.0))