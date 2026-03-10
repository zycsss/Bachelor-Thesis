import pandas as pd
import numpy as np


def generate_df(n_sensor, D, p, f_s, d, H, b, f, lam_0=0.1):
    """
    Generate a pandas DataFrame containing sensor parameters.

    This function builds a DataFrame with one row per sensor. For the
    parameters `D`, `p`, `f_s`, `d`, `b`, and `f`, if a single integer is
    provided, it is broadcast to all sensors. Otherwise, the function expects
    array-like inputs of length `n_sensor`.

    Parameters
    ----------
    n_sensor : int
        Number of sensors, i.e. the number of rows in the output DataFrame.
    D : int or array-like
        Sensor parameter `D`. If an integer is given, the same value is used
        for all sensors.
    p : int or array-like
        Sensor parameter `p`. If an integer is given, the same value is used
        for all sensors.
    f_s : int or array-like
        Sampling frequency or sensor-specific parameter `f_s`. If an integer
        is given, the same value is used for all sensors.
    d : int or array-like
        Sensor parameter `d`. If an integer is given, the same value is used
        for all sensors.
    H : array-like
        Sensor parameter `H`. This is assigned directly to the DataFrame and
        should have length `n_sensor`.
    b : int or array-like
        Sensor parameter `b`. If an integer is given, the same value is used
        for all sensors.
    f : int or array-like
        Sensor parameter `f`. If an integer is given, the same value is used
        for all sensors.
    lam_0 : float, optional
        Initial lambda value. Default is 0.1.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - `D`
        - `p`
        - `f_s`
        - `d`
        - `H`
        - `b`
        - `f`
        - `lam`

    Notes
    -----
    - The `lam` column is initialized with `np.ones((n_sensor, 2))`.
    - The parameter `lam_0` is currently not used in the function body.
    - All array-like inputs should be compatible with length `n_sensor`.

    Example
    -------
    >>> df = generate_df(
    ...     n_sensor=3,
    ...     D=10,
    ...     p=[1, 2, 3],
    ...     f_s=100,
    ...     d=5,
    ...     H=[0.1, 0.2, 0.3],
    ...     b=2,
    ...     f=50
    ... )
    >>> print(df)
    """
    df = pd.DataFrame(index=range(n_sensor))

    if np.isscalar(D):
        D = np.full(n_sensor, D)
    else:
        D = np.asarray(D)

    if np.isscalar(p):
        p = np.full(n_sensor, p)
    else:
        p = np.asarray(p)

    if np.isscalar(f_s):
        f_s = np.full(n_sensor, f_s)
    else:
        f_s = np.asarray(f_s)

    if np.isscalar(d):
        d = np.full(n_sensor, d)
    else:
        d = np.asarray(d)

    if np.isscalar(b):
        b = np.full(n_sensor, b)
    else:
        b = np.asarray(b)

    if np.isscalar(f):
        f = np.full(n_sensor, f)
    else:
        f = np.asarray(f)

    H = np.asarray(H)

    df['D'] = D
    df['p'] = p
    df['f_s'] = f_s
    df['d'] = d
    df['H'] = H
    df['b'] = b
    df['f'] = f
    df['lam'] = [np.ones(2) * lam_0 for _ in range(n_sensor)]
    return df