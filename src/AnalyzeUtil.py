import matplotlib.pyplot as plt
import numpy as np
import ProcessTime as process
import pandas as pd
from ProcessTime import t_total, t_comp, t_tr, t_dt
from IPython.core.display import display, Markdown


def _plot_result(t_comp, t_tr, t_dt):
    num_sensors = t_comp.shape[0]
    sensor_labels = [f'Sensor {k}' for k in range(num_sensors)]

    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 6)) # Wider figure to fit multiple sensors
    bar_width = 0.6
    x_pos = np.arange(num_sensors) # [0, 1, 2, 3, 4]

    # 3. Plotting the Stack
    # Layer 1: Computation (Bottom)
    p1 = ax.bar(x_pos, t_comp, width=bar_width, 
                color='#FDBF2E', edgecolor='black', linewidth=1, 
                label=r'$T_k^{\mathrm{comp}}$')

    # Layer 2: Transmission (Middle) -> bottom is t_comp
    p2 = ax.bar(x_pos, t_tr, width=bar_width, bottom=t_comp, 
                color='#4DAF4A', edgecolor='black', linewidth=1, 
                label=r'$T_k^{\mathrm{tr}}$')

    # Layer 3: DT (Top) -> bottom is t_comp + t_tr
    p3 = ax.bar(x_pos, t_dt, width=bar_width, bottom=t_comp + t_tr, 
                color='#377EB8', edgecolor='black', linewidth=1, 
                label=r'$T_k^{\mathrm{DT}}$')

    # 4. Formatting
    ax.set_ylabel('completion time [s]', fontsize=12)
    ax.set_xlabel('Sensor Index', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sensor_labels) # Set labels to "Sensor 0", "Sensor 1"...

    # Grid behind bars
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # 5. Legend (Reversed to match visual stack)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', framealpha=1, edgecolor='black')

    plt.tight_layout()
    
def plot_result(X, b, f):
    _plot_result(
        process.t_comp(X, b, f, True).mean(0), process.t_tr(X, b, f, True).mean(0), process.t_dt(X, b, f, True).mean(0)
    )
    
def print_avg(X, b, f):
    # 1. Calculate the means (Move to CPU/Numpy)
    #    Assumes shape is (N_sensors, ...) or just (N_sensors,)
    K = b.shape[1]
    H = X[:, K:2*K]
    beta = X[:, 2*K:]
    b_mean = b.mean(0).cpu().detach().numpy()
    H_mean = to_db(H.mean(0).cpu().detach().numpy())
    f_mean = f.mean(0).cpu().detach().numpy()
    beta_mean = beta.mean(0).cpu().detach().numpy()
    time_mean = t_total(X, b, f).mean(0).cpu().detach().numpy()
    comp_mean = t_comp(X, b, f).mean(0).cpu().detach().numpy()
    tr_mean = t_tr(X, b, f).mean(0).cpu().detach().numpy()
    dt_mean = t_dt(X, b, f).mean(0).cpu().detach().numpy()

    rows = []
    
    # 2. Handle single sensor vs multiple sensors
    if b_mean.ndim == 0:
        # Case: Single sensor (Scalar values)
        count = 1
    else:
        # Case: Multiple sensors (Vector values)
        count = len(b_mean)

    # 3. Build the rows
    for k in range(count):
        # Helper to extract value safely: 
        # If it's 0-dim (scalar), use .item(). If it's a vector, access index [k].
        val_b = b_mean if b_mean.ndim == 0 else b_mean[k]
        val_H = H_mean if H_mean.ndim == 0 else H_mean[k]
        val_f = f_mean if f_mean.ndim == 0 else f_mean[k]
        val_beta = beta_mean if beta_mean.ndim == 0 else beta_mean[k]
        val_time = time_mean if time_mean.ndim == 0 else time_mean[k]
        val_comp = comp_mean if comp_mean.ndim == 0 else comp_mean[k]
        val_tr = tr_mean if tr_mean.ndim == 0 else tr_mean[k]
        val_dt = dt_mean if dt_mean.ndim == 0 else dt_mean[k]

        rows.append({
            "sensor_num": k,
            # wrap in str() if it's an array to fix ValueError
            "b": val_b if np.isscalar(val_b) else str(val_b),
            "H(dB)": val_H if np.isscalar(val_H) else str(val_H),
            "f": val_f if np.isscalar(val_f) else str(val_f),
            "beta": val_beta if np.isscalar(val_beta) else str(val_beta),
            r"$t_{total}$": val_time if np.isscalar(val_time) else str(val_time),
            r"$t_{comp}$": val_comp if np.isscalar(val_comp) else str(val_comp),
            r"$t_{tr}$": val_tr if np.isscalar(val_tr) else str(val_tr),
            r"$t_{dt}$": val_dt if np.isscalar(val_dt) else str(val_dt),
        })

    # 4. Create DataFrame and print
    df = pd.DataFrame(rows)
    display(Markdown(df.to_markdown(index=False, floatfmt=".4f")))
    
def to_db(x):
    return np.log10(x)*10

def to_normal(x):
    return 10 ** (x / 10)