import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ProcessTime import t_comp, t_dt, t_tr, t_total
from IPython.core.display import display, Markdown


def plot_result(t_comp, t_tr, t_dt):
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
    
    
def print_avg(X, b, f, beta):
    # 1. Calculate the means (Move to CPU/Numpy)
    #    Assumes shape is (N_sensors, ...) or just (N_sensors,)
    K = b.shape[1]
    D_k = X[:, :K]
    H_k_mag = X[:, K:]

    D_mean = D_k.mean(0).cpu().detach().numpy()
    H_mean = np.log10(H_k_mag.mean(0).cpu().detach().numpy())
    b_mean = b.mean(0).cpu().detach().numpy()
    f_mean = f.mean(0).cpu().detach().numpy()
    beta_mean = beta.mean(0).cpu().detach().numpy()
    t_total_mean = t_total(X, b, f, beta).mean(0).cpu().detach().numpy()
    t_comp_mean = t_comp(X, b, f, beta).mean(0).cpu().detach().numpy()
    t_tr_mean = t_tr(X, b, f, beta).mean(0).cpu().detach().numpy()
    t_dt_mean = t_dt(X, b, f, beta).mean(0).cpu().detach().numpy()

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
        val_D = D_mean if D_mean.ndim == 0 else D_mean[k]
        val_H = H_mean if H_mean.ndim == 0 else H_mean[k]
        val_b = b_mean if b_mean.ndim == 0 else b_mean[k]
        val_f = f_mean if f_mean.ndim == 0 else f_mean[k]
        val_beta = beta_mean if beta_mean.ndim == 0 else beta_mean[k]
        val_t_comp = t_comp_mean if t_comp_mean.ndim == 0 else t_comp_mean[k]
        val_t_tr = t_tr_mean if t_tr_mean.ndim == 0 else t_tr_mean[k]
        val_t_dt = t_dt_mean if t_dt_mean.ndim == 0 else t_dt_mean[k]
        val_t_total = t_total_mean if t_total_mean.ndim == 0 else t_total_mean[k]

        rows.append(
            {
                "sensor": str(k),
                "D [kbit]": val_D / 1e3 if np.isscalar(val_D) else str(val_D / 1e3),
                "H [dB]": val_H * 10 if np.isscalar(val_H) else str(val_H * 10),
                # wrap in str() if it's an array to fix ValueError
                "b [kHz]": val_b / 1e3 if np.isscalar(val_b) else str(val_b / 1e3),
                "f [kHz]": val_f / 1e3 if np.isscalar(val_f) else str(val_f / 1e3),
                "beta": val_beta if np.isscalar(val_beta) else str(val_beta),
                r"$\bar{T}^{comp}$": (
                    val_t_comp if np.isscalar(val_t_comp) else str(val_t_comp)
                ),
                r"$\bar{T}^{tr}$": val_t_tr if np.isscalar(val_t_tr) else str(val_t_tr),
                r"$\bar{T}^{DT}$": val_t_dt if np.isscalar(val_t_dt) else str(val_t_dt),
                r"$\bar{T}^{total}$": (
                    val_t_total if np.isscalar(val_t_total) else str(val_t_total)
                ),
            }
        )

    # 4. Create DataFrame and print
    df = pd.DataFrame(rows)
    display(Markdown(df.to_markdown(index=False, floatfmt=".2f")))
