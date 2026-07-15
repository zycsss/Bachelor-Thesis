import matplotlib.pyplot as plt
import numpy as np
import ProcessTime as process
import pandas as pd
from ProcessTime import t_total, t_comp, t_tr, t_dt
import IPython.display


def _plot_result(t_comp, t_tr, t_dt):
    num_sensors = t_comp.shape[0]
    sensor_labels = np.arange(num_sensors)

    
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
    
def to_np(x):
    return x.detach().cpu().numpy()


def to_df(X, b, f):
    K = b.shape[1]

    data = {
        "sensor_num": np.arange(K),
        "D": to_np(X[0, :K]).astype(int),
        "H(dB)": to_db(to_np(X[:, K:2*K].mean(0))),
        "b": to_np(b.mean(0)),
        "f": to_np(f.mean(0)),
        "beta": to_np(X[:, 2*K:3*K].mean(0)),
        "comp_speed": to_np(X[:, 3*K:4*K].mean(0)),
        "tr_power": to_np(X[:, 4*K:5*K].mean(0)),
        r"$t_{total}$": to_np(t_total(X, b, f).mean(0)),
        r"$t_{comp}$": to_np(t_comp(X, b, f).mean(0)),
        r"$t_{tr}$": to_np(t_tr(X, b, f).mean(0)),
        r"$t_{dt}$": to_np(t_dt(X, b, f).mean(0)),
    }
    return pd.DataFrame(data)

def print_avg(X, b, f):
    
    fmt = {
        "sensor_num": ".0f",
        "D": ".2e",
        "H(dB)": ".2f",
        "b": ".3e",
        "f": ".3e",
        "beta": ".2f",
        "comp_speed": ".2e",
        "tr_power": ".2e",
        r"$t_{total}$": ".2f",
        r"$t_{comp}$": ".2f",
        r"$t_{tr}$": ".2f",
        r"$t_{dt}$": ".2f",
    }
    
    IPython.display.display(IPython.display.Markdown(to_df(X, b, f).to_markdown(index=False, floatfmt=fmt.values())))
    
def to_db(x):
    return np.log10(x)*10

def to_normal(x):
    return 10 ** (x / 10)
