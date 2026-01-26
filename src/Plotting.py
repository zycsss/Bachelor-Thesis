import matplotlib.pyplot as plt
import numpy as np


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
    ax.set_ylim(0, 30) 

    # Grid behind bars
    ax.grid(axis='y', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    # 5. Legend (Reversed to match visual stack)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', framealpha=1, edgecolor='black')

    plt.tight_layout()