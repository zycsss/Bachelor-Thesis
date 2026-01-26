import torch
import numpy as np
from EarlyStopping import EarlyStopping
import Constants as c
import math
from torch import nn
import pandas as pd
from ProcessTime import t_total


def unsupervised_loss(X, b, f, beta):
    
    T_total = t_total(X, b, f, beta)

    # --- NEW LOSS (LogSumExp) ---
    # This mathematically approximates max(T_total) smoothly.
    # 'alpha' controls sharpness. 
    # alpha=1.0 is soft. alpha=10.0 is very close to strict Max.
    alpha = 5.0 
    
    # LSE = (1/alpha) * log( sum( exp(alpha * T) ) )
    # We apply it per batch item (dim=1)
    lse = (1/alpha) * torch.logsumexp(alpha * T_total, dim=1)
    
    # Minimize the average of the maximums across the batch
    loss = torch.mean(lse)
    
    return loss


def train_loop(
    model,
    data_loader,
    optimizer,
    loss_fn,
    device: str,
):
    model.to(device)
    model.train()
    total_loss = []

    for i, X in enumerate(data_loader):

        optimizer.zero_grad()
        X = X[0].to(device)
        # 2. Forward Pass
        # The model predicts optimal Bandwidth, Compute, and Compression based on the scenario
        b_pred, f_pred, beta_pred = model(X)  # Model takes |H_k|^2 as input
        # 3. Physics Calculation for Loss
        loss = loss_fn(X, b_pred, f_pred, beta_pred)

        # 5. Backward Pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


def train(
    model,
    data_loader,
    optimizer,
    scheduler,
    early_stopper,
    loss_fn,
    device: str,
    n_epoch=30,
    path_to_weight="model/checkpoint.pth",
):
    loss_list = []
    print("start training")
    for i in range(n_epoch):
        loss = train_loop(model, data_loader, optimizer, loss_fn, device)
        loss_list.append(loss)
        print(f"epoch: {i+1:02d}, avg loss: {loss}")
        scheduler.step(loss)
        should_stop, best_epoch = early_stopper(i, loss, model)
        if should_stop:

            break
    print(f"best epoch: {best_epoch+1:02d}, best avg loss: {loss_list[best_epoch]:>2f}")
    model.load_state_dict(torch.load(path_to_weight))


def print_avg(X, b, f, beta):
    # 1. Calculate the means (Move to CPU/Numpy)
    #    Assumes shape is (N_sensors, ...) or just (N_sensors,)
    b_mean = b.mean(0).cpu().detach().numpy()
    f_mean = f.mean(0).cpu().detach().numpy()
    beta_mean = beta.mean(0).cpu().detach().numpy()
    time_mean = t_total(X, b, f, beta).mean(0).cpu().detach().numpy()

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
        val_f = f_mean if f_mean.ndim == 0 else f_mean[k]
        val_beta = beta_mean if beta_mean.ndim == 0 else beta_mean[k]
        val_time = time_mean if time_mean.ndim == 0 else time_mean[k]

        rows.append({
            "sensor_num": k,
            # wrap in str() if it's an array to fix ValueError
            "b": val_b if np.isscalar(val_b) else str(val_b),
            "f": val_f if np.isscalar(val_f) else str(val_f),
            "beta": val_beta if np.isscalar(val_beta) else str(val_beta),
            "time": val_time if np.isscalar(val_time) else str(val_time),
        })

    # 4. Create DataFrame and print
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False, floatfmt=".4f"))