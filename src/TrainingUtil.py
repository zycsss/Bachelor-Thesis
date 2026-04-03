import torch
import numpy as np
from ProcessTime import t_total


def logsumexp_loss(X, b, f, beta):

    T_total = t_total(X, b, f, beta)

    # --- NEW LOSS (LogSumExp) ---
    # This mathematically approximates max(T_total) smoothly.
    # 'alpha' controls sharpness.
    # alpha=1.0 is soft. alpha=10.0 is very close to strict Max.
    alpha = 10.0

    # LSE = (1/alpha) * log( sum( exp(alpha * T) ) )
    # We apply it per batch item (dim=1)
    lse = (1 / alpha) * torch.logsumexp(alpha * T_total, dim=1)

    # Minimize the average of the maximums across the batch
    loss = torch.mean(lse) * 10

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
