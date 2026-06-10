import torch
import numpy as np
from ProcessTime import t_total
import random


def unsupervised_loss(X, b, f):
    T_total = t_total(X, b, f)

    T_mean = torch.mean(T_total, dim=1, keepdim=True)
    T_norm = T_total / (T_mean + 1e-12)

    alpha = 5.0
    smooth_max_norm = (1 / alpha) * torch.logsumexp(alpha * T_norm, dim=1)
    
    loss_norm = smooth_max_norm

    return torch.mean(T_mean.squeeze(1) * loss_norm)


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
        b_pred, f_pred = model(X)  # Model takes |H_k|^2 as input
        # 3. Physics Calculation for Loss
        loss = loss_fn(X, b_pred, f_pred)

        # 5. Backward Pass
        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf loss detected")
            print("b_pred min/max:", b_pred.min().item(), b_pred.max().item())
            print("f_pred min/max:", f_pred.min().item(), f_pred.max().item())

            T = t_total(X, b_pred, f_pred)
            print("T_total has nan:", torch.isnan(T).any().item())
            print("T_total has inf:", torch.isinf(T).any().item())
            print("T_total min/max:", T.min().item(), T.max().item())

            print("X min/max:", X.min().item(), X.max().item())
            raise RuntimeError("Stopping because loss is NaN/Inf")
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


def test_loop(
    model,
    data_loader,
    loss_fn,
    device: str,
):
    model.to(device)
    model.eval()
    total_loss = []

    for i, X in enumerate(data_loader):

        X = X[0].to(device)
        # 2. Forward Pass
        # The model predicts optimal Bandwidth, Compute, and Compression based on the scenario
        b_pred, f_pred = model(X)  # Model takes |H_k|^2 as input
        # 3. Physics Calculation for Loss
        loss = loss_fn(X, b_pred, f_pred)

        total_loss.append(loss.item())

    return np.mean(total_loss)


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    early_stopper,
    loss_fn,
    device: str,
    n_epoch=30,
    path_to_weight="model/checkpoint.pth",
    print_losses = True
):
    loss_list = []
    print("start training")
    for i in range(n_epoch):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_loop(model, test_loader, loss_fn, device)
        loss_list.append(test_loss)
        if print_losses:
            print(f"epoch: {i+1:02d}, avg loss: {test_loss}")
        scheduler.step(train_loss)
        should_stop, best_epoch = early_stopper(i, test_loss, model)
        if should_stop:

            break
    print(f"best epoch: {best_epoch+1:02d}, best avg loss: {loss_list[best_epoch]:>2f}")
    model.load_state_dict(torch.load(path_to_weight))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False