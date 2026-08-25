import torch
import numpy as np
from ProcessTime import t_total
import random


def unsupervised_loss(X, b, f):
    T_total = t_total(X, b, f)

    alpha = 5.0
    smooth_max_norm = (1 / alpha) * torch.logsumexp(alpha * T_total, dim=1)

    return torch.mean(smooth_max_norm)


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
        
        b_pred, f_pred = model(X)
        
        loss = loss_fn(X, b_pred, f_pred)

        loss.backward()
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

    with torch.no_grad():
        for i, X in enumerate(data_loader):
            X = X[0].to(device, non_blocking=True)
            b_pred, f_pred = model(X)
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
        scheduler.step()
        should_stop, best_epoch = early_stopper(i, test_loss, model)
        if should_stop:

            break
    print(f"best epoch: {best_epoch+1:02d}, best avg loss: {loss_list[best_epoch]:>2f}")
    early_stopper.restore_best_model(model)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
