import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path="model/checkpoint.pth"):
        """
        Args:
            patience (int): How many epochs to wait after the latest best model occurs.
            min_delta (float): Minimum change to qualify as an improvement.
            path (str): Path for the checkpoint model to be saved.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_epoch, current_loss, model):
        """_summary_

        Args:
            current_epoch (int)
            current_loss (float)
            model (nn.module)

        Returns:
            bool: if the training should be stopped (the result hasn't been improved for several epochs)
            int: the epoch number of the best model
        """
        if self.best_loss is None:
            self.save_checkpoint(current_epoch, current_loss, model)
        elif current_loss > self.best_loss - self.min_delta:
            # Loss didn't improve enough
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss improved
            self.save_checkpoint(current_epoch, current_loss, model)
            self.counter = 0
        return self.early_stop, self.best_epoch

    def save_checkpoint(self, current_epoch, current_loss, model):
        """save model from current epoch as the best

        Args:
            current_epoch (int): the current epoch
            current_loss (float): the current loss
            model (nn.Module): the current model
        """
        self.best_epoch = current_epoch
        self.best_loss = current_loss
        torch.save(model.state_dict(), self.path)