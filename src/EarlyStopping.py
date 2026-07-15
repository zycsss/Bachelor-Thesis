class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path=None, relative=False):
        """
        Args:
            patience (int): How many epochs to wait after the latest best model occurs.
            min_delta (float): Minimum change to qualify as an improvement.
            relative (bool): If True, min_delta is interpreted as a fraction of
                the current best loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.relative = relative
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_state_dict = None

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
            self.remember_best_model(current_epoch, current_loss, model)
        elif not self.is_improvement(current_loss):
            # Loss didn't improve enough
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss improved
            self.remember_best_model(current_epoch, current_loss, model)
            self.counter = 0
        return self.early_stop, self.best_epoch

    def is_improvement(self, current_loss):
        if self.relative:
            min_delta = abs(self.best_loss) * self.min_delta
        else:
            min_delta = self.min_delta

        return current_loss < self.best_loss - min_delta

    def remember_best_model(self, current_epoch, current_loss, model):
        """Keep the current best model weights in memory.

        Args:
            current_epoch (int): the current epoch
            current_loss (float): the current loss
            model (nn.Module): the current model
        """
        self.best_epoch = current_epoch
        self.best_loss = current_loss
        self.best_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    def restore_best_model(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
