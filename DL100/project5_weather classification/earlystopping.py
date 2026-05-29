import torch

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=0.0, verbose=False, path='./checkpoint.pt', mode='min'):
        self.patience = patience # Number of epochs to wait after last improvement
        self.min_delta = min_delta # Minimum change in the monitored metric to qualify as an improvement
        self.verbose = verbose # Whether to print messages about improvements and early stopping
        self.path = path # Path to save the best model checkpoint
        self.mode = mode # 'min' for minimizing the metric (e.g., loss), 'max' for maximizing the metric (e.g., accuracy)
        self.counter = 0 # Counter for epochs without improvement
        self.best_score = None # Best score seen so far
        self.early_stop = False # Flag to indicate whether to stop training

    def __call__(self, metric, model=None):
        if self.best_score is None:
            self.best_score = metric
            if model is not None:
                torch.save(model.state_dict(), self.path)
            return False

        if self.mode == 'min':
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta

        if improved:
            self.best_score = metric
            if model is not None:
                torch.save(model.state_dict(), self.path)
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: improved metric to {metric:.6f}, saving model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: no improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop