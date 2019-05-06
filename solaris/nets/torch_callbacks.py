"""PyTorch Callbacks."""


class EarlyStopping(object):
    """Tracks if model training should stop based on rate of improvement.

    Arguments
    ---------
    patience : int, optional
        The number of epochs to wait before stopping the model if the metric
        didn't improve. Defaults to 5.
    threshold : float, optional
        The minimum metric improvement required to count as "improvement".
        Defaults to ``0.0`` (any improvement satisfies the requirement).
    verbose : bool, optional
        Verbose text output. Defaults to off (``False``). _NOTE_ : This
        currently does nothing.
    """

    def __init__(self, patience=5, threshold=0.0, verbose=False):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric_score):

        if self.best is None:
            self.best = metric_score
            self.counter = 0
        else:
            if self.best - self.threshold < metric_score:
                self.counter += 1
            else:
                self.best = metric_score
                self.counter = 0

        if self.counter >= self.patience:
            self.stop = True
