from keras.callbacks import Callback
import numpy as np


def terminate_on_metric_nan(model, metric):
    """Get a terminate on metric NaN callback for `metric` in `model`.

    Arguments:
    ----------
    model : Keras, PyTorch, or TensorFlow model
        A model object to instantiate the callback for.
    metric : str
        Name of metric that should be tested for ``NaN`` or infinite values.

    Returns:
    A ``Callback`` or related object for the deep learning framework used to
    produce ``model`` to be used during model training.
    """
    # TODO: IMPLEMENT
    pass


class KerasTerminateOnMetricNaN(Callback):
    """Callback to stop training if a metric has value NaN or infinity.

    Arguments:
    ----------
    metric (str): The name of the metric to be tested for NaN value.
    checkpoint (['epoch', 'batch']): Should the metric be checked at the end of
        every epoch (default) or every batch?

    Usage:
    ------
    Instantiate as you would any other keras callback. For example, to end
    training if a validation metric called `f1_score` reaches value NaN:
    ```
    m = Model(inputs, outputs)
    m.compile()
    m.fit(X, y, callbacks=[TerminateOnMetricNaN('val_f1_score')])
    ```
    """
    def __init__(self, metric=None, checkpoint='epoch'):
        super(KerasTerminateOnMetricNaN, self).__init__()
        self.metric = metric
        self.ckpt = checkpoint

    def on_epoch_end(self, epoch, logs=None):
        if self.ckpt == 'epoch':
            logs = logs or {}
            metric_score = logs.get(self.metric)
            if self.metric is not None:
                if np.isnan(metric_score) or np.isinf(metric_score):
                    print('Epoch {}: Invalid score for metric'.format(epoch) +
                          '{}, terminating training'.format(self.metric))
                    self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        if self.ckpt == 'batch':
            logs = logs or {}
            metric_score = logs.get(self.metric)
            print('metric score: {}'.format(metric_score))
            if np.isnan(metric_score) or np.isinf(metric_score):
                print('Batch {}: Invalid score for metric'.format(batch) +
                      '{}, terminating training'.format(self.metric))
                self.model.stop_training = True
