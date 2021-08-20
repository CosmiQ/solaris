import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from .torch_callbacks import torch_callback_dict
import torch


def get_callbacks(framework, config):
    """Load callbacks based on a config file for a specific framework.

    Usage
    -----
    Note that this function is primarily intended for use with Keras. PyTorch
    does not use the same object-oriented training approach as Keras, and
    therefore doesn't generally have the same checkpointing objects to pass to
    model compilers - instead these are defined in model training code. See
    solaris.nets.train for examples of this. The only torch callback
    instantiated here is a learning rate scheduler.

    Arguments
    ---------
    framework : str
        Deep learning framework used for the model. Options are
        ``['keras', 'torch']`` .
    config : dict
        Configuration dict generated from the YAML config file.

    Returns
    -------
    callbacks : list
        A `list` of callbacks to pass to the compiler (Keras) or to wrap the
        optimizer (torch learning rate scheduling) for model training.
    """

    callbacks = []

    if framework == 'keras':
        for callback, params in config['training']['callbacks'].items():
            if callback == 'lr_schedule':
                callbacks.append(get_lr_schedule(framework, config))
            else:
                callbacks.append(keras_callbacks[callback](**params))
    elif framework == 'torch':
        for callback, params in config['training']['callbacks'].items():
            if callback == 'lr_schedule':
                callbacks.append(get_lr_schedule(framework, config))
            else:
                callbacks.append(torch_callback_dict[callback](**params))

    return callbacks


class KerasTerminateOnMetricNaN(Callback):
    """Callback to stop training if a metric has value NaN or infinity.

    Notes
    -----
    Instantiate as you would any other keras callback. For example, to end
    training if a validation metric called `f1_score` reaches value NaN::

        m = Model(inputs, outputs)
        m.compile()
        m.fit(X, y, callbacks=[TerminateOnMetricNaN('val_f1_score')])


    Attributes
    ----------
    metric : str, optional
        Name of the metric being monitored.
    checkpoint : str, optional
        One of ``['epoch', 'batch']``: Should the metric be checked at the end
        of every epoch (default) or every batch?

    Methods
    -------
    on_epoch_end : operations to complete at the end of each epoch.
    on_batch_end : operations to complete at the end of each batch.
    """

    def __init__(self, metric=None, checkpoint='epoch'):
        """

        Parameters
        ----------
        metric (str): The name of the metric to be tested for NaN value.
        checkpoint (['epoch', 'batch']): Should the metric be checked at the end of
            every epoch (default) or every batch?

        """
        super(KerasTerminateOnMetricNaN, self).__init__()
        self.metric = metric
        self.ckpt = checkpoint

    def on_epoch_end(self, epoch, logs=None):
        if self.ckpt == 'epoch':
            logs = logs or {}
            metric_score = logs.get(self.metric)
            if self.metric is not None:
                if np.isnan(metric_score) or np.isinf(metric_score):
                    print('Epoch {}: Invalid score for metric {}, terminating'
                          ' training'.format(epoch, self.metric))
                    self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        if self.ckpt == 'batch':
            logs = logs or {}
            metric_score = logs.get(self.metric)
            print('metric score: {}'.format(metric_score))
            if np.isnan(metric_score) or np.isinf(metric_score):
                print('Batch {}: Invalid score for metric'
                      '{}, terminating training'.format(batch, self.metric))
                self.model.stop_training = True


keras_callbacks = {
    'terminate_on_nan': keras.callbacks.TerminateOnNaN,
    'terminate_on_metric_nan': KerasTerminateOnMetricNaN,
    'model_checkpoint': keras.callbacks.ModelCheckpoint,
    'early_stopping': keras.callbacks.EarlyStopping,
    'reduce_lr_on_plateau': keras.callbacks.ReduceLROnPlateau,
    'csv_logger': keras.callbacks.CSVLogger
    }


def get_lr_schedule(framework, config):
    """Get a LR scheduling function for model training.

    Arguments
    ---------
    framework : str
        Deep learning framework used for the model. Options are
        ``['keras', 'torch']`` .
    config : dict
        Configuration dict generated from the YAML config file.

    Returns
    -------
    lr_scheduler : :class:`tensorflow.keras.callbacks.LearningRateScheduler` or
    ``torch.optim.lr_schedule`` scheduler class
        A scheduler to provide during training. For Keras, this takes the form
        of a callback passed to the optimizer; for PyTorch, it's a class object
        that wraps the optimizer. Because the torch version must wrap the
        optimizer, it's not instantiated here - the class is returned instead.

    """

    schedule_type = config['training'][
        'callbacks']['lr_schedule']['schedule_type']
    initial_lr = config['training']['lr']
    update_frequency = config['training']['callbacks']['lr_schedule'].get(
        'update_frequency', 1)
    factor = config['training']['callbacks']['lr_schedule'].get(
        'factor', 0)
    schedule_dict = config['training']['callbacks']['lr_schedule'].get(
        'schedule_dict', None)
    if framework == 'keras':
        lr_sched_func = keras_lr_schedule(schedule_type, initial_lr,
                                          update_frequency, factor,
                                          schedule_dict)
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_sched_func)
    elif framework == 'torch':
        # just get the class itself to use; don't instantiate until the
        # optimizer has been created.
        if config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'linear':
            lr_scheduler = torch.optim.lr_scheduler.StepLR
        elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
        elif config['training'][
                'callbacks']['lr_schedule']['schedule_type'] == 'arbitrary':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR

    return lr_scheduler


def keras_lr_schedule(schedule_type, initial_lr=0.001, update_frequency=1,
                      factor=0, schedule_dict=None):
    """Create a learning rate schedule for Keras from a schedule dict.

    Arguments
    ---------
    schedule_type : str
        Type of learning rate schedule to use. Options are:
        ``['arbitrary', 'exponential', 'linear']`` .
    initial_lr : float, optional
        The initial learning rate to use. Defaults to ``0.001`` .
    update_frequency : int, optional
        How frequently should learning rate be reduced (or increased)? Defaults
        to ``1`` (every epoch). Has no effect if ``schedule_type='arbitrary'``.
    factor : float, optional
        The magnitude by which learning rate should be changed at each update.
        Use a positive number to increase learning rate and a negative number
        to decrease learning rate. See Usage for more details.
    schedule_dict : dict, optional
        A dictionary with ``{epoch: learning rate}`` pairs. The learning rate
        defined in each pair will be used beginning at the specified epoch and
        continuing until the next highest epoch number is reached during
        training.

    Returns
    -------
    lr_schedule : func
        a function that takes epoch number integers as an argument and returns
        a learning rate.

    Usage
    -----
    ``schedule_type='arbitrary'`` usage is documented in the arguments above.
    For ``schedule_type='exponential'``, the following equation is applied to
    determine the learning rate at each epoch:

    .. math::

        lr = initial_lr*e^{factor\times(floor(epoch/update_frequency))}

    For ``schedule_type='linear'``, the following equation is applied:

    .. math::

        lr = initial_lr\times(1+factor\times(floor(epoch/update_frequency)))

    """
    if schedule_type == 'arbitrary':
        if schedule_dict is None:
            raise ValueError('If using an arbitrary schedule, an epoch: lr '
                             'dict must be provided.')
        lookup_dict = {}
        epoch_vals = np.array(list(schedule_dict.keys()))
        for e in range(0, epoch_vals.max() + 1):
            if e < epoch_vals.min():
                lookup_dict[e] = schedule_dict[epoch_vals.min()]
            else:
                # get all the epochs from the dict <= e
                lower_epochs = epoch_vals[epoch_vals <= e]
                # get the LR for the highest epoch number <= e
                lookup_dict[e] = schedule_dict[lower_epochs.max()]

        def lr_schedule(epoch):
            if epoch < epoch_vals.min():
                return initial_lr
            elif epoch > epoch_vals.max():
                return lookup_dict[epoch_vals.max()]
            else:
                return lookup_dict[epoch]

    elif schedule_type == 'exponential':
        def lr_schedule(epoch):
            if not np.floor(epoch/update_frequency):
                return initial_lr
            else:
                return initial_lr*factor/np.floor(epoch/update_frequency)

    elif schedule_type == 'linear':
        def lr_schedule(epoch):
            return initial_lr*(1+factor*np.floor(epoch/update_frequency))

    return lr_schedule
