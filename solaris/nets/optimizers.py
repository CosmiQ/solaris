"""Wrappers for training optimizers."""

import torch
from tensorflow import keras


torch_optimizers = {
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD,
}

keras_optimizers = {
    'adadelta': keras.optimizers.Adadelta,
    'adagrad': keras.optimizers.Adagrad,
    'adam': keras.optimizers.Adam,
    'adamax': keras.optimizers.Adamax,
    'nadam': keras.optimizers.Nadam,
    'rmsprop': keras.optimizers.RMSprop,
    'sgd': keras.optimizers.SGD
}


def get_optimizer(framework, config):
    """Get the optimizer specified in config for model training.

    Arguments
    ---------
    framework : str
        Name of the deep learning framework used. Current options are
        ``['torch', 'keras']``.
    config : dict
        The config dict generated from the YAML config file.

    Returns
    -------
    An optimizer object for the specified deep learning framework.
    """

    if config['training']['optimizer'] is None:
        raise ValueError('An optimizer must be specified in the config '
                         'file.')

    if framework in ['torch', 'pytorch']:
        return torch_optimizers.get(config['training']['optimizer'], None)
    elif framework == 'keras':
        return keras_optimizers.get(config['training']['optimizer'], None)
