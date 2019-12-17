import numpy as np
from tensorflow.keras import backend as K
from ._keras_losses import keras_losses, k_focal_loss
from ._torch_losses import torch_losses
from torch import nn


def get_loss(framework, loss, loss_weights=None, custom_losses=None):
    """Load a loss function based on a config file for the specified framework.

    Arguments
    ---------
    framework : string
        Which neural network framework to use.
    loss : dict
        Dictionary of loss functions to use.  Each key is a loss function name,
        and each entry is a (possibly-empty) dictionary of hyperparameter-value
        pairs.
    loss_weights : dict, optional
        Optional dictionary of weights for loss functions.  Each key is a loss
        function name (same as in the ``loss`` argument), and the corresponding
        entry is its weight.
    custom_losses : dict, optional
        Optional dictionary of Pytorch classes or Keras functions of any
        user-defined loss functions.  Each key is a loss function name, and the
        corresponding entry is the Python object implementing that loss.
    """
    # lots of exception handling here. TODO: Refactor.
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1:

        # get the weights for each loss within the composite
        if loss_weights is None:
            # weight all losses equally
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights

        # check if sublosses dict and weights dict have the same keys
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        if framework == 'keras':
            return keras_composite_loss(loss, weights, custom_losses)
        elif framework in ['pytorch', 'torch']:
            return TorchCompositeLoss(loss, weights, custom_losses)

    else:  # parse individual loss functions
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(framework, loss_name, loss_dict, custom_losses)


def get_single_loss(framework, loss_name, params_dict, custom_losses=None):
    if framework == 'keras':
        if loss_name.lower() == 'focal':
            return k_focal_loss(**params_dict)
        else:
            # keras_losses in the next line is a matching dict
            # TODO: the next block doesn't handle non-focal loss functions that
            # have hyperparameters associated with them. It would be great to
            # refactor this to handle that possibility.
            if custom_losses is not None and loss_name in custom_losses:
                return custom_losses.get(loss_name)
            else:
                return keras_losses.get(loss_name.lower())
    elif framework in ['torch', 'pytorch']:
        if params_dict is None:
            if custom_losses is not None and loss_name in custom_losses:
                return custom_losses.get(loss_name)()
            else:
                return torch_losses.get(loss_name.lower())()
        else:
            if custom_losses is not None and loss_name in custom_losses:
                return custom_losses.get(loss_name)(**params_dict)
            else:
                return torch_losses.get(loss_name.lower())(**params_dict)


def keras_composite_loss(loss_dict, weight_dict, custom_losses=None):
    """Wrapper to other loss functions to create keras-compatible composite."""

    def composite(y_true, y_pred):
        loss = K.sum(K.flatten(K.stack([weight_dict[loss_name]*get_single_loss(
                'keras', loss_name, loss_params, custom_losses)(y_true, y_pred)
                for loss_name, loss_params in loss_dict.items()], axis=-1)))
        return loss

    return composite


class TorchCompositeLoss(nn.Module):
    """Composite loss function."""

    def __init__(self, loss_dict, weight_dict=None, custom_losses=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss('pytorch',
                                                  loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]

        return loss
