import numpy as np
from tensorflow.keras import backend as K
from ._keras_losses import keras_losses, k_focal_loss
from ._torch_losses import torch_losses
import tensorflow as tf
import torch
from torch import nn


def get_loss(framework, config):
    """Load a loss function based on a config file for the specified framework.
    """
    # lots of exception handling here. TODO: Refactor.
    if not isinstance(config['training']['loss'], dict):
        raise TypeError('The loss description in the config file is formatted'
                        ' improperly. See the docs for details.')
    if len(config['training']['loss']) > 1:

        # get the weights for each loss within the composite
        if config['training'].get('loss_weights') is None:
            # weight all losses equally
            weights = {k: 1 for k in config['training']['loss'].keys()}
        else:
            weights = config['training']['loss_weights']

        # check if sublosses dict and weights dict have the same keys
        if list(config['training']['loss'].keys()).sort() !=    \
                list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')

        if framework == 'keras':
            return keras_composite_loss(config['training']['loss'], weights)
        elif framework in ['pytorch', 'torch']:
            return TorchCompositeLoss(config['training']['loss'], weights)

    else:  # parse individual loss functions
        loss_name, loss_dict = list(config['training']['loss'].items())[0]
        return get_single_loss(framework, loss_name, loss_dict)


def get_single_loss(framework, loss_name, loss_dict):
    if framework == 'keras':
        if loss_name.lower() == 'focal':
            return k_focal_loss(**loss_dict)
        else:
            # keras_losses in the next line is a matching dict
            # TODO: the next line doesn't handle non-focal loss functions that
            # have hyperparameters associated with them. It would be great to
            # refactor this to handle that possibility.
            return keras_losses.get(loss_name.lower(), None)
    elif framework in ['torch', 'pytorch']:
        return torch_losses.get(loss_name.lower(), None)(**loss_dict)


def keras_composite_loss(loss_dict, weight_dict):
    """Wrapper to other loss functions to create keras-compatible composite."""

    def composite(y_true, y_pred):
        loss = K.sum(K.flatten(K.stack([weight_dict[loss_name]*get_single_loss(
                'keras', loss_name, loss_params)(y_true, y_pred)
                for loss_name, loss_params in loss_dict.items()], axis=-1)))
        return loss

    return composite


class TorchCompositeLoss(nn.Module):
    """Composite loss function."""

    def __init__(self, loss_dict, weight_dict=None):
        """Create a composite loss function from a set of pytorch losses."""
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss('pytorch',
                                                  loss_name,
                                                  loss_params)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}  # values from the individual loss functions

    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]

        return loss


def k_weighted_bce(y_true, y_pred, weight):
    """Weighted binary cross-entropy for Keras.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.
    weight (numeric): Weight to assign to mask foreground pixels. Use values
        >1 to over-weight foreground or 0<value<1 to under-weight foreground.
        weight=1 is identical to vanilla binary cross-entropy.

    Returns:
    --------
    The binary cross-entropy loss function output multiplied by a weighting
    mask.

    Usage:
    ------
    Because Keras doesn't make it easy to implement loss functions that
    take arguments beyond `y_true` and `y_pred`, this function's arguments
    must be partially defined before passing it into your `model.compile`
    command. See example below, modified from ternausnet.py:

    ```
    model = Model(input=inputs, output=output_layer) # defined in ternausnet.py

    loss_func = partial(weighted_bce, weight=loss_weight)
    loss_func = update_wrapper(loss_func, weighted_bce)

    model.compile(optimizer=Adam(), loss=loss_func)
    ```

    If you wish to save and re-load a model which used this loss function,
    you must pass the loss function as a custom object:

    ```
    model.save('path_to_your_model.hdf5')
    wbce_loss = partial(weighted_bce, weight=loss_weight)
    wbce_loss = update_wrapper(wbce_loss, weighted_bce)
    reloaded_model = keras.models.load_model(
        'path_to_your_model.hdf5', custom_objects={'weighted_bce': wbce_loss}
        )
    ```

    """
    if weight == 1:  # identical to vanilla bce
        return K.binary_crossentropy(y_pred, y_true)
    weight_mask = K.ones_like(y_true)  # initialize weight mask
    class_two = K.equal(y_true, weight_mask)  # identify foreground pixels
    class_two = K.cast(class_two, 'float32')
    if weight < 1:
        class_two = class_two*(1-weight)
        final_mask = weight_mask - class_two  # foreground pixels weighted
    elif weight > 1:
        class_two = class_two*(weight-1)
        final_mask = weight_mask + class_two  # foreground pixels weighted
    return K.binary_crossentropy(y_pred, y_true) * final_mask


def k_layered_weighted_bce(y_true, y_pred, weights):
    """Binary cross-entropy function with different weights for mask channels.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.
    weights (list-like): Weights to assign to mask foreground pixels for each
        channel in the 3rd axis of the mask.

    Returns:
    --------
    The binary cross-entropy loss function output multiplied by a weighting
    mask.

    Usage:
    ------
    See implementation instructions for `weighted_bce`.

    This loss function is intended to allow different weighting of different
    segmentation outputs - for example, if a model outputs a 3D image mask,
    where the first channel corresponds to foreground objects and the second
    channel corresponds to object edges. `weights` must be a list of length
    equal to the depth of the output mask. The output mask's "z-axis"
    corresponding to the mask channel must be the third axis in the output
    array.

    """
    weight_mask = K.ones_like(y_true)
    submask_list = []
    for i in range(len(weights)):
        class_two = K.equal(y_true[:, :, :, i], weight_mask[:, :, :, i])
        class_two = K.cast(class_two, 'float32')
        if weights[i] < 1:
            class_two = class_two*(1-weights[i])
            layer_mask = weight_mask[:, :, :, i] - class_two
        elif weights[i] > 1:
            class_two = class_two*(weights[i]-1)
            layer_mask = weight_mask[:, :, :, i] + class_two
        else:
            layer_mask = weight_mask[:, :, :, i]
        submask_list.append(layer_mask)
    final_mask = K.stack(submask_list, axis=-1)
    return K.binary_crossentropy(y_pred, y_true) * final_mask
