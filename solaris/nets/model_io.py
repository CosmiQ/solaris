import os
from tensorflow import keras
import torch
from .zoo import model_dict


# below dictionary lists models compatible with solaris. alternatively, your
# own model can be used by using the path to the model as the value for
# model_name in the config file.


def get_model(model_name, framework, model_path=None, pretrained=False,
              custom_model_dict=None):
    """Load a model from a file based on its name."""

    md = model_dict.get(model_name, None)
    if md is None:  # if the model's not provided by solaris
        if custom_model_dict is None:
            raise ValueError(f"{model_name} can't be found in solaris and no "
                             "custom_model_dict was provided. Check your "
                             "model_name in the config file and/or provide a "
                             "custom_model_dict argument to Trainer().")
        else:
            md = custom_model_dict
    if model_path is None:
        model_path = md.get('weight_path')
    model = md.get('arch')()
    if model is not None and pretrained:
        try:
            model = _load_model_weights(model, model_path, framework)
        except (OSError, FileNotFoundError):
            pass  # TODO: IMPLEMENT MODEL DOWNLOAD FROM STORAGE HERE

    return model


def _load_model_weights(model, path, framework):
    """Backend for loading the model."""

    if framework.lower() == 'keras':
        try:
            model.load_weights(path)
        except OSError:
            raise FileNotFoundError("{} doesn't exist.".format(path))

    elif framework.lower() in ['torch', 'pytorch']:
        # pytorch already throws the right error on failed load, so no need
        # to fix exception
        model.load_state_dict(path)

    return model


def reset_weights(model, framework):
    """Re-initialize model weights for training.

    Arguments
    ---------
    model : :class:`tensorflow.keras.Model` or :class:`torch.nn.Module`
        A pre-trained, compiled model with weights saved.
    framework : str
        The deep learning framework used. Currently valid options are
        ``['torch', 'keras']`` .

    Returns
    -------
    reinit_model : model object
        The model with weights re-initialized. Note this model object will also
        lack an optimizer, loss function, etc., which will need to be added.
    """

    if framework == 'keras':
        model_json = model.to_json()
        reinit_model = keras.models.model_from_json(model_json)
    elif framework == 'torch':
        reinit_model = model.apply(_reset_torch_weights)

    return reinit_model


def _reset_torch_weights(torch_layer):
    if isinstance(torch_layer, torch.nn.Conv2d) or \
            isinstance(torch_layer, torch.nn.Linear):
        torch_layer.reset_parameters()
