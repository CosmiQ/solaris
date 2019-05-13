import os
from tensorflow import keras
import torch


# below dictionary lists models compatible with solaris. alternatively, your
# own model can be used by using the path to the model as the value for
# model_name in the config file.

model_dict = {'placeholder': 'model_path.hdf5'}


def get_model(model_name, framework, model_path=None):
    """Load a model from a file based on its name."""
    if model_path is None:
        model_path = model_dict.get(model_name, None)
    try:
        model = _load_model(model_path, framework)
    except (OSError, FileNotFoundError):
        pass  # TODO: IMPLEMENT MODEL DOWNLOAD FROM STORAGE HERE

    return model


def _load_model(path, framework):
    """Backend for loading the model."""

    if framework.lower() == 'keras':
        try:
            model = keras.models.load_model(path)
        except OSError:
            raise FileNotFoundError("{} doesn't exist.".format(path))

    elif framework.lower() == 'pytorch':
        # pytorch already throws the right error on failed load, so no need
        # to fix exception
        model = torch.load(path)

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
