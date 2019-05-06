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
