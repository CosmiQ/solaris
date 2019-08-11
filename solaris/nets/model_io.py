import os
from tensorflow import keras
import torch
from warnings import warn
import requests
import numpy as np
from tqdm import tqdm
from ..nets import weights_dir
from .zoo import model_dict


def get_model(model_name, framework, model_path=None, pretrained=False,
              custom_model_dict=None):
    """Load a model from a file based on its name."""
    if custom_model_dict is not None:
        md = custom_model_dict
    else:
        md = model_dict.get(model_name, None)
        if md is None:  # if the model's not provided by solaris
            raise ValueError(f"{model_name} can't be found in solaris and no "
                             "custom_model_dict was provided. Check your "
                             "model_name in the config file and/or provide a "
                             "custom_model_dict argument to Trainer().")
    if model_path is None or custom_model_dict is not None:
        model_path = md.get('weight_path')
    model = md.get('arch')()
    if model is not None and pretrained:
        try:
            model = _load_model_weights(model, model_path, framework)
        except (OSError, FileNotFoundError):
            warn(f'The model weights file {model_path} was not found.'
                 ' Attempting to download from the SpaceNet repository.')
            weight_path = _download_weights(md)
            model = _load_model_weights(model, weight_path, framework)

    return model


def _load_model_weights(model, path, framework):
    """Backend for loading the model."""

    if framework.lower() == 'keras':
        try:
            model.load_weights(path)
        except OSError:
            # first, check to see if the weights are in the default sol dir
            default_path = os.path.join(weights_dir, os.path.split(path)[1])
            try:
                model.load_weights(default_path)
            except OSError:
                # if they can't be found anywhere, raise the error.
                raise FileNotFoundError("{} doesn't exist.".format(path))

    elif framework.lower() in ['torch', 'pytorch']:
        # pytorch already throws the right error on failed load, so no need
        # to fix exception
        if torch.cuda.is_available():
            try:
                loaded = torch.load(path)
            except FileNotFoundError:
                # first, check to see if the weights are in the default sol dir
                default_path = os.path.join(weights_dir,
                                            os.path.split(path)[1])
                loaded = torch.load(path)
        else:
            try:
                loaded = torch.load(path, map_location='cpu')
            except FileNotFoundError:
                default_path = os.path.join(weights_dir,
                                            os.path.split(path)[1])
                loaded = torch.load(path, map_location='cpu')

        if isinstance(loaded, torch.nn.Module):  # if it's a full model already
            model.load_state_dict(loaded.state_dict())
        else:
            model.load_state_dict(loaded)

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


def _download_weights(model_dict):
    """Download pretrained weights for a model."""
    weight_url = model_dict.get('weight_url', None)
    weight_dest_path = model_dict.get('weight_path', os.path.join(
            weights_dir, weight_url.split('/')[-1]))
    if weight_url is None:
        raise KeyError("Can't find the weights file.")
    else:
        r = requests.get(weight_url, stream=True)
        if r.status_code != 200:
            raise ValueError('The file could not be downloaded. Check the URL'
                             ' and network connections.')
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        with open(weight_dest_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(block_size),
                              total=np.ceil(total_size//block_size),
                              unit='KB', unit_scale=False):
                if chunk:
                    f.write(chunk)

    return weight_dest_path
