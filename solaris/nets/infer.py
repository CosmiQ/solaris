from .datagen import make_data_generator
from .model_io import get_model
from ..utils.core import get_data_paths


class Inferer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config):
        self.config = config
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        # check if the model was trained as part of the same pipeline; if so,
        # use the output from that. If not, use the pre-trained model directly.
        if self.config['train']:
            self.model_path = self.config['training']['model_dest_path']
        else:
            self.model_path = self.config['model_path']
        self.model = get_model(self.model_name, self.nn_framework,
                               self.model_path)
        self.infer_df = get_infer_df(self.config)
#        self.infer_datagen = make_data_generator(self.framework, self.config,
#                                                 self.infer_df, stage='infer')


def get_infer_df(config):
    """Get the inference df based on the contents of ``config``.

    This function uses the logic described in the documentation for the config
    file to determine where to find images to be used for inference.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.

    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.

    Returns
    -------
    infer_df : :class:`dict`
        :class:`dict` containing at least one column: ``'image'`` . The values
        in this column correspond to the path to filenames to perform inference
        on.
    """

    infer_df = get_data_paths(config['inference_data_csv'], infer=True)
    return infer_df
