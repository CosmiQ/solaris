"""Training code for `solaris` models."""

import numpy as np
from .model_io import get_model
from .datagen import make_data_generator
from .losses import get_loss
from .callbacks import get_callbacks
from .metrics import get_metrics
from ..utils.core import get_data_paths


class Trainer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config):
        self.config = config
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.model_path = self.config['model_path']
        self.model = get_model(self.model_name, self.nn_framework,
                               self.model_path)
        self.train_df, self.val_df = get_train_val_dfs(config)
        self.train_datagen = make_data_generator(self.framework, self.config,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.framework, self.config,
                                               self.train_df, stage='train')
        self.epochs = self.config['training']['epochs']
        self.optimizer = get_optimizer(self.framework, self.config)
        self.loss = get_loss(self.framework, self.config)
        self.callbacks = get_callbacks(self.framework, self.config)
        self.metrics = get_metrics(self.framework, self.config)

    def train(self):
        """Run training on the model."""
        pass  # TODO: IMPLEMENT

    def save(self):
        """Save the final model output."""
        pass  # TODO: IMPLEMENT


def get_train_val_dfs(config):
    """Get the training and validation dfs based on the contents of ``config``.

    This function uses the logic described in the documentation for the config
    files to determine where to find training and validation dataset files.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.

    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.

    Returns
    -------
    train_df, val_df : :class:`tuple` of :class:`dict` s
        :class:`dict` s containing two columns: ``'image'`` and ``'label'``.
        Each column corresponds to paths to find matching image and label files
        for training.
    """

    train_df = get_data_paths(config['training_data_csv'])

    if config['data_specs']['val_holdout_frac'] is None:
        if config['validation_data_csv'] is None:
            raise ValueError(
                "If val_holdout_frac isn't specified in config, validation_data_csv must be.")
        val_df = get_data_paths(config['validation_data_csv'])

    else:
        val_frac = config['data_specs']['val_holdout_frac']
        val_subset = np.random.choice(train_df.index,
                                      int(len(train_df)*val_frac),
                                      replace=False)
        val_df = train_df.loc[val_subset]
        # remove the validation samples from the training df
        train_df = train_df.drop(index=val_subset)

    return train_df, val_df


def get_optimizer(framework, config):
    """Load in the framework-specific optimizer for the model."""
    # TODO: IMPLEMENT
    pass
