"""Training code for `solaris` models."""

import numpy as np
from .model_io import get_model, reset_weights
from .datagen import make_data_generator
from .losses import get_loss
from .optimizers import get_optimizer
from .callbacks import get_callbacks
from .torch_callbacks import TorchEarlyStopping, TorchTerminateOnNaN
from .torch_callbacks import TorchModelCheckpoint
from .metrics import get_metrics
from ..utils.core import get_data_paths
import torch
from torch.optim.lr_scheduler import _LRScheduler


class Trainer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config):
        self.config = config
        self.pretrained = self.config['pretrained']
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.model_path = self.config['model_path']
        self.model = get_model(self.model_name, self.nn_framework,
                               self.model_path)
        self.train_df, self.val_df = get_train_val_dfs(self.config)
        self.train_datagen = make_data_generator(self.framework, self.config,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.framework, self.config,
                                               self.train_df, stage='train')
        self.epochs = self.config['training']['epochs']
        self.optimizer = get_optimizer(self.framework, self.config)
        self.lr = self.config['training']['lr']
        self.loss = get_loss(self.framework, self.config)
        self.callbacks = get_callbacks(self.framework, self.config)
        self.metrics = get_metrics(self.framework, self.config)
        self.verbose = self.config['training']['verbose']

        self.is_initialized = False
        self.stop = False

        self.initialize_model()

    def initialize_model(self):
        """Load in and create all model training elements."""
        if not self.pretrained:
            self.model = reset_weights(self.model, self.framework)

        if self.framework == 'keras':
            self.model = self.model.compile(optimizer=self.optimizer,
                                            loss=self.loss,
                                            metrics=self.metrics)

        elif self.framework == 'torch':
            # create optimizer
            self.optimizer = self.optimizer(
                self.model.parameters(), lr=self.lr,
                **self.config['training']['opt_args']
                )
            # wrap in lr_scheduler if one was created
            for cb in self.callbacks:
                if isinstance(cb, _LRScheduler):
                    self.optimizer = cb(
                        self.optimizer,
                        **self.config['training']['callbacks'][
                            'lr_schedule'].get(['schedule_dict'], {})
                        )
                    # drop the LRScheduler callback from the list
                    self.callbacks = [i for i in self.callbacks if i != cb]

        self.is_initialized = True

    def train(self):
        """Run training on the model."""
        if not self.is_initialized:
            self.initialize_model()

        if self.framework == 'keras':
            self.model.fit_generator(self.train_datagen,
                                     validation_data=self.val_datagen,
                                     epochs=self.epochs,
                                     callbacks=self.callbacks)

        elif self.framework == 'torch':
            for epoch in range(self.epochs):
                if self.verbose:
                    print('Beginning training epoch {}'.format(epoch))
                # TRAINING
                self.model.train()
                for batch_idx, (data, target) in enumerate(self.train_datagen):
                    data, target = data.cuda(), target.cuda()
                    self.optimizer.zero_grad()
                    output = self.model(data)

                    loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                    if self.verbose and batch_idx % 10 == 0:

                        print('    loss at batch {}: {}'.format(
                            batch_idx, np.round(loss, 3)))
                        # calculate metrics
                        for metric in self.metrics:
                            print('{} score: {}'.format(
                                metric, metric(target, output)))
                # VALIDATION
                self.model.eval()
                val_loss = []
                for batch_idx, (data,
                                target) in enumerate(self.val_datagen):
                    val_output = self.model(data)
                    val_loss.append(self.loss(val_output, target))
                val_loss = np.mean(val_loss)
                if self.verbose:
                    print()
                    print('    Validation loss at epoch {}: {}'.format(
                        epoch, val_loss))
                    print()
                check_continue = self._run_torch_callbacks(loss, val_loss)
                if not check_continue:
                    break

            self.save_model()

    def _run_torch_callbacks(self, loss, val_loss):
        for cb in self.callbacks:

            if isinstance(cb, TorchEarlyStopping):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchTerminateOnNaN):
                cb(val_loss)
                if cb.stop:
                    if self.verbose:
                        print('Early stopping triggered - '
                              'ending training')
                    return False

            elif isinstance(cb, TorchModelCheckpoint):
                if cb.monitor == 'loss':
                    cb(self.model, loss_value=loss)
                elif cb.monitor == 'val_loss':
                    cb(self.model, loss_value=val_loss)
                elif cb.monitor == 'periodic':
                    cb(self.model)

            return True

    def save_model(self):
        """Save the final model output."""
        if self.framework == 'keras':
            self.model.save(self.config['training']['model_dest_path'])
        elif self.framework == 'torch':
            torch.save(self.model, self.config['training']['model_dest_path'])


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
                "If val_holdout_frac isn't specified in config,"
                " validation_data_csv must be.")
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
