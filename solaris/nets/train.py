"""Training code for `solaris` models."""

import numpy as np
import pandas as pd
from .model_io import get_model, reset_weights
from .datagen import make_data_generator
from .losses import get_loss
from .optimizers import get_optimizer
from .callbacks import get_callbacks
from .torch_callbacks import TorchEarlyStopping, TorchTerminateOnNaN
from .torch_callbacks import TorchModelCheckpoint
from .metrics import get_metrics
import torch
from torch.optim.lr_scheduler import _LRScheduler
import tensorflow as tf


class Trainer(object):
    """Object for training `solaris` models using PyTorch or Keras."""

    def __init__(self, config, custom_model_dict=None):
        self.config = config
        self.pretrained = self.config['pretrained']
        self.batch_size = self.config['batch_size']
        self.framework = self.config['nn_framework']
        self.model_name = self.config['model_name']
        self.model_path = self.config.get('model_path', None)
        self.model = get_model(self.model_name, self.framework,
                               self.model_path, self.pretrained,
                               custom_model_dict)
        self.train_df, self.val_df = get_train_val_dfs(self.config)
        self.train_datagen = make_data_generator(self.framework, self.config,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.framework, self.config,
                                               self.val_df, stage='train')
        self.epochs = self.config['training']['epochs']
        self.optimizer = get_optimizer(self.framework, self.config)
        self.lr = self.config['training']['lr']
        self.loss = get_loss(self.framework, self.config)
        self.callbacks = get_callbacks(self.framework, self.config)
        self.metrics = get_metrics(self.framework, self.config)
        self.verbose = self.config['training']['verbose']
        if self.framework in ['torch', 'pytorch']:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
            else:
                self.gpu_count = 0
        elif self.framework == 'keras':
            self.gpu_available = tf.test.is_gpu_available()

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
            if self.gpu_available:
                self.model = self.model.cuda()
                if self.gpu_count > 1:
                    self.model = torch.nn.DataParallel(self.model)
            # create optimizer
            if self.config['training']['opt_args'] is not None:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr,
                    **self.config['training']['opt_args']
                )
            else:
                self.optimizer = self.optimizer(
                    self.model.parameters(), lr=self.lr
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
#            tf_sess = tf.Session()
            for epoch in range(self.epochs):
                if self.verbose:
                    print('Beginning training epoch {}'.format(epoch))
                # TRAINING
                self.model.train()
                for batch_idx, batch in enumerate(self.train_datagen):
                    if self.config['data_specs'].get('additional_inputs',
                                                     None) is not None:
                        data = []
                        for i in ['image'] + self.config[
                                'data_specs']['additional_inputs']:
                            data.append(torch.Tensor(batch[i]).cuda())
                    else:
                        data = batch['image'].cuda()
                    target = batch['mask'].cuda().float()
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()

                    if self.verbose and batch_idx % 10 == 0:

                        print('    loss at batch {}: {}'.format(
                            batch_idx, loss), flush=True)
                        # calculate metrics
#                        for metric in self.metrics['train']:
#                            with tf_sess.as_default():
#                                print('{} score: {}'.format(
#                                    metric, metric(tf.convert_to_tensor(target.detach().cpu().numpy(), dtype='float64'), tf.convert_to_tensor(output.detach().cpu().numpy(), dtype='float64')).eval()))
                # VALIDATION
                with torch.no_grad():
                    self.model.eval()
                    torch.cuda.empty_cache()
                    val_loss = []
                    for batch_idx, batch in enumerate(self.val_datagen):
                        if self.config['data_specs'].get('additional_inputs',
                                                         None) is not None:
                            data = []
                            for i in ['image'] + self.config[
                                    'data_specs']['additional_inputs']:
                                data.append(torch.Tensor(batch[i]).cuda())
                        else:
                            data = batch['image'].cuda()
                        target = batch['mask'].cuda().float()
                        val_output = self.model(data)
                        val_loss.append(self.loss(val_output, target))
                    val_loss = torch.mean(torch.stack(val_loss))
                if self.verbose:
                    print()
                    print('    Validation loss at epoch {}: {}'.format(
                        epoch, val_loss))
                    print()
#                    for metric in self.metrics['val']:
#                        with tf_sess.as_default():
#                            print('validation {} score: {}'.format(
#                            metric, metric(tf.convert_to_tensor(target.detach().cpu().numpy(), dtype='float64'), tf.convert_to_tensor(output.detach().cpu().numpy(), dtype='float64')).eval()))
                check_continue = self._run_torch_callbacks(
                    loss.detach().cpu().numpy(),
                    val_loss.detach().cpu().numpy())
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
            if isinstance(self.model, nn.DataParallel):
                torch.save(self.model.module, self.config['training']['model_dest_path'])


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

    train_df = pd.read_csv(config['training_data_csv'])

    if config['data_specs']['val_holdout_frac'] is None:
        if config['validation_data_csv'] is None:
            raise ValueError(
                "If val_holdout_frac isn't specified in config,"
                " validation_data_csv must be.")
        val_df = pd.read_csv(config['validation_data_csv'])

    else:
        val_frac = config['data_specs']['val_holdout_frac']
        val_subset = np.random.choice(train_df.index,
                                      int(len(train_df)*val_frac),
                                      replace=False)
        val_df = train_df.loc[val_subset]
        # remove the validation samples from the training df
        train_df = train_df.drop(index=val_subset)

    return train_df, val_df
