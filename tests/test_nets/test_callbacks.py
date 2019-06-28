from solaris.nets.callbacks import KerasTerminateOnMetricNaN, get_callbacks
from solaris.nets.callbacks import get_lr_schedule
from solaris.nets.torch_callbacks import TorchEarlyStopping, \
    TorchModelCheckpoint, TorchTerminateOnNaN, TorchTerminateOnMetricNaN

import tensorflow as tf
import torch
import numpy as np


class TestGetCallbacksFunction(object):
    """test sol.nets.callbacks.get_callbacks()"""

    def test_keras_get_callbacks(self):
        framework = 'keras'
        config = {'training':
                  {'lr': 0.001,
                   'callbacks':
                   {'lr_schedule':
                    {'schedule_type': 'exponential',
                     'factor': 1,
                     'update_frequency': None
                     },
                    'terminate_on_nan': {},
                    'terminate_on_metric_nan': {},
                    'model_checkpoint':
                    {'filepath': 'sample_path.h5'},
                    'early_stopping': {},
                    'csv_logger':
                    {'filename': 'sample_path.csv'},
                    'reduce_lr_on_plateau': {}
                    }
                   }
                  }
        result = get_callbacks(framework, config)
        assert len(result) == 7
        has_lr_sched = False
        has_term_nan = False
        has_term_met_nan = False
        has_mod_ckpt = False
        has_early_stopping = False
        has_csv_logger = False
        has_red_lr_plat = False
        for callback in result:
            if isinstance(callback, tf.keras.callbacks.LearningRateScheduler):
                has_lr_sched = True
            elif isinstance(callback, tf.keras.callbacks.TerminateOnNaN):
                has_term_nan = True
            elif isinstance(callback, KerasTerminateOnMetricNaN):
                has_term_met_nan = True
            elif isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                has_mod_ckpt = True
            elif isinstance(callback, tf.keras.callbacks.EarlyStopping):
                has_early_stopping = True
            elif isinstance(callback, tf.keras.callbacks.CSVLogger):
                has_csv_logger = True
            elif isinstance(callback, tf.keras.callbacks.ReduceLROnPlateau):
                has_red_lr_plat = True
        assert has_lr_sched
        assert has_term_nan
        assert has_term_met_nan
        assert has_mod_ckpt
        assert has_early_stopping
        assert has_csv_logger
        assert has_red_lr_plat

    def test_get_torch_callbacks(self):
        framework = 'torch'
        config = {'training':
                  {'lr': 0.001,
                   'callbacks':
                   {'lr_schedule':
                    {'schedule_type': 'exponential',
                     'factor': 1,
                     'update_frequency': None
                     },
                    'terminate_on_nan': {},
                    'terminate_on_metric_nan': {
                        'stopping_metric': 'accuracy'
                        },
                    'model_checkpoint': {
                        'filepath': 'sample_path.h5'
                        },
                    'early_stopping': {}
                    }
                   }
                  }
        result = get_callbacks(framework, config)
        assert len(result) == 5
        has_lr_sched = False
        has_term_nan = False
        has_term_met_nan = False
        has_mod_ckpt = False
        has_early_stopping = False

        for callback in result:
            if callback == torch.optim.lr_scheduler.ExponentialLR:
                has_lr_sched = True
            elif isinstance(callback, TorchTerminateOnNaN):
                has_term_nan = True
            elif isinstance(callback, TorchTerminateOnMetricNaN):
                has_term_met_nan = True
            elif isinstance(callback, TorchModelCheckpoint):
                has_mod_ckpt = True
            elif isinstance(callback, TorchEarlyStopping):
                has_early_stopping = True

        assert has_lr_sched
        assert has_term_nan
        assert has_term_met_nan
        assert has_mod_ckpt
        assert has_early_stopping


class TestLRSchedulers(object):
    """Test LR scheduling from get_lr_scheduler()."""

    def test_keras_schedulers(self):
        epsilon = 1e-9
        framework = 'keras'
        config = {'training':
                  {'lr': 0.001,
                   'callbacks': {}
                   }
                  }
        schedule_dicts = [
             {'schedule_type': 'exponential',
              'factor': 0.5,
              'update_frequency': 1
              },
             {'schedule_type': 'arbitrary',
              'schedule_dict': {
                   10: 0.0001,
                   20: 0.00001
               }
              },
             {'schedule_type': 'linear',
              'factor': -.01
              }
        ]

        for schedule_dict in schedule_dicts:
            config['training']['callbacks']['lr_schedule'] = schedule_dict
            lr_scheduler = get_lr_schedule(framework, config)
            # test lr schedule function outputs to make sure they're right
            if schedule_dict['schedule_type'] == 'exponential':
                assert np.abs(lr_scheduler.schedule(0) - 0.001) < epsilon
                assert np.abs(lr_scheduler.schedule(1) - 0.0005) < epsilon
                assert np.abs(lr_scheduler.schedule(2) - 0.00025) < epsilon
            elif schedule_dict['schedule_type'] == 'linear':
                assert np.abs(lr_scheduler.schedule(0) - 0.001) < epsilon
                assert np.abs(lr_scheduler.schedule(1) - 0.00099) < epsilon
                assert np.abs(lr_scheduler.schedule(10) - 0.0009) < epsilon
            elif schedule_dict['schedule_type'] == 'arbitrary':
                assert np.abs(lr_scheduler.schedule(0) - 0.001) < epsilon
                assert np.abs(lr_scheduler.schedule(10) - 0.0001) < epsilon
                assert np.abs(lr_scheduler.schedule(20) - 0.00001) < epsilon

    def test_torch_schedulers(self):
        framework = 'torch'
        config = {'training':
                  {'lr': 0.001,
                   'callbacks': {}
                   }
                  }
        schedule_dicts = [
             {'schedule_type': 'exponential',
              'factor': 0.5,
              'update_frequency': 1
              },
             {'schedule_type': 'arbitrary',
              'schedule_dict': {
                   10: 0.0001,
                   20: 0.00001
               }
              },
             {'schedule_type': 'linear',
              'factor': -.01
              }
        ]
        for schedule_dict in schedule_dicts:
            config['training']['callbacks']['lr_schedule'] = schedule_dict
            lr_scheduler = get_lr_schedule(framework, config)
            if schedule_dict['schedule_type'] == 'exponential':
                assert lr_scheduler == torch.optim.lr_scheduler.ExponentialLR
            elif schedule_dict['schedule_type'] == 'linear':
                assert lr_scheduler == torch.optim.lr_scheduler.StepLR
            elif schedule_dict['schedule_type'] == 'arbitrary':
                assert lr_scheduler == torch.optim.lr_scheduler.MultiStepLR
