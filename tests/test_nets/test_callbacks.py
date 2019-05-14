import numpy as np
from solaris.nets.callbacks import KerasTerminateOnMetricNaN, get_callbacks
import tensorflow as tf
import torch


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
            if callback == torch.optim.lr_scheduler.ExponentialLR:
                has_lr_sched = True
            elif callback == 'terminate_on_nan':
                has_term_nan = True
            elif callback == 'terminate_on_metric_nan':
                has_term_met_nan = True
            elif callback == 'model_checkpoint':
                has_mod_ckpt = True
            elif callback == 'early_stopping':
                has_early_stopping = True
            elif callback == 'csv_logger':
                has_csv_logger = True
            elif callback == 'reduce_lr_on_plateau':
                has_red_lr_plat = True
        assert has_lr_sched
        assert has_term_nan
        assert has_term_met_nan
        assert has_mod_ckpt
        assert has_early_stopping
        assert has_csv_logger
        assert has_red_lr_plat


class TestTorchCallbacks(object):
    pass


class TestLRSchedulers(object):
    pass
