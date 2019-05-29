import numpy as np
import tensorflow as tf
from tensorflow import keras
from solaris.nets.metrics import get_metrics, precision, recall, f1_score
from solaris.nets.metrics import metric_dict


class TestMetricLoads(object):
    """Test that metrics load correctly and produce the expected value."""

    def test_binary_metrics(self):
        self.epsilon = 1e-6
        self.truth = tf.convert_to_tensor(
            np.array(
                [0., 1., 1., 1., 0., 1., 0., 1., 1.]
                ).reshape((3, 3)))
        self.pred = tf.convert_to_tensor(
            np.array(
                [0.1, 0.9, 1, 0.3, 0.8, 1., 0.7, 0.2, 0.8]
                ).reshape((3, 3)))
        self.metrics_and_exp_scores = {
            'accuracy': np.array([1., 0.33333334, 0.33333334]),
            'binary_accuracy': np.array([1., 0.33333334, 0.33333334]),
            'precision': 0.6666666555555557,
            'recall': 0.6666666666666666,
            'f1_score': 0.6666666611111112,
            'cosine': np.array([-0.99587059, -0.69888433, -0.65372045]),
            'cosine_proximity': np.array([-0.99587059, -0.69888433,
                                          -0.65372045]),
            'hinge': np.array([0.36666667, 0.56666667, 0.66666667]),
            'squared_hinge': np.array([0.33666667, 0.49666667, 0.56]),
            'kld': np.array([0.10535913, 1.20397121, 1.83257989]),
            'kullback_leibler_divergence': np.array([0.10535913, 1.20397121,
                                                     1.83257989]),
            'mae': np.array([0.06666667, 0.5, 0.56666667]),
            'mean_absolute_error': np.array([0.06666667, 0.5, 0.56666667]),
            'mse': np.array([0.00666667, 0.37666667, 0.39]),
            'mean_squared_error': np.array([0.00666667, 0.37666667, 0.39]),
            'msle': np.array([0.003905, 0.17702232, 0.18453663]),
            'mean_squared_logarithmic_error': np.array([0.003905, 0.17702232,
                                                        0.18453663])
        }
        sess = tf.Session()
        with sess.as_default():
            for metric, expected_result in self.metrics_and_exp_scores.items():
                assert np.abs(np.sum(
                    metric_dict[metric](
                        self.truth, self.pred).eval() - expected_result)
                              ) < self.epsilon


class TestGetMetrics(object):
    """Test the get_metrics() function in solaris.nets.metrics."""

    def test_get_metrics(self):
        self.config = {'training':
                       {'metrics':
                        {'training': ['precision', 'recall', 'f1_score',
                                      'accuracy', 'categorical_accuracy',
                                      'cosine', 'hinge', 'squared_hinge',
                                      'kld', 'mae', 'mse', 'msle',
                                      'sparse_categorical_accuracy',
                                      'top_k_categorical_accuracy'],
                         'validation': ['precision']
                         }
                        }
                       }
        self.expected_dict = {
            'train': [
                precision, recall, f1_score, keras.metrics.binary_accuracy,
                keras.metrics.categorical_accuracy,
                keras.metrics.cosine_proximity,
                keras.metrics.hinge, keras.metrics.squared_hinge,
                keras.metrics.kullback_leibler_divergence,
                keras.metrics.mean_absolute_error,
                keras.metrics.mean_squared_error,
                keras.metrics.mean_squared_logarithmic_error,
                keras.metrics.sparse_categorical_accuracy,
                keras.metrics.top_k_categorical_accuracy
                ],
            'val': [precision]
            }
        assert get_metrics('keras', self.config) == self.expected_dict
