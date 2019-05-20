from solaris.nets.losses import get_loss
from solaris.nets._keras_losses import k_jaccard_loss, k_focal_loss
from solaris.nets._keras_losses import k_lovasz_hinge
from solaris.nets._torch_losses import TorchFocalLoss, torch_lovasz_hinge
from tensorflow import keras
import torch
import numpy as np
import tensorflow as tf


class TestGetLoss(object):
    """Test solaris.nets.losses.get_loss()."""

    def test_keras_vanilla_loss(self):
        config = {'training':
                  {'loss':
                   {'bce': {}}
                   }
                  }
        lf = get_loss('keras', config)
        assert lf == keras.losses.binary_crossentropy

    def test_keras_composite_loss_noweight(self):
        epsilon = 1e-6
        config = {'training':
                  {'loss':
                   {'bce': {},
                    'hinge': {}}
                   }
                  }
        lf = get_loss('keras', config)
        y_true = tf.constant([1, 1, 1], dtype='float')
        y_pred = tf.constant([0, 1, 0], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            assert np.abs(
                lf(y_true, y_pred).eval() - 11.41206380063888) < epsilon

    def test_torch_vanilla_loss(self):
        config = {'training':
                  {'loss':
                   {'bce': {}}
                   }
                  }
        lf = get_loss('torch', config)
        assert isinstance(lf, torch.nn.BCELoss)

    def test_torch_composite_loss(self):
        epsilon = 1e-4
        config = {'training':
                  {'loss':
                   {'bce': {},
                    'hinge': {}}
                   }
                  }
        lf = get_loss('torch', config)
        y_true = torch.tensor([1, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([0, 1, 0], dtype=torch.float)
        assert np.abs(
            lf.forward(y_true, y_pred) - 19.4207) < epsilon


class TestKerasCustomLosses(object):

    def test_keras_focal_loss(self):
        epsilon = 1e-6
        y_true = tf.constant([1, 1, 1], dtype='float')
        y_pred = tf.constant([0, 1, 0], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            foc_loss = k_focal_loss()(y_true, y_pred).eval()
            assert np.abs(foc_loss - 41.446533) < epsilon

    def test_keras_lovasz_hinge(self):
        epsilon = 1e-6
        y_true = tf.constant([1, 1, 1], dtype='float')
        y_pred = tf.constant([0, 1, 0], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            lov_loss = k_lovasz_hinge()(y_true, y_pred).eval()
            assert np.abs(lov_loss - 19.087347) < epsilon

    def test_keras_jaccard_loss(self):
        epsilon = 1e-6
        y_true = tf.constant([1, 1, 1], dtype='float')
        y_pred = tf.constant([0, 1, 0], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            jac_loss = k_jaccard_loss(y_true, y_pred).eval()
            assert np.abs(jac_loss - 0.66666666667) < epsilon


class TestTorchCustomLosses(object):

    def test_torch_focal_loss(self):
        epsilon = 1e-6
        y_true = torch.tensor([1, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([0, 1, 0], dtype=torch.float)
        lf = TorchFocalLoss()
        assert np.abs(
            lf.forward(y_pred, y_true) - 0.2769237458705902) < epsilon

    def test_torch_lovasz_hinge(self):
        epsilon = 1e-6
        y_true = torch.tensor([1, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([0, 1, 0], dtype=torch.float)
        lf = torch_lovasz_hinge
        assert np.abs(
            lf(y_pred, y_true) - 0.6666666269302368) < epsilon
