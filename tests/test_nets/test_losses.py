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
        loss_dict = {'bce' : {}}
        lf = get_loss('keras', loss_dict)
        assert lf == keras.losses.binary_crossentropy

    def test_keras_composite_loss_noweight(self):
        epsilon = 1e-6
        loss_dict = {'bce' : {}, 'hinge' : {}}
        lf = get_loss('keras', loss_dict)
        y_true = tf.constant([0, 1, 1], dtype='float')
        y_pred = tf.constant([.1, .9, .4], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            assert np.abs(
                lf(y_true, y_pred).eval() - 0.9423373) < epsilon

    def test_torch_vanilla_loss(self):
        loss_dict = {'bce' : {}}
        lf = get_loss('torch', loss_dict)
        assert isinstance(lf, torch.nn.BCELoss)

    def test_torch_composite_loss(self):
        epsilon = 1e-4
        loss_dict = {'bce' : {}, 'hinge' : {}}
        lf = get_loss('torch', loss_dict)
        y_true = torch.tensor([0, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([.1, .9, .4], dtype=torch.float)
        assert np.abs(
            lf.forward(y_pred, y_true) - 1.1423372030) < epsilon


class TestKerasCustomLosses(object):

    def test_keras_focal_loss(self):
        epsilon = 1e-6
        y_true = tf.constant([0, 1, 1], dtype='float')
        y_pred = tf.constant([.1, .9, .4], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            foc_loss = k_focal_loss()(y_true, y_pred).eval()
            assert np.abs(foc_loss - 0.24845211) < epsilon

    def test_keras_lovasz_hinge(self):
        epsilon = 1e-6
        y_true = tf.constant([0, 1, 1], dtype='float')
        y_pred = tf.constant([.1, .9, .4], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            lov_loss = k_lovasz_hinge()(y_true, y_pred).eval()
            assert np.abs(lov_loss - 0.70273256) < epsilon

    def test_keras_jaccard_loss(self):
        epsilon = 1e-6
        y_true = tf.constant([0, 1, 1], dtype='float')
        y_pred = tf.constant([.1, .9, .4], dtype='float')
        sess = tf.Session()
        with sess.as_default():
            jac_loss = k_jaccard_loss(y_true, y_pred).eval()
            assert np.abs(jac_loss - 0.38095242) < epsilon


class TestTorchCustomLosses(object):

    def test_torch_focal_loss(self):
        epsilon = 1e-6
        y_true = torch.tensor([0, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([.1, .9, .4], dtype=torch.float)
        lf = TorchFocalLoss()
        assert np.abs(
            lf.forward(y_pred, y_true) - 0.1106572822) < epsilon

    def test_torch_focal_loss_same(self):
        epsilon = 1e-5
        y_true = torch.tensor([0, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([0, 1, 1], dtype=torch.float)
        lf = TorchFocalLoss()
        assert np.abs(
            lf.forward(y_pred, y_true) - 0.) < epsilon

    def test_torch_lovasz_hinge(self):
        epsilon = 1e-6
        y_true = torch.tensor([0, 1, 1], dtype=torch.float)
        y_pred = torch.tensor([.1, .9, .4], dtype=torch.float)
        lf = torch_lovasz_hinge
        assert np.abs(
            lf(y_pred, y_true) - 0.6000000000) < epsilon
