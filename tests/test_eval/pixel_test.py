"""Tests for ``solaris.eval.pixel_metrics`` functions."""

import numpy as np
from solaris.eval.pixel import iou, f1, relaxed_f1


class TestIoU(object):
    """Test ``sol.eval.pixel.iou()``."""

    def test_iou_basic(self):
        truth = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 1, 1]])
        prop = np.array([[0, 0, 0],
                         [1, 1, 0],
                         [1, 0, 1]])
        assert iou(truth, prop) == 0.5

    def test_iou_pvals(self):
        truth = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 1, 1]])
        prop = np.array([[0, 0.1, 0.4],
                         [0.8, 0.7, 0.5],
                         [1, 0, 1]])
        assert iou(truth, prop, prop_threshold=0.55) == 0.5


class TestF1(object):
    """Test ``sol.eval.pixel.f1()``."""

    def test_f1_basic(self):
        eps = 1e-7
        truth = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 1, 1]])
        prop = np.array([[0, 0, 0],
                         [1, 1, 0],
                         [1, 0, 1]])
        f1_score, precision, recall = f1(truth, prop)
        assert (precision - 0.75) < eps
        assert (recall - 0.6) < eps
        assert (f1_score - 2./3) < eps

    def test_f1_pvals(self):
        eps = 1e-7
        truth = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 1, 1]])
        prop = np.array([[0, 0.1, 0.4],
                         [0.8, 0.7, 0.5],
                         [1, 0, 1]])
        f1_score, precision, recall = f1(truth, prop, prop_threshold=0.55)
        assert (precision - 0.75) < eps
        assert (recall - 0.6) < eps
        assert (f1_score - 2./3) < eps

    def test_f1_no_pos_preds(self):
        truth = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 1, 1]])
        prop = np.zeros(shape=(3, 3))
        assert f1(truth, prop) == (0, 0, 0)

    def test_f1_no_pos_truth(self):
        prop = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 1, 1]])
        truth = np.zeros(shape=(3, 3))
        assert f1(truth, prop) == (0, 0, 0)


class TestRelaxedF1(object):
    """Test ``sol.eval.pixel.relaxed_f1()``."""

    def test_relaxed_f1_basic(self):
        eps = 1e-7
        truth_mask = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
            )
        prop_mask = np.array(
            [[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [1., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]]
            )

        rel_f1, rel_prec, rel_rec = relaxed_f1(truth_mask,
                                               prop_mask,
                                               radius=3)
        assert (rel_f1 - 0.8571428571428571) < eps
        assert rel_prec - 0.75 < eps
        assert rel_rec - 1.0 < eps
