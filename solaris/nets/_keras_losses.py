from tensorflow.keras import losses
from tensorflow.keras import backend as K
from .metrics import dice_coef_binary
import tensorflow as tf


def k_dice_loss(y_true, y_pred):
    return 1 - dice_coef_binary(y_true, y_pred)


def k_jaccard_loss(y_true, y_pred):
    """Jaccard distance for semantic segmentation.

    Modified from the `keras-contrib` package.

    """
    eps = 1e-12  # for stability
    y_pred = K.clip(y_pred, eps, 1-eps)
    intersection = K.sum(K.abs(y_true*y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = intersection/(sum_ - intersection)
    return 1 - jac


def k_focal_loss(gamma=2, alpha=0.75):
    # from github.com/atomwh/focalloss_keras

    def focal_loss_fixed(y_true, y_pred):  # with tensorflow

        eps = 1e-12  # improve the stability of the focal loss
        y_pred = K.clip(y_pred, eps, 1.-eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(
            alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum(
                (1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def k_lovasz_hinge(per_image=False):
    """Wrapper for the Lovasz Hinge Loss Function, for use in Keras.

    This is a mess. I'm sorry.
    """

    def lovasz_hinge_flat(y_true, y_pred):
        # modified from Maxim Berman's GitHub repo tf implementation for Lovasz
        eps = 1e-12  # for stability
        y_pred = K.clip(y_pred, eps, 1-eps)
        logits = K.log(y_pred/(1-y_pred))
        logits = tf.reshape(logits, (-1,))
        y_true = tf.reshape(y_true, (-1,))
        y_true = tf.cast(y_true, logits.dtype)
        signs = 2. * y_true - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0],
                                          name="descending_sort")
        gt_sorted = tf.gather(y_true, perm)
        grad = tf_lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted),
                            tf.stop_gradient(grad),
                            1, name="loss_non_void")
        return loss

    def lovasz_hinge_per_image(y_true, y_pred):
        # modified from Maxim Berman's GitHub repo tf implementation for Lovasz
        losses = tf.map_fn(_treat_image, (y_true, y_pred), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
        return loss

    def _treat_image(ytrue_ypred):
        y_true, y_pred = ytrue_ypred
        y_true, y_pred = tf.expand_dims(y_true, 0), tf.expand_dims(y_pred, 0)
        return lovasz_hinge_flat(y_true, y_pred)

    if per_image:
        return lovasz_hinge_per_image
    else:
        return lovasz_hinge_flat


def tf_lovasz_grad(gt_sorted):
    """
    Code from Maxim Berman's GitHub repo for Lovasz.

    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# matching dicts to get the right loss function based on the config file
keras_losses = {
    'binary_crossentropy': losses.binary_crossentropy,
    'bce': losses.binary_crossentropy,
    'categorical_crossentropy': losses.categorical_crossentropy,
    'cce': losses.categorical_crossentropy,
    'cosine': losses.cosine,
    'hinge': losses.hinge,
    'kullback_leibler_divergence': losses.kullback_leibler_divergence,
    'kld': losses.kullback_leibler_divergence,
    'mean_absolute_error': losses.mean_absolute_error,
    'mae': losses.mean_absolute_error,
    'mean_squared_logarithmic_error': losses.mean_squared_logarithmic_error,
    'msle': losses.mean_squared_logarithmic_error,
    'mean_squared_error': losses.mean_squared_error,
    'mse': losses.mean_squared_error,
    'sparse_categorical_crossentropy': losses.sparse_categorical_crossentropy,
    'squared_hinge': losses.squared_hinge,
    'jaccard': k_jaccard_loss,
    'dice': k_dice_loss
}
