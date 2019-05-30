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


def k_weighted_bce(y_true, y_pred, weight):
    """Weighted binary cross-entropy for Keras.

    Arguments:
    ----------
    y_true : ``tf.Tensor``
        passed silently by Keras during model training.
    y_pred : ``tf.Tensor``
        passed silently by Keras during model training.
    weight : :class:`float` or :class:`int`
        Weight to assign to mask foreground pixels. Use values
        >1 to over-weight foreground or 0<value<1 to under-weight foreground.
        weight=1 is identical to vanilla binary cross-entropy.

    Returns:
    --------
    The binary cross-entropy loss function output multiplied by a weighting
    mask.

    Usage:
    ------
    Because Keras doesn't make it easy to implement loss functions that
    take arguments beyond `y_true` and `y_pred `, this function's arguments
    must be partially defined before passing it into your `model.compile`
    command. See example below, modified from ternausnet.py::

        model = Model(input=inputs, output=output_layer) # defined in ternausnet.py

        loss_func = partial(weighted_bce, weight=loss_weight)
        loss_func = update_wrapper(loss_func, weighted_bce)

        model.compile(optimizer=Adam(), loss=loss_func)

    If you wish to save and re-load a model which used this loss function,
    you must pass the loss function as a custom object::

        model.save('path_to_your_model.hdf5')
        wbce_loss = partial(weighted_bce, weight=loss_weight)
        wbce_loss = update_wrapper(wbce_loss, weighted_bce)
        reloaded_model = keras.models.load_model(
            'path_to_your_model.hdf5', custom_objects={'weighted_bce': wbce_loss}
            )

    """
    if weight == 1:  # identical to vanilla bce
        return K.binary_crossentropy(y_pred, y_true)
    weight_mask = K.ones_like(y_true)  # initialize weight mask
    class_two = K.equal(y_true, weight_mask)  # identify foreground pixels
    class_two = K.cast(class_two, 'float32')
    if weight < 1:
        class_two = class_two*(1-weight)
        final_mask = weight_mask - class_two  # foreground pixels weighted
    elif weight > 1:
        class_two = class_two*(weight-1)
        final_mask = weight_mask + class_two  # foreground pixels weighted
    return K.binary_crossentropy(y_pred, y_true) * final_mask


def k_layered_weighted_bce(y_true, y_pred, weights):
    """Binary cross-entropy function with different weights for mask channels.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.
    weights (list-like): Weights to assign to mask foreground pixels for each
        channel in the 3rd axis of the mask.

    Returns:
    --------
    The binary cross-entropy loss function output multiplied by a weighting
    mask.

    Usage:
    ------
    See implementation instructions for `weighted_bce`.

    This loss function is intended to allow different weighting of different
    segmentation outputs - for example, if a model outputs a 3D image mask,
    where the first channel corresponds to foreground objects and the second
    channel corresponds to object edges. `weights` must be a list of length
    equal to the depth of the output mask. The output mask's "z-axis"
    corresponding to the mask channel must be the third axis in the output
    array.

    """
    weight_mask = K.ones_like(y_true)
    submask_list = []
    for i in range(len(weights)):
        class_two = K.equal(y_true[:, :, :, i], weight_mask[:, :, :, i])
        class_two = K.cast(class_two, 'float32')
        if weights[i] < 1:
            class_two = class_two*(1-weights[i])
            layer_mask = weight_mask[:, :, :, i] - class_two
        elif weights[i] > 1:
            class_two = class_two*(weights[i]-1)
            layer_mask = weight_mask[:, :, :, i] + class_two
        else:
            layer_mask = weight_mask[:, :, :, i]
        submask_list.append(layer_mask)
    final_mask = K.stack(submask_list, axis=-1)
    return K.binary_crossentropy(y_pred, y_true) * final_mask
