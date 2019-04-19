from keras import backend as K

# TODO: IMPLEMENT WRAPPER FOR THESE FOR DIFFERENT ML FRAMEWORKS

def weighted_bce(y_true, y_pred, weight):
    """Weighted binary cross-entropy for Keras.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.
    weight (numeric): Weight to assign to mask foreground pixels. Use values
        >1 to over-weight foreground or 0<value<1 to under-weight foreground.
        weight=1 is identical to vanilla binary cross-entropy.

    Returns:
    --------
    The binary cross-entropy loss function output multiplied by a weighting
    mask.

    Usage:
    ------
    Because Keras doesn't make it easy to implement loss functions that
    take arguments beyond `y_true` and `y_pred`, this function's arguments
    must be partially defined before passing it into your `model.compile`
    command. See example below, modified from ternausnet.py:

    ```
    model = Model(input=inputs, output=output_layer) # defined in ternausnet.py

    loss_func = partial(weighted_bce, weight=loss_weight)
    loss_func = update_wrapper(loss_func, weighted_bce)

    model.compile(optimizer=Adam(), loss=loss_func)
    ```

    If you wish to save and re-load a model which used this loss function,
    you must pass the loss function as a custom object:

    ```
    model.save('path_to_your_model.hdf5')
    wbce_loss = partial(weighted_bce, weight=loss_weight)
    wbce_loss = update_wrapper(wbce_loss, weighted_bce)
    reloaded_model = keras.models.load_model(
        'path_to_your_model.hdf5', custom_objects={'weighted_bce': wbce_loss}
        )
    ```

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


def layered_weighted_bce(y_true, y_pred, weights):
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


def jaccard_loss(y_true, y_pred):
    """Jaccard (IoU) loss function for use with Keras.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.

    Returns:
    --------
    The Jaccard (IoU) loss

    The Jaccard loss, or IoU loss, is defined as:

    1 - intersection(true, pred)/union(true, pred).

    This loss function can be very useful in semantic segmentation problems
    with imbalanced classes.
    """

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jaccard = intersection / (sum_ - intersection)

    return 1-jaccard


def hybrid_bce_jaccard(y_true, y_pred, jac_frac=0.25):
    """Hybrid binary cross-entropy and Jaccard loss function.

    Arguments:
    ----------
    y_true (tensor): passed silently by Keras during model training.
    y_pred (tensor): passed silently by Keras during model training.
    jac_frac (float, range [0, 1]): Fraction of the loss comprised by Jaccard.
        binary cross-entropy will make up the remainder.

    Returns:
    --------
    The hybrid BCE-Jaccard (IoU) loss.

    As with the pure Jaccard loss, this loss function is often used in
    optimization of semantic segmentation problems with imbalanced classes
    where BCE has a strong propensity to fall into a valley of predicting all
    one class. See https://arxiv.org/abs/1806.05182 and others for similar
    approaches.
    """
    jac_loss = jaccard_loss(y_true, y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)

    return jac_frac*jac_loss + (1-jac_frac)*bce
