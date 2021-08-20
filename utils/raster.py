import torch
import numpy as np
import tensorflow as tf


def reorder_axes(arr, target='tensorflow'):
    """Check order of axes in an array or tensor and convert to desired format.

    Arguments
    ---------
    arr : :class:`numpy.array` or :class:`torch.Tensor` or :class:`tensorflow.Tensor`
    target : str, optional
        Desired axis order type. Possible values:
        - ``'tensorflow'`` (default): ``[N, Y, X, C]`` or ``[Y, X, C]``
        - ``'torch'`` : ``[N, C, Y, X]`` or ``[C, Y, X]``

    Returns
    -------
    out_arr : an object of the same class as `arr` with axes in the desired
    order.
    """

    if isinstance(arr, torch.Tensor) or isinstance(arr, np.ndarray):
        axes = list(arr.shape)
    elif isinstance(arr, tf.Tensor):
        axes = arr.get_shape().as_list()

    if isinstance(arr, torch.Tensor):
        if len(axes) == 3:
            if target == 'tensorflow' and axes[0] < axes[1]:
                arr = arr.permute(1, 2, 0)
            elif target == 'torch' and axes[2] < axes[1]:
                arr = arr.permute(2, 0, 1)
        elif len(axes) == 4:
            if target == 'tensorflow' and axes[1] < axes[2]:
                arr = arr.permute(0, 2, 3, 1)
            elif target == 'torch' and axes[3] < axes[2]:
                arr = arr.permute(0, 3, 1, 2)

    elif isinstance(arr, np.ndarray):
        if len(axes) == 3:
            if target == 'tensorflow' and axes[0] < axes[1]:
                arr = np.moveaxis(arr, 0, -1)
            elif target == 'torch' and axes[2] < axes[1]:
                arr = np.moveaxis(arr, 2, 0)
        elif len(axes) == 4:
            if target == 'tensorflow' and axes[1] < axes[2]:
                arr = np.moveaxis(arr, 1, -1)
            elif target == 'torch' and axes[3] < axes[2]:
                arr = np.moveaxis(arr, 3, 1)

    elif isinstance(arr, tf.Tensor):
        # permutation is obnoxious in tensorflow; convert to numpy, permute,
        # convert back.
        np_version = arr.eval()
        np_version = reorder_axes(np_version, target=target)
        arr = tf.convert_to_tensor(np_version)

    return arr
