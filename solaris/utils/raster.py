import numpy as np


def reorder_axes(arr, target="tensorflow"):
    """Check order of axes in an array or tensor and convert to desired format.

    Arguments
    ---------
    arr : :class:`numpy.array`
    target : str, optional
        Desired axis order type. Possible values:
        - ``'tensorflow'`` (default): ``[N, Y, X, C]`` or ``[Y, X, C]``
        - ``'torch'`` : ``[N, C, Y, X]`` or ``[C, Y, X]``

    Returns
    -------
    out_arr : an object of the same class as `arr` with axes in the desired
    order.
    """

    axes = list(arr.shape)
    if len(axes) == 3:
        if target == "tensorflow" and axes[0] < axes[1]:
            arr = np.moveaxis(arr, 0, -1)
        elif target == "torch" and axes[2] < axes[1]:
            arr = np.moveaxis(arr, 2, 0)
    elif len(axes) == 4:
        if target == "tensorflow" and axes[1] < axes[2]:
            arr = np.moveaxis(arr, 1, -1)
        elif target == "torch" and axes[3] < axes[2]:
            arr = np.moveaxis(arr, 3, 1)
    return arr
