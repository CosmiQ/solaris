"""Utility functions for data io."""
import numpy as np
import skimage


def imread(path):
    """Read in an image file.

    Note
    ----
    Because overhead imagery is often either 16-bit or multispectral (i.e. >3
    channels or bands that don't directly translate into the RGB scheme of
    photographs), this package using scikit-image_ ``io`` algorithms. Though
    slightly slower, these algorithms are compatible with any bit depth or
    channel count.

    .. _scikit-image: https://scikit-image.org

    Arguments
    ---------
    path : str
        Path to the image file to load.

    Returns
    -------
    im : :func:`numpy.array`
        A NumPy array of shape ``[Y, X, C]`` containing the imagery, with dtype
        ``uint8``.

    """
    im_arr = skimage.io.imread(path)
    # check dtype for preprocessing
    if im_arr.dtype == np.uint8:
        dtype = 'uint8'
    elif im_arr.dtype == np.uint16:
        dtype = 'uint16'
    elif im_arr.dtype in [np.float16, np.float32, np.float64]:
        if np.amax(im_arr) <= 1 and np.amin(im_arr) >= 0:
            dtype = 'zero-one normalized'  # range = 0-1
        elif np.amax(im_arr) > 0 and np.amin(im_arr) < 0:
            dtype = 'z-scored'
        elif np.amax(im_arr) <= 255:
            dtype = '255 float'
        elif np.amax(im_arr) <= 65535:
            dtype = '65535 float'
        else:
            raise TypeError('The loaded image array is an unexpected dtype.')
    else:
        raise TypeError('The loaded image array is an unexpected dtype.')

    im_arr = preprocess_im_arr(im_arr, dtype)
    return im_arr


def preprocess_im_arr(im_arr, im_format):
    """Convert image to standard shape and dtype for use in the pipeline.

    Notes
    -----
    This repo will require the following of images:

       - Their shape is of form [X, Y, C]
       - Input images are dtype ``uint8``

    This function will take an image array `im_arr` and reshape it accordingly.

    Arguments
    ---------
    im_arr : :func:`numpy.array`
        A numpy array representation of an image. `im_arr` should have either
        two or three dimensions.
    im_format : str
        One of ``'uint8'``, ``'uint16'``, ``'z-scored'``,
        ``'zero-one normalized'``, ``'255 float'``, or ``'65535 float'``.
        String indicating the dtype of the input, which will dictate the
        preprocessing applied.

    Returns
    -------
    A :func:`numpy.array` with shape ``[X, Y, C]`` and dtype ``uint8``.

    """
    if im_arr.ndim not in [2, 3]:
        raise ValueError('This package can only read two-dimensional' +
                         'image data with an optional channel dimension.')
    if im_arr.ndim == 2:
        im_arr = im_arr[:, :, np.newaxis]
    if im_arr.shape[0] < im_arr.shape[2]:  # if the channel axis comes first
        im_arr = np.moveaxis(im_arr, 0, -1)  # move 0th axis tolast position

    # TODO: I'd like to find a better way to implement the following steps,
    # ideally by automatically determining the dtype. This could be tricky,
    # however, for images with large floating point values (i.e. is an image
    # with float values up to 245.556 actually supposed to be 8-bit or is it
    # a 16-bit that happens to have a low range of values)?

    if im_format == 'uint8':
        return im_arr.astype('uint8')  # just to be sure
    elif im_format == 'uint16':
        im_arr = (im_arr*255/65535.).astype('uint8')
    elif im_format == 'z-scored':
        im_arr = ((im_arr+1)*177.5).astype('uint8')
    elif im_format == 'zero-one normalized':
        im_arr = (im_arr*255).astype('uint8')
    elif im_format == '255 float':
        im_arr = im_arr.astype('uint8')
    elif im_format == '65535 float':
        # why are you using this format?
        im_arr = (im_arr*255/65535).astype('uint8')
    return im_arr
