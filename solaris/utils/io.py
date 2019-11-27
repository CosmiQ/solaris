"""Utility functions for data io."""
import numpy as np
import skimage.io


def imread(path, make_8bit=False, rescale=False,
           rescale_min='auto', rescale_max='auto'):
    """Read in an image file and rescale pixel values (if applicable).

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
    make_8bit : bool, optional
        Should the image be converted to an 8-bit format? Defaults to False.
    rescale : bool, optional
        Should pixel intensities be rescaled? Defaults to no (False).
    rescale_min : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The minimum pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_min`, the minimum pixel intensity in
        each channel of the image will be subtracted such that the minimum
        value becomes zero. If a single number is provided, that number will be
        subtracted from each channel. If a list of values is provided that is
        the same length as the number of channels, then those values will be
        subtracted from the corresponding channels.
    rescale_max : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The max pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_max`, each channel will be rescaled such
        that the maximum value in the channel is set to the bit range's max.
        If a single number is provided, that number will be set as the upper
        limit for all channels. If a list of values is provided that is the
        same length as the number of channels, then those values will be
        set to the maximum value in the corresponding channels.

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
    if make_8bit:
        im_arr = preprocess_im_arr(im_arr, dtype, rescale=rescale,
                                   rescale_min=rescale_min,
                                   rescale_max=rescale_max)
    return im_arr


def preprocess_im_arr(im_arr, im_format, rescale=False,
                      rescale_min='auto', rescale_max='auto'):
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
    rescale : bool, optional
        Should pixel intensities be rescaled? Defaults to no (False).
    rescale_min : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The minimum pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_min`, the minimum pixel intensity in
        each channel of the image will be subtracted such that the minimum
        value becomes zero. If a single number is provided, that number will be
        subtracted from each channel. If a list of values is provided that is
        the same length as the number of channels, then those values will be
        subtracted from the corresponding channels.
    rescale_max : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The max pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_max`, each channel will be rescaled such
        that the maximum value in the channel is set to the bit range's max.
        If a single number is provided, that number will be set as the upper
        limit for all channels. If a list of values is provided that is the
        same length as the number of channels, then those values will be
        set to the maximum value in the corresponding channels.

    Returns
    -------
    A :func:`numpy.array` with shape ``[X, Y, C]`` and dtype ``uint8``.

    """
    # get [Y, X, C] axis order set up
    if im_arr.ndim not in [2, 3]:
        raise ValueError('This package can only read two-dimensional'
                         'image data with an optional channel dimension.')
    if im_arr.ndim == 2:
        im_arr = im_arr[:, :, np.newaxis]
    if im_arr.shape[0] < im_arr.shape[2]:  # if the channel axis comes first
        im_arr = np.moveaxis(im_arr, 0, -1)  # move 0th axis tolast position

    # rescale images (if applicable)
    if rescale:
        im_arr = rescale_arr(im_arr, im_format, rescale_min, rescale_max)

    if im_format == 'uint8':
        return im_arr.astype('uint8')  # just to be sure
    elif im_format == 'uint16':
        im_arr = (im_arr.astype('float64')*255./65535.).astype('uint8')
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


def scale_for_model(image, output_type=None):
    """Scale an image to a model's required parameters.

    Arguments
    ---------
    image : :class:`np.array`
        The image array to be transformed to a desired output format.
    output_type : str, optional
        The data format of the output to pass into the model. There are five
        possible values:

        * ``'normalized'`` : values rescaled to 0-1.
        * ``'zscored'`` : image converted to zero mean and unit stdev.
        * ``'8bit'`` : image converted to 8-bit format.
        * ``'16bit'`` : image converted to 16-bit format.

        If no value is provided, no re-scaling is performed (input array is
        returned directly).
    """

    if output_type is None:
        return image
    elif output_type == 'normalized':
        out_im = image/image.max()
        return out_im
    elif output_type == 'zscored':
        return (image - np.mean(image))/np.std(image)
    elif output_type == '8bit':
        if image.max() > 255:
            # assume it's 16-bit, rescale to 8-bit scale to min/max
            out_im = 255.*image/65535
            return out_im.astype('uint8')
        elif image.max() <= 1:
            out_im = 255.*image
            return out_im.astype('uint8')
        else:
            return image.astype('uint8')
    elif output_type == '16bit':
        if (image.max() < 255) and (image.max() > 1):
            # scale to min/max
            out_im = 65535.*image/255
            return out_im.astype('uint16')
        elif image.max() <= 1:
            out_im = 65535.*image
            return out_im.astype('uint16')
        else:
            return image.astype('uint16')
    else:
        raise ValueError('output_type must be one of'
                         ' "normalized", "zscored", "8bit", "16bit"')


def rescale_arr(im_arr, im_format, rescale_min='auto', rescale_max='auto'):
    """Rescale array values in a 3D image array with channel order [Y, X, C].

    Arguments
    ---------
    im_arr : :class:`numpy.array`
        A numpy array representation of an image. `im_arr` should have either
        two or three dimensions.
    im_format : str
        One of ``'uint8'``, ``'uint16'``, ``'z-scored'``,
        ``'zero-one normalized'``, ``'255 float'``, or ``'65535 float'``.
        String indicating the dtype of the input, which will dictate the
        preprocessing applied.
    rescale_min : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The minimum pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_min`, the minimum pixel intensity in
        each channel of the image will be subtracted such that the minimum
        value becomes zero. If a single number is provided, that number will be
        subtracted from each channel. If a list of values is provided that is
        the same length as the number of channels, then those values will be
        subtracted from the corresponding channels.
    rescale_max : ``'auto'`` or :class:`int` or :class:`float` or :class:`list`
        The max pixel value(s) for rescaling. If ``rescale=True`` but no
        value is provided for `rescale_max`, each channel will be rescaled such
        that the maximum value in the channel is set to the bit range's max.
        If a single number is provided, that number will be set as the upper
        limit for all channels. If a list of values is provided that is the
        same length as the number of channels, then those values will be
        set to the maximum value in the corresponding channels.

    Returns
    -------
    normalized_arr : :class:`numpy.array`
    """

    if isinstance(rescale_min, list):
        if len(rescale_min) != im_arr.shape[2]:  # if list len != channels
            raise ValueError('The channel rescaling parameters must be '
                             'either a single value or a list of length '
                             'n_channels.')
        else:
            rescale_min = np.array(rescale_min)
    elif isinstance(rescale_min, int) or isinstance(rescale_min, float):
        rescale_min = np.array([rescale_min]*im_arr.shape[2])
    elif rescale_min == 'auto':
        rescale_min = np.amin(im_arr, axis=(0, 1))

    if isinstance(rescale_max, list):
        if len(rescale_max) != im_arr.shape[2]:  # if list len != channels
            raise ValueError('The channel rescaling parameters must be '
                             'either a single value or a list of length '
                             'n_channels.')
        else:
            rescale_max = np.array(rescale_max)
    elif isinstance(rescale_max, int) or isinstance(rescale_max, float):
        rescale_max = np.array([rescale_max]*im_arr.shape[2])
    elif rescale_max == 'auto':
        rescale_max = np.amax(im_arr, axis=(0, 1))

    scale_factor = None
    if im_format in ['uint8', '255 float']:
        scale_factor = 255
    elif im_format in ['uint16', '65535 float']:
        scale_factor = 65535
    elif im_format == 'zero-one normalized':
        scale_factor = 1

    # set all values above the scale max to the scale max, and all values
    # below the scale min to the scale min
    for channel in range(im_arr.shape[2]):
        subarr = im_arr[:, :, channel]
        subarr[subarr < rescale_min[channel]] = rescale_min[channel]
        subarr[subarr > rescale_max[channel]] = rescale_max[channel]
        im_arr[:, :, channel] = subarr

    if scale_factor is not None:
        im_arr = (im_arr-rescale_min)*(
            scale_factor/(rescale_max-rescale_min))

    return im_arr


def _check_channel_order(im_arr, framework):
    im_shape = im_arr.shape
    if len(im_shape) == 3:  # doesn't matter for 1-channel images
        if im_shape[0] > im_shape[2] and framework in ['torch', 'pytorch']:
            # in [Y, X, C], needs to be in [C, Y, X]
            im_arr = np.moveaxis(im_arr, 2, 0)
        elif im_shape[2] > im_shape[0] and framework == 'keras':
            # in [C, Y, X], needs to be in [Y, X, C]
            im_arr = np.moveaxis(im_arr, 0, 2)
    elif len(im_shape) == 4:  # for a whole minibatch
        if im_shape[1] > im_shape[3] and framework in ['torch', 'pytorch']:
            # in [Y, X, C], needs to be in [C, Y, X]
            im_arr = np.moveaxis(im_arr, 3, 1)
        elif im_shape[3] > im_shape[1] and framework == 'keras':
            # in [C, Y, X], needs to be in [Y, X, C]
            im_arr = np.moveaxis(im_arr, 1, 3)

    return im_arr
