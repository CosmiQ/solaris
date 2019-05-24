from osgeo import gdal
import rasterio
from affine import Affine
import numpy as np
from ..utils.raster import reorder_axes


def get_geo_transform(raster_src):
    """Get the geotransform for a raster image source.

    Arguments
    ---------
    raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.

    Returns
    -------
    transform : :class:`affine.Affine`
        An affine transformation object to the image's location in its CRS.
    """

    if isinstance(raster_src, str):
        affine_obj = rasterio.open(raster_src).transform
    elif isinstance(raster_src, rasterio.DatasetReader):
        affine_obj = raster_src.transform
    elif isinstance(raster_src, gdal.Dataset):
        affine_obj = Affine.from_gdal(*raster_src.GetGeoTransform())

    return affine_obj


def stitch_images(im_arr, idx_refs=None, out_width=None,
                  out_height=None, method='average', use_GPU=True):
    """Stitch together images into a single 2- or 3-channel array.

    Arguments
    ---------
    im_arr : :class:`numpy.array` or :class:`list` of :class:`numpy.array` s
        A 3- or 4-D :class:`numpy.array` with shape ``[N, Y, X(, C)]`` or a
        list of length N made up of 2- or 3-D tensors with shape
        ``[Y, X(, C)]``. These array(s) will be stitched together to produce a
        single output of shape ``[Y, X(, C)]`` .
    idx_refs : list, optional
        A list of ``(Y, X)`` indices for each sub-array to define the location
        of the first corner in the final output. Used for stitching together
        non-overlapping or partially overlapping tiles into a single output.
        Note that the index reference output of
        :class:`solaris.nets.datagen.InferenceTiler` provides the required
        reference system for stitching here.
    out_width : int, optional
        The width of the output array in pixels. If not provided, it is assumed
        that the width is the same as the width of ``im_arr`` .
    out_height : int, optional
        The height of the output array in pixels. If not provided, it is
        assumed that the height is the same as the height of ``im_arr`` .
    method : str, optional
        possible values are ``'average'``  (default), ``'first'`` , and
        ``'confidence'`` .
        * If ``'average'`` , all pixels corresponding to the same location in
        ``[Y, X, C]`` space are averaged.
        * If ``'first'`` , the value of the first pixel along the ``N`` axis
        for a given ``[Y, X, C]`` location is selected.
        * If ``'confidence'`` , it's assumed that pixel values correspond to
        probabilities in ``[0, 1]`` . In this case, for a given ``[Y, X, C]``
        location, the pixel with the greatest distance from ``0.5`` will be
        selected (being the value with the highest confidence).
    use_GPU : bool, optional
        Should processing be performed on the GPU if a GPU is available?
        Defaults to yes (``True``). If a GPU isn't available, this argument is
        ignored. ``False`` will force CPU-located processing.

    Returns
    -------
    output_arr : a :class:`numpy.array` with shape ``[Y, X(, C)]`` .
    """
    # determine what shape the input is and stitch together accordingly
    if isinstance(im_arr, list):
        im_arr = np.stack(im_arr)  # stack along a new 1st axis

    im_arr = reorder_axes(im_arr, 'tensorflow')

    if idx_refs is not None:
        if len(idx_refs) != im_arr.shape[0]:
            raise ValueError('len(idx_refs) must be equal to the number of '
                             'images being stitched.')
    if idx_refs is not None and (out_width is None or out_height is None):
        raise ValueError('If idx_refs are provided, the desired '
                         'out_height and out_width must be provided as well.')
    if len(im_arr.shape) == 4:
        has_channels = True
    elif len(im_arr.shape) == 3:
        has_channels = False

    if idx_refs is not None:  # proxy for whether dims were provided as args
        if has_channels:
            stitching_arr = np.empty(shape=(im_arr.shape[0],
                                            out_height, out_width,
                                            im_arr.shape[3]))
        else:
            stitching_arr = np.empty(shape=(im_arr.shape[0],
                                            out_height, out_width))
        stitching_arr[:] = np.nan
        for idx in range(len(idx_refs)):
            if has_channels:
                stitching_arr[
                    idx,
                    idx_refs[idx][0]:idx_refs[idx][0]+im_arr.shape[1],
                    idx_refs[idx][1]:idx_refs[idx][1]+im_arr.shape[2],
                    :] = im_arr[idx, :, :, :]
            else:
                stitching_arr[
                    idx,
                    idx_refs[idx][0]:idx_refs[idx][0]+im_arr.shape[1],
                    idx_refs[idx][1]:idx_refs[idx][1]+im_arr.shape[2]
                    ] = im_arr[idx, :, :]
    else:
        stitching_arr = im_arr  # just stitching across images with no offset

    if method == 'average':
        output_arr = np.nanmean(stitching_arr, axis=0)

    elif method == 'first':
        # get index along 1st axis of the first non-NaN value
        first_non_nan = np.invert(np.isnan(stitching_arr)).argmax(axis=0)
        # subset along 1st axis for only the first non-NaN value
        output_arr = np.take_along_axis(stitching_arr,
                                        np.expand_dims(first_non_nan, axis=0),
                                        axis=0)[0, :, :, :]  # drop extra axis

    elif method == 'confidence':
        # convert from 0-1 to 0-0.5, values originally 0.5 become 0
        conf_scale = np.abs(stitching_arr - 0.5)
        # set NaN values to -1 so they're never selected
        conf_scale[np.isnan(conf_scale)] = -1
        # get highest conf slice at each [Y, X, C] position
        max_conf_ind = conf_scale.argmax(axis=0)
        # subset to take only the highest-conf value
        output_arr = np.take_along_axis(stitching_arr,
                                        np.expand_dims(max_conf_ind, axis=0),
                                        axis=0)[0, :, :, :]  # drop extra axis
    output_arr = output_arr.astype(im_arr.dtype)

    return output_arr
