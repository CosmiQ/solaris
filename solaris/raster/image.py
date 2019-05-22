from osgeo import gdal
import rasterio
from affine import Affine
import numpy as np


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
                  out_height=None, method='average'):
    """Stitch together images into a single 2- or 3-channel array.

    Arguments
    ---------
    im_arr : :class:`numpy.array` or :class:`list` of :class:`numpy.array` s
        A 3- or 4-D array with shape ``[N, Y, X(, C)]`` or a list of length N
        made up of 2- or 3-D arrays with shape ``[Y, X(, C)]`` . These array(s)
        will be stitched together to produce a single output of shape
        ``[Y, X(, C)]`` .
    idx_refs : list, optional
        A list of ``(Y, X)`` indices for each sub-array to define the location
        of the first corner in the final output. Used for stitching together
        non-overlapping or partially overlapping tiles into a single output.
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

    Returns
    -------
    output_arr : a :class:`numpy.array` with shape ``[Y, X(, C)]`` .
    """
    # determine what shape the input is and stitch together accordingly
    if isinstance(im_arr, list):
        im_arr = np.stack(im_arr)  # stack along a new 1st axis

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
        desired_inds = np.isnan(stitching_arr).argmax(axis=0)
        output_arr = stitching_arr.take(desired_inds, axis=0)
    elif method == 'confidence':
        max_conf = np.abs(stitching_arr - 0.5).argmax(axis=0)
        output_arr = stitching_arr.take(max_conf, axis=0)

    return output_arr
