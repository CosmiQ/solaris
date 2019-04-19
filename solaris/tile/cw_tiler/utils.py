"""cw_tiler.utils: utility functions for raster files."""


import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rio_tiler.errors import RioTilerError
from rasterio import windows
from rasterio import transform
from shapely.geometry import box


def utm_getZone(longitude):
    """Calculate UTM Zone from Longitude.

    Arguments
    ---------
    longitude: float
        longitude coordinate (Degrees.decimal degrees)

    Returns
    -------
    out: int
        UTM Zone number.

    """

    return (int(1+(longitude+180.0)/6.0))


def utm_isNorthern(latitude):
    """Determine if a latitude coordinate is in the northern hemisphere.

    Arguments
    ---------
    latitude: float
        latitude coordinate (Deg.decimal degrees)

    Returns
    -------
    out: bool
        ``True`` if `latitude` is in the northern hemisphere, ``False``
        otherwise.

    """

    return (latitude > 0.0)


def calculate_UTM_crs(coords):
    """Calculate UTM Projection String.

    Arguments
    ---------
    coords: list
        ``[longitude, latitude]`` or
        ``[min_longitude, min_latitude, max_longitude, max_latitude]`` .

    Returns
    -------
    out: str
        returns `proj4 projection string <https://proj4.org/usage/quickstart.html>`__

    """

    if len(coords) == 2:
        longitude, latitude = coords
    elif len(coords) == 4:
        longitude = np.mean([coords[0], coords[2]])
        latitude = np.mean([coords[1], coords[3]])

    utm_zone = utm_getZone(longitude)

    utm_isNorthern(latitude)

    if utm_isNorthern(latitude):
        direction_indicator = "+north"
    else:
        direction_indicator = "+south"

    utm_crs = "+proj=utm +zone={} {} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(utm_zone,
                                                                                         direction_indicator)

    return utm_crs


def get_utm_vrt(source, crs='EPSG:3857', resampling=Resampling.bilinear,
                src_nodata=None, dst_nodata=None):
    """Get a :py:class:`rasterio.vrt.WarpedVRT` projection of a dataset.

    Arguments
    ---------
    source : :py:class:`rasterio.io.DatasetReader`
        The dataset to virtually warp using :py:class:`rasterio.vrt.WarpedVRT`.
    crs : :py:class:`rasterio.crs.CRS`, optional
        Coordinate reference system for the VRT. Defaults to 'EPSG:3857'
        (Web Mercator).
    resampling : :py:class:`rasterio.enums.Resampling` method, optional
        Resampling method to use. Defaults to
        :py:func:`rasterio.enums.Resampling.bilinear`. Alternatives include
        :py:func:`rasterio.enums.Resampling.average`,
        :py:func:`rasterio.enums.Resampling.cubic`, and others. See docs for
        :py:class:`rasterio.enums.Resampling` for more information.
    src_nodata : int or float, optional
        Source nodata value which will be ignored for interpolation. Defaults
        to ``None`` (all data used in interpolation).
    dst_nodata : int or float, optional
        Destination nodata value which will be ignored for interpolation.
        Defaults to ``None``, in which case the value of `src_nodata` will be
        used if provided, or ``0`` otherwise.

    Returns
    -------
    A :py:class:`rasterio.vrt.WarpedVRT` instance with the transformation.

    """

    vrt_params = dict(
        crs=crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata,
        dst_nodata=dst_nodata)

    return WarpedVRT(source, **vrt_params)


def get_utm_vrt_profile(source, crs='EPSG:3857',
                        resampling=Resampling.bilinear,
                        src_nodata=None, dst_nodata=None):
    """Get a :py:class:`rasterio.profiles.Profile` for projection of a VRT.

    Arguments
    ---------
    source : :py:class:`rasterio.io.DatasetReader`
        The dataset to virtually warp using :py:class:`rasterio.vrt.WarpedVRT`.
    crs : :py:class:`rasterio.crs.CRS`, optional
        Coordinate reference system for the VRT. Defaults to ``"EPSG:3857"``
        (Web Mercator).
    resampling : :py:class:`rasterio.enums.Resampling` method, optional
        Resampling method to use. Defaults to
        ``rasterio.enums.Resampling.bilinear``. Alternatives include
        ``rasterio.enums.Resampling.average``,
        ``rasterio.enums.Resampling.cubic``, and others. See docs for
        :py:class:`rasterio.enums.Resampling` for more information.
    src_nodata : int or float, optional
        Source nodata value which will be ignored for interpolation. Defaults
        to ``None`` (all data used in interpolation).
    dst_nodata : int or float, optional
        Destination nodata value which will be ignored for interpolation.
        Defaults to ``None``, in which case the value of `src_nodata`
        will be used if provided, or ``0`` otherwise.

    Returns
    -------
    A :py:class:`rasterio.profiles.Profile` instance with the transformation
    applied.

    """

    with get_utm_vrt(source, crs=crs, resampling=resampling,
                     src_nodata=src_nodata, dst_nodata=dst_nodata) as vrt:
        vrt_profile = vrt.profile

    return vrt_profile


def tile_read_utm(source, bounds, tilesize, indexes=[1], nodata=None,
                  alpha=None, dst_crs='EPSG:3857', verbose=False,
                  boundless=False):
    """Read data and mask.

    Arguments
    ---------
    source : str or :py:class:`rasterio.io.DatasetReader`
        input file path or :py:class:`rasterio.io.DatasetReader` object.
    bounds : ``(W, S, E, N)`` tuple
        bounds in `dst_crs` .
    tilesize : int
        Length of one edge of the output tile in pixels.
    indexes : list of ints or int, optional
        Channel index(es) to output. Returns a 3D :py:class:`np.ndarray` of
        shape (C, Y, X) if `indexes` is a list, or a 2D array if `indexes` is
        an int channel index. Defaults to ``1``.
    nodata: int or float, optional
        nodata value to use in :py:class:`rasterio.vrt.WarpedVRT`.
        Defaults to ``None`` (use all data in warping).
    alpha: int, optional
        Force alphaband if not present in the dataset metadata. Defaults to
        ``None`` (don't force).
    dst_crs: str, optional
        Destination coordinate reference system. Defaults to ``"EPSG:3857"``
        (Web Mercator)
    verbose : bool, optional
        Verbose text output. Defaults to ``False``.
    boundless : bool, optional
        This argument is deprecated and should never be used.

    Returns
    -------
    data : :py:class:`np.ndarray`
        int pixel values. Shape is ``(C, Y, X)`` if retrieving multiple
        channels, ``(Y, X)`` otherwise.
    mask : :py:class:`np.ndarray`
        int mask indicating which pixels contain information and which are
        `nodata`. Pixels containing data have value ``255``, `nodata`
        pixels have value ``0``.
    window : :py:class:`rasterio.windows.Window`
        :py:class:`rasterio.windows.Window` object indicating the raster
        location of the dataset subregion being returned in `data`.
    window_transform : :py:class:`affine.Affine`
        Affine transformation for `window` .



    """
    w, s, e, n = bounds
    if alpha is not None and nodata is not None:
        raise RioTilerError('cannot pass alpha and nodata option')

    if isinstance(indexes, int):
        indexes = [indexes]
    out_shape = (len(indexes), tilesize, tilesize)
    if verbose:
        print(dst_crs)
    vrt_params = dict(crs=dst_crs, resampling=Resampling.bilinear,
                      src_nodata=nodata, dst_nodata=nodata)

    if not isinstance(source, DatasetReader):
        src = rasterio.open(source)
    else:
        src = source
    with WarpedVRT(src, **vrt_params) as vrt:
        window = vrt.window(w, s, e, n, precision=21)
        if verbose:
            print(window)
        window_transform = transform.from_bounds(w, s, e, n,
                                                 tilesize, tilesize)

        data = vrt.read(window=window,
                        resampling=Resampling.bilinear,
                        out_shape=out_shape,
                        indexes=indexes)
        if verbose:
            print(bounds)
            print(window)
            print(out_shape)
            print(indexes)
            print(boundless)
            print(window_transform)

        if nodata is not None:
            mask = np.all(data != nodata, axis=0).astype(np.uint8) * 255
        elif alpha is not None:
            mask = vrt.read(alpha, window=window,
                            out_shape=(tilesize, tilesize),
                            resampling=Resampling.bilinear)
        else:
            mask = vrt.read_masks(1, window=window,
                                  out_shape=(tilesize, tilesize),
                                  resampling=Resampling.bilinear)
    return data, mask, window, window_transform


def tile_exists_utm(boundsSrc, boundsTile):
    """Check if suggested tile is within bounds.

    Arguments
    ---------
    boundsSrc : list-like
        Bounding box limits for the source data in the shape ``(W, S, E, N)``.
    boundsTile : list-like
        Bounding box limits for the target tile in the shape ``(W, S, E, N)``.

    Returns
    -------
    bool
        Do the `boundsSrc` and `boundsTile` bounding boxes overlap?

    """

    boundsSrcBox = box(*boundsSrc)
    boundsTileBox = box(*boundsTile)

    return boundsSrcBox.intersects(boundsTileBox)


def get_wgs84_bounds(source):
    """Transform dataset bounds from source crs to wgs84.

    Arguments
    ---------
    source : str or :py:class:`rasterio.io.DatasetReader`
        Source dataset to get bounds transformation for. Can either be a string
        path to a dataset file or an opened
        :py:class:`rasterio.io.DatasetReader`.

    Returns
    -------
    wgs_bounds : tuple
        Bounds tuple for `source` in wgs84 crs with shape ``(W, S, E, N)``.

    """
    if isinstance(source, DatasetReader):
        src = source
    else:
        src = rasterio.open(source)
    wgs_bounds = transform_bounds(*[src.crs, 'epsg:4326'] +
                                  list(src.bounds), densify_pts=21)
    return wgs_bounds


def get_utm_bounds(source, utm_EPSG):
    """Transform bounds from source crs to a UTM crs.

    Arguments
    ---------
    source : str or :py:class:`rasterio.io.DatasetReader`
        Source dataset. Can either be a string path to a dataset GeoTIFF or
        a :py:class:`rasterio.io.DatasetReader` object.
    utm_EPSG : str
        :py:class:`rasterio.crs.CRS` string indicating the UTM crs to transform
        into.

    Returns
    -------
    utm_bounds : tuple
        Bounding box limits in `utm_EPSG` crs coordinates with shape
        ``(W, S, E, N)``.

    """
    if isinstance(source, DatasetReader):
        src = source
    else:
        src = rasterio.open(source)
    utm_bounds = transform_bounds(*[src.crs, utm_EPSG] + list(src.bounds),
                                  densify_pts=21)
    return utm_bounds
