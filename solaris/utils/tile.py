import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio import transform


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


def clip_gdf(gdf, poly_to_cut, min_partial_perc=0.0, geom_type="Polygon",
             use_sindex=True):
    """Clip GDF to a provided polygon.

    Note
    ----
    Clips objects within `gdf` to the region defined by
    `poly_to_cut`. Also adds several columns to the output:

    `origarea`
        The original area of the polygons (only used if `geom_type` ==
        ``"Polygon"``).
    `origlen`
        The original length of the objects (only used if `geom_type` ==
        ``"LineString"``).
    `partialDec`
        The fraction of the object that remains after clipping
        (fraction of area for Polygons, fraction of length for
        LineStrings.) Can filter based on this by using `min_partial_perc`.
    `truncated`
        Boolean indicator of whether or not an object was clipped.

    Arguments
    ---------
    gdf : :py:class:`geopandas.GeoDataFrame`
        A :py:class:`geopandas.GeoDataFrame` of polygons to clip.
    poly_to_cut : :py:class:`shapely.geometry.Polygon`
        The polygon to clip objects in `gdf` to.
    min_partial_perc : float, optional
        The minimum fraction of an object in `gdf` that must be
        preserved. Defaults to 0.0 (include any object if any part remains
        following clipping).
    geom_type : str, optional
        Type of objects in `gdf`. Can be one of
        ``["Polygon", "LineString"]`` . Defaults to ``"Polygon"`` .
    use_sindex : bool, optional
        Use the `gdf` sindex be used for searching. Improves efficiency
        but requires `libspatialindex <http://libspatialindex.github.io/>`__ .

    Returns
    -------
    cutGeoDF : :py:class:`geopandas.GeoDataFrame`
        `gdf` with all contained objects clipped to `poly_to_cut` .
        See notes above for details on additional clipping columns added.

    """

    # check if geoDF has origAreaField

    if use_sindex:
        gdf = search_gdf_polygon(gdf, poly_to_cut)

    # if geom_type == "LineString":
    if 'origarea' in gdf.columns:
        pass
    else:
        if "geom_type" == "LineString":
            gdf['origarea'] = 0
        else:
            gdf['origarea'] = gdf.area
    if 'origlen' in gdf.columns:
        pass
    else:
        if "geom_type" == "LineString":
            gdf['origlen'] = gdf.length
        else:
            gdf['origlen'] = 0
    # TODO must implement different case for lines and for spatialIndex
    # (Assume RTree is already performed)

    cutGeoDF = gdf.copy()
    cutGeoDF.geometry = gdf.intersection(poly_to_cut)

    if geom_type == 'Polygon':
        cutGeoDF['partialDec'] = cutGeoDF.area / cutGeoDF['origarea']
        cutGeoDF = cutGeoDF.loc[cutGeoDF['partialDec'] > min_partial_perc, :]
        cutGeoDF['truncated'] = (cutGeoDF['partialDec'] != 1.0).astype(int)
    else:
        cutGeoDF = cutGeoDF[cutGeoDF.geom_type != "GeometryCollection"]
        cutGeoDF['partialDec'] = 1
        cutGeoDF['truncated'] = 0
    # TODO: IMPLEMENT TRUNCATION MEASUREMENT FOR LINESTRINGS

    return cutGeoDF
