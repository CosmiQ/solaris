import os
import shapely
from affine import Affine
import rasterio
from rasterio.warp import transform_bounds
from ..utils.geo import list_to_affine, _reduce_geom_precision
from ..utils.core import _check_gdf_load
from ..raster_image.image import get_geo_transform
from shapely.geometry import box, Polygon
import pandas as pd
import geopandas as gpd
from rtree.core import RTreeError


def convert_poly_coords(geom, raster_src=None, affine_obj=None, inverse=False,
                        precision=None):
    """Georegister geometry objects currently in pixel coords or vice versa.

    Arguments
    ---------
    geom : :class:`shapely.geometry.shape` or str
        A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
        object currently in pixel coordinates.
    raster_src : str, optional
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    affine_obj: list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    inverse : bool, optional
        If true, will perform the inverse affine transformation, going from
        geospatial coordinates to pixel coordinates.
    precision : int, optional
        Decimal precision for the polygon output. If not provided, rounding
        is skipped.

    Returns
    -------
    out_geom
        A geometry in the same format as the input with its coordinate system
        transformed to match the destination object.
    """

    if not raster_src and not affine_obj:
        raise ValueError("Either raster_src or affine_obj must be provided.")

    if raster_src is not None:
        affine_xform = get_geo_transform(raster_src)
    else:
        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            # (list_to_affine checks which it is)
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = list_to_affine(affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. ' +
                        'This function can only accept WKT strings and ' +
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)
    if precision is not None:
        xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

    return xformed_g


def affine_transform_gdf(gdf, affine_obj, inverse=False, geom_col="geometry",
                         precision=None):
    """Perform an affine transformation on a GeoDataFrame.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`, :class:`pandas.DataFrame`, or `str`
        A GeoDataFrame, pandas DataFrame with a ``"geometry"`` column (or a
        different column containing geometries, identified by `geom_col` -
        note that this column will be renamed ``"geometry"`` for ease of use
        with geopandas), or the path to a saved file in .geojson or .csv
        format.
    affine_obj : list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
    inverse : bool, optional
        Use this argument to perform the inverse transformation.
    geom_col : str, optional
        The column in `gdf` corresponding to the geometry. Defaults to
        ``'geometry'``.
    precision : int, optional
        Decimal precision to round the geometries to. If not provided, no
        rounding is performed.
    """
    if isinstance(gdf, str):  # assume it's a geojson
        if gdf.lower().endswith('json'):
            gdf = gpd.read_file(gdf)
        elif gdf.lower().endswith('csv'):
            gdf = pd.read_csv(gdf)
            gdf = gdf.rename(columns={geom_col: 'geometry'})
            if not isinstance(gdf['geometry'][0], Polygon):
                gdf['geometry'] = gdf['geometry'].apply(shapely.wkt.loads)
        else:
            raise ValueError(
                "The file format is incompatible with this function.")
    gdf["geometry"] = gdf["geometry"].apply(convert_poly_coords,
                                            affine_obj=affine_obj,
                                            inverse=inverse)
    if precision is not None:
        gdf['geometry'] = gdf['geometry'].apply(
            _reduce_geom_precision, precision=precision)
    return gdf


def georegister_px_df(df, im_path=None, affine_obj=None, crs=None,
                      geom_col='geometry', precision=None, output_path=None):
    """Convert a dataframe of geometries in pixel coordinates to a geo CRS.

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` with polygons in a column named
        ``"geometry"``.
    im_path : str, optional
        A filename or :class:`rasterio.DatasetReader` object containing an
        image that has the same bounds as the pixel coordinates in `df`. If
        not provided, `affine_obj` and `crs` must both be provided.
    affine_obj : `list` or :class:`affine.Affine`, optional
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    crs : dict, optional
        The coordinate reference system for the output GeoDataFrame. Required
        if not providing a raster image to extract the information from. Format
        should be ``{'init': 'epsgxxxx'}``, replacing xxxx with the EPSG code.
    geom_col : str, optional
        The column containing geometry in `df`. If not provided, defaults to
        ``"geometry"``.
    precision : int, optional
        The decimal precision for output geometries. If not provided, the
        vertex locations won't be rounded.
    output_path : str, optional
        Path to save the resulting output to. If not provided, the object
        won't be saved to disk.

    """
    if im_path is not None:
        affine_obj = rasterio.open(im_path).transform
        crs = rasterio.open(im_path).crs
    else:
        if not affine_obj or not crs:
            raise ValueError(
                'If an image path is not provided, ' +
                'affine_obj and crs must be.')
    tmp_df = affine_transform_gdf(df, affine_obj, geom_col=geom_col,
                                  precision=precision)
    result = gpd.GeoDataFrame(tmp_df, crs=crs)

    if output_path is not None:
        if output_path.lower().endswith('json'):
            result.to_file(output_path, driver='GeoJSON')
        else:
            result.to_csv(output_path, index=False)

    return result


def geojson_to_px_gdf(geojson, im_path, geom_col='geometry', precision=None,
                      output_path=None):
    """Convert a geojson or set of geojsons from geo coords to px coords.

    Arguments
    ---------
    geojson : str
        Path to a geojson. This function will also accept a
        :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` with a
        column named ``'geometry'`` in this argument.
    im_path : str
        Path to a georeferenced image (ie a GeoTIFF) that geolocates to the
        same geography as the `geojson`(s). If a directory, the bounds of each
        GeoTIFF will be loaded in and all overlapping geometries will be
        transformed. This function will also accept a
        :class:`osgeo.gdal.Dataset` or :class:`rasterio.DatasetReader` with
        georeferencing information in this argument.
    geom_col : str, optional
        The column containing geometry in `geojson`. If not provided, defaults
        to ``"geometry"``.
    precision : int, optional
        The decimal precision for output geometries. If not provided, the
        vertex locations won't be rounded.
    output_path : str, optional
        Path to save the resulting output to. If not provided, the object
        won't be saved to disk.

    Returns
    -------
    output_df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` with all geometries in `geojson` that
        overlapped with the image at `im_path` converted to pixel coordinates.
        Additional columns are included with the filename of the source
        geojson (if available) and images for reference.

    """
    # get the bbox and affine transforms for the image
    if isinstance(im_path, str):
        bbox = box(*rasterio.open(im_path).bounds)
        affine_obj = rasterio.open(im_path).transform
        im_crs = rasterio.open(im_path).crs

    else:
        bbox = box(im_path.bounds)
        affine_obj = im_path.transform
        im_crs = im_path.crs

    # make sure the geo vector data is loaded in as geodataframe(s)
    gdf = _check_gdf_load(geojson)

    overlap_gdf = get_overlapping_subset(gdf, bbox=bbox, bbox_crs=im_crs)
    transformed_gdf = affine_transform_gdf(overlap_gdf, affine_obj=affine_obj,
                                           inverse=True, precision=precision,
                                           geom_col=geom_col)
    transformed_gdf['image_fname'] = os.path.split(im_path)[1]

    if output_path is not None:
        if output_path.lower().endswith('json'):
            transformed_gdf.to_file(output_path, driver='GeoJSON')
        else:
            transformed_gdf.to_csv(output_path, index=False)
    return transformed_gdf


def get_overlapping_subset(gdf, im=None, bbox=None, bbox_crs=None):
    """Extract a subset of geometries in a GeoDataFrame that overlap with `im`.

    Notes
    -----
    This function uses RTree's spatialindex, which is much faster (but slightly
    less accurate) than direct comparison of each object for overlap.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` instance or a path to a geojson.
    im : :class:`rasterio.DatasetReader` or `str`, optional
        An image object loaded with `rasterio` or a path to a georeferenced
        image (i.e. a GeoTIFF).
    bbox : `list` or :class:`shapely.geometry.Polygon`, optional
        A bounding box (either a :class:`shapely.geometry.Polygon` or a
        ``[bottom, left, top, right]`` `list`) from an image. Has no effect
        if `im` is provided (`bbox` is inferred from the image instead.) If
        `bbox` is passed and `im` is not, a `bbox_crs` should be provided to
        ensure correct geolocation - if it isn't, it will be assumed to have
        the same crs as `gdf`.

    Returns
    -------
    output_gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` with all geometries in `gdf` that
        overlapped with the image at `im`.
        Coordinates are kept in the CRS of `gdf`.

    """
    if not im and not bbox:
        raise ValueError('Either `im` or `bbox` must be provided.')
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    if isinstance(im, str):
        im = rasterio.open(im)
    sindex = gdf.sindex
    # use transform_bounds in case the crs is different - no effect if not
    if im:
        bbox = transform_bounds(im.crs, gdf.crs, *im.bounds)
    else:
        if isinstance(bbox, Polygon):
            bbox = bbox.bounds
        if not bbox_crs:
            bbox_crs = gdf.crs
        bbox = transform_bounds(bbox_crs, gdf.crs, *bbox)
    try:
        intersectors = list(sindex.intersection(bbox))
    except RTreeError:
        intersectors = []

    return gdf.iloc[intersectors, :]
