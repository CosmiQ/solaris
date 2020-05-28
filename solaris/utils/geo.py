import os
from .core import _check_df_load, _check_gdf_load, _check_rasterio_im_load
from .core import _check_geom, _check_crs
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
import rasterio
from rasterio.warp import calculate_default_transform, Resampling
from rasterio.warp import transform_bounds
from shapely.affinity import affine_transform
from shapely.wkt import loads
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry import MultiLineString, MultiPolygon, mapping, box, shape
from shapely.geometry.collection import GeometryCollection
from shapely.ops import cascaded_union
import osr
import gdal
import json
from warnings import warn
import sys


def reproject(input_object, input_crs=None,
              target_crs=None, target_object=None, dest_path=None,
              resampling_method='cubic'):
    """Reproject a dataset (df, gdf, or image) to a new coordinate system.

    This function takes a georegistered image or a dataset of vector geometries
    and converts them to a new coordinate reference system. If no target CRS
    is provided, the data will be converted to the appropriate UTM zone by
    default. To convert a pixel-coordinate dataset to geographic coordinates or
    vice versa, use :func:`solaris.vector.polygon.georegister_px_df` or
    :func:`solaris.vector.polygon.geojson_to_px_gdf` instead.

    Arguments
    ---------
    input_object : `str` or :class:`rasterio.DatasetReader` or :class:`gdal.Dataset` or :class:`geopandas.GeoDataFrame`
        An object to transform to a new CRS. If a string, it must be a path
        to a georegistered image or vector dataset (e.g. a .GeoJSON). If the
        object itself does not contain georeferencing information, the
        coordinate reference system can be provided with `input_crs`.
    input_crs : int, optional
        The EPSG code integer for the input data's CRS. If provided and a CRS
        is also associated with `input_object`, this argument's value has
        precedence.
    target_crs : int, optional
        The EPSG code for the output projection. If values are not provided
        for this argument or `target_object`, the input data will be
        re-projected into the appropriate UTM zone. If both `target_crs` and
        `target_object` are provided, `target_crs` takes precedence (and a
        warning is raised).
    target_object : str, optional
        An object in the desired destination CRS. If neither this argument nor
        `target_crs` is provided, the input will be projected into the
        appropriate UTM zone. `target_crs` takes precedence if both it and
        `target_object` are provided.
    dest_path : str, optional
        The path to save the output to (if desired). This argument is only
        required if the input is a :class:`gdal.Dataset`; otherwise, it is
        optional.
    resampling_method : str, optional
        The resampling method to use during reprojection of raster data. **Only
        has an effect if the input is a :class:`rasterio.DatasetReader` !**
        Possible values are
        ``['cubic' (default), 'bilinear', 'nearest', 'average']``.

    Returns
    -------
    output : :class:`rasterio.DatasetReader` or :class:`gdal.Dataset` or :class:`geopandas.GeoDataFrame`
        An output in the same format as `input_object`, but reprojected
        into the destination CRS.
    """
    input_data, input_type = _parse_geo_data(input_object)
    if input_crs is None:
        input_crs = _check_crs(get_crs(input_data))
    else:
        input_crs = _check_crs(input_crs)
    if target_object is not None:
        target_data, _ = _parse_geo_data(target_object)
    else:
        target_data = None
    # get CRS from target_object if it's not provided
    if target_crs is None and target_data is not None:
        target_crs = get_crs(target_data)

    if target_crs is not None:
        target_crs = _check_crs(target_crs)
        output = _reproject(input_data, input_type, input_crs, target_crs,
                            dest_path, resampling_method)
    else:
        output = reproject_to_utm(input_data, input_type, input_crs,
                                  dest_path, resampling_method)
    return output


def _reproject(input_data, input_type, input_crs, target_crs, dest_path,
               resampling_method='bicubic'):

    input_crs = _check_crs(input_crs)
    target_crs = _check_crs(target_crs)
    if input_type == 'vector':
        output = input_data.to_crs(target_crs)
        if dest_path is not None:
            output.to_file(dest_path, driver='GeoJSON')

    elif input_type == 'raster':

        if isinstance(input_data, rasterio.DatasetReader):
            transform, width, height = calculate_default_transform(
                input_crs.to_wkt("WKT1_GDAL"), target_crs.to_wkt("WKT1_GDAL"),
                input_data.width, input_data.height, *input_data.bounds
            )
            kwargs = input_data.meta.copy()
            kwargs.update({'crs': target_crs.to_wkt("WKT1_GDAL"),
                           'transform': transform,
                           'width': width,
                           'height': height})

            if dest_path is not None:
                with rasterio.open(dest_path, 'w', **kwargs) as dst:
                    for band_idx in range(1, input_data.count + 1):
                        rasterio.warp.reproject(
                            source=rasterio.band(input_data, band_idx),
                            destination=rasterio.band(dst, band_idx),
                            src_transform=input_data.transform,
                            src_crs=input_data.crs,
                            dst_transform=transform,
                            dst_crs=target_crs.to_wkt("WKT1_GDAL"),
                            resampling=getattr(Resampling, resampling_method)
                        )
                output = rasterio.open(dest_path)
                input_data.close()

            else:
                output = np.zeros(shape=(height, width, input_data.count))
                for band_idx in range(1, input_data.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(input_data, band_idx),
                        destination=output[:, :, band_idx-1],
                        src_transform=input_data.transform,
                        src_crs=input_data.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=getattr(Resampling, resampling_method)
                    )

        elif isinstance(input_data, gdal.Dataset):
            if dest_path is not None:
                gdal.Warp(dest_path, input_data,
                          dstSRS='EPSG:' + str(target_crs.to_epsg()))
                output = gdal.Open(dest_path)
            else:
                raise ValueError('An output path must be provided for '
                                 'reprojecting GDAL datasets.')
    return output


def reproject_to_utm(input_data, input_type, input_crs=None, dest_path=None,
                     resampling_method='bicubic'):
    """Convert an input to a UTM CRS (after determining the correct UTM zone).

    """
    if input_crs is None:
        input_crs = get_crs(input_data)
    if input_crs is None:
        raise ValueError('An input CRS must be provided by input_data or'
                         ' input_crs.')
    input_crs = _check_crs(input_crs)

    bounds = get_bounds(input_data, crs=_check_crs(4326))  # need in wkt84 for UTM zone
    midpoint = [(bounds[1] + bounds[3])/2., (bounds[0] + bounds[2])/2.]
    utm_epsg = latlon_to_utm_epsg(*midpoint)

    output = _reproject(input_data, input_type=input_type, input_crs=input_crs,
                        target_crs=utm_epsg, dest_path=dest_path,
                        resampling_method=resampling_method)
    # cleanup
    if os.path.isfile('tmp'):
        os.remove('tmp')

    return output


def get_bounds(geo_obj, crs=None):
    """Get the ``[left, bottom, right, top]`` bounds in any CRS.

    Arguments
    ---------
    geo_obj : a georeferenced raster or vector dataset.
    crs : int, optional
        The EPSG code (or other CRS format supported by rasterio.warp)
        for the CRS the bounds should be returned in. If not provided,
        the bounds will be returned in the same crs as `geo_obj`.

    Returns
    -------
    bounds : list
        ``[left, bottom, right, top]`` bounds in the input crs (if `crs` is
        ``None``) or in `crs` if it was provided.
    """
    input_data, input_type = _parse_geo_data(geo_obj)
    if input_type == 'vector':
        bounds = list(input_data.geometry.total_bounds)
    elif input_type == 'raster':
        if isinstance(input_data, rasterio.DatasetReader):
            bounds = list(input_data.bounds)
        elif isinstance(input_data, gdal.Dataset):
            input_gt = input_data.GetGeoTransform()
            min_x = input_gt[0]
            max_x = min_x + input_gt[1]*input_data.RasterXSize
            max_y = input_gt[3]
            min_y = max_y + input_gt[5]*input_data.RasterYSize

            bounds = [min_x, min_y, max_x, max_y]

    if crs is not None:
        crs = _check_crs(crs)
        src_crs = get_crs(input_data)
        # transform bounds to desired CRS
        bounds = transform_bounds(src_crs.to_wkt("WKT1_GDAL"),
                                  crs.to_wkt("WKT1_GDAL"), *bounds)

    return bounds


def get_crs(obj):
    """Get a coordinate reference system from any georegistered object."""
    if isinstance(obj, gpd.GeoDataFrame):
        return _check_crs(obj.crs)
    elif isinstance(obj, rasterio.DatasetReader):
        return _check_crs(obj.crs)
    elif isinstance(obj, gdal.Dataset):
        # rawr
        return _check_crs(int(osr.SpatialReference(wkt=obj.GetProjection()).GetAttrValue(
            'AUTHORITY', 1)))
    else:
        raise TypeError("solaris doesn't know how to extract a crs from an "
                        "object of type {}".format(type(obj)))


def _parse_geo_data(input):
    if isinstance(input, str):
        if input.lower().endswith('json') or input.lower().endswith('csv'):
            input_type = 'vector'
            input_data = _check_df_load(input)
        elif input.lower().endswith('tif') or input.lower().endswith('tiff'):
            input_type = 'raster'
            input_data = _check_rasterio_im_load(input)
    else:
        input_data = input
        if isinstance(input_data, pd.DataFrame):
            input_type = 'vector'
        elif isinstance(
                input_data, rasterio.DatasetReader
        ) or isinstance(
                input_data, gdal.Dataset
        ):
            input_type = 'raster'
        else:
            raise ValueError('The input format {} is not compatible with '
                             'solaris.'.format(type(input)))
    return input_data, input_type


def reproject_geometry(input_geom, input_crs=None, target_crs=None,
                       affine_obj=None):
    """Reproject a geometry or coordinate into a new CRS.

    Arguments
    ---------
    input_geom : `str`, `list`, or `Shapely <https://shapely.readthedocs.io>`_ geometry
        A geometry object to re-project. This can be a 2-member ``list``, in
        which case `input_geom` is assumed to coorespond to ``[x, y]``
        coordinates in `input_crs`. It can also be a Shapely geometry object or
        a wkt string.
    input_crs : int, optional
        The coordinate reference system for `input_geom`'s coordinates, as an
        EPSG :class:`int`. Required unless `affine_transform` is provided.
    target_crs : int, optional
        The target coordinate reference system to re-project the geometry into.
        If not provided, the appropriate UTM zone will be selected by default,
        unless `affine_transform` is provided (and therefore CRSs are ignored.)
    affine_transform : :class:`affine.Affine`, optional
        An :class:`affine.Affine` object (or a ``[a, b, c, d, e, f]`` list to
        convert to that format) to use for transformation. Has no effect unless
        `input_crs` **and** `target_crs` are not provided.

    Returns
    -------
    output_geom : Shapely geometry
        A shapely geometry object:
        - in `target_crs`, if one was provided;
        - in the appropriate UTM zone, if `input_crs` was provided and
          `target_crs` was not;
        - with `affine_transform` applied to it if neither `input_crs` nor
          `target_crs` were provided.
    """
    input_geom = _check_geom(input_geom)

    if input_crs is not None:
        input_crs = _check_crs(input_crs)
        if target_crs is None:
            geom = reproject_geometry(input_geom, input_crs,
                                      target_crs=_check_crs(4326))
            target_crs = latlon_to_utm_epsg(geom.centroid.y, geom.centroid.x)
        target_crs = _check_crs(target_crs)
        gdf = gpd.GeoDataFrame(geometry=[input_geom], crs=input_crs.to_wkt())
        # create a new instance of the same geometry class as above with the
        # new coordinates
        output_geom = gdf.to_crs(target_crs.to_wkt()).iloc[0]['geometry']

    else:
        if affine_obj is None:
            raise ValueError('If an input CRS is not provided, '
                             'affine_transform is required to complete the '
                             'transformation.')
        elif isinstance(affine_obj, Affine):
            affine_obj = affine_to_list(affine_obj)

        output_geom = affine_transform(input_geom, affine_obj)

    return output_geom


def gdf_get_projection_unit(vector_file):
    """Get the projection unit for a vector_file or gdf.

    Arguments
    ---------
    vector_file : :py:class:`geopandas.GeoDataFrame` or geojson/shapefile
        A vector file or gdf with georeferencing

    Notes
    -----
    If vector file is already in UTM coords, the projection WKT is complex:
        https://www.spatialreference.org/ref/epsg/wgs-84-utm-zone-11n/html/
    In this case, return the second instance of 'UNIT'.

    Returns
    -------
    unit : String
        The unit i.e. meter, metre, or degree, of the projection
    """
    c = _check_gdf_load(vector_file).crs
    return get_projection_unit(c)


def raster_get_projection_unit(image):
    """Get the projection unit for an image.

    Arguments
    ---------
    image : raster image, GeoTIFF or other format
        A raster file with georeferencing

    Notes
    -----
    If raster is already in UTM coords, the projection WKT is complex:
        https://www.spatialreference.org/ref/epsg/wgs-84-utm-zone-11n/html/
    In this case, return the second instance of 'UNIT'.

    Returns
    -------
    unit : String
        The unit i.e. meters or degrees, of the projection
    """
    c = _check_rasterio_im_load(image).crs
    return get_projection_unit(c)


def get_projection_unit(crs):
    """Get the units of a specific SRS.

    Arguments
    ---------
    crs : :class:`pyproj.crs.CRS`, :class:`rasterio.crs.CRS`, `str`, or `int`
        The coordinate reference system to retrieve a unit for.

    Returns
    -------
    unit : str
        The string-formatted unit.
    """
    crs = _check_crs(crs)
    unit = crs.axis_info[0].unit_name

    return unit



def list_to_affine(xform_mat):
    """Create an Affine from a list or array-formatted [a, b, d, e, xoff, yoff]

    Arguments
    ---------
    xform_mat : `list` or :class:`numpy.array`
        A `list` of values to convert to an affine object.

    Returns
    -------
    aff : :class:`affine.Affine`
        An affine transformation object.
    """
    # first make sure it's not in gdal order
    if len(xform_mat) > 6:
        xform_mat = xform_mat[0:6]
    if rasterio.transform.tastes_like_gdal(xform_mat):
        return Affine.from_gdal(*xform_mat)
    else:
        return Affine(*xform_mat)


def affine_to_list(affine_obj):
    """Convert a :class:`affine.Affine` instance to a list for Shapely."""
    return [affine_obj.a, affine_obj.b,
            affine_obj.d, affine_obj.e,
            affine_obj.xoff, affine_obj.yoff]


def geometries_internal_intersection(polygons):
    """Get the intersection geometries between all geometries in a set.

    Arguments
    ---------
    polygons : `list`-like
        A `list`-like containing geometries. These will be placed in a
        :class:`geopandas.GeoSeries` object to take advantage of `rtree`
        spatial indexing.

    Returns
    -------
    intersect_list
        A `list` of geometric intersections between polygons in `polygons`, in
        the same CRS as the input.
    """
    # convert `polygons` to geoseries and get spatialindex
    # TODO: Implement test to see if `polygon` items are actual polygons or
    # WKT strings
    if isinstance(polygons, gpd.GeoSeries):
        gs = polygons
    else:
        gs = gpd.GeoSeries(polygons).reset_index(drop=True)
    sindex = gs.sindex
    gs_bboxes = gs.apply(lambda x: x.bounds)

    # find indices of polygons that overlap in gs
    intersect_lists = gs_bboxes.apply(lambda x: list(sindex.intersection(x)))
    intersect_lists = intersect_lists.dropna()
    # drop all objects that only have self-intersects
    # first, filter down to the ones that have _some_ intersection with others
    intersect_lists = intersect_lists[
        intersect_lists.apply(lambda x: len(x) > 1)]
    if len(intersect_lists) == 0:  # if there are no real intersections
        return GeometryCollection()  # same result as failed union below
    # the below is a royal pain to follow. what it does is create a dataframe
    # with two columns: 'gs_idx' and 'intersectors'. 'gs_idx' corresponds to
    # a polygon's original index in gs, and 'intersectors' gives a list of
    # gs indices for polygons that intersect with its bbox.
    intersect_lists.name = 'intersectors'
    intersect_lists.index.name = 'gs_idx'
    intersect_lists = intersect_lists.reset_index()
    # first, we get rid  of self-intersection indices in 'intersectors':
    intersect_lists['intersectors'] = intersect_lists.apply(
        lambda x: [i for i in x['intersectors'] if i != x['gs_idx']],
        axis=1)
    # for each row, we next create a union of the polygons in 'intersectors',
    # and find the intersection of that with the polygon at gs[gs_idx]. this
    # (Multi)Polygon output corresponds to all of the intersections for the
    # polygon at gs[gs_idx]. we add that to a list of intersections stored in
    # output_polys.
    output_polys = []
    _ = intersect_lists.apply(lambda x: output_polys.append(
        gs[x['gs_idx']].intersection(cascaded_union(gs[x['intersectors']]))
    ), axis=1)
    # we then generate the union of all of these intersections and return it.
    return cascaded_union(output_polys)


def split_multi_geometries(gdf, obj_id_col=None, group_col=None,
                           geom_col='geometry'):
    """Split apart MultiPolygon or MultiLineString geometries.

    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame` or `str`
        A :class:`geopandas.GeoDataFrame` or path to a geojson containing
        geometries.
    obj_id_col : str, optional
        If one exists, the name of the column that uniquely identifies each
        geometry (e.g. the ``"BuildingId"`` column in many SpaceNet datasets).
        This will be tracked so multiple objects don't get produced with
        the same ID. Note that object ID column will be renumbered on output.
        If passed, `group_col` must also be provided.
    group_col : str, optional
        A column to identify groups for sequential numbering (for example,
        ``'ImageId'`` for sequential number of ``'BuildingId'``). Must be
        provided if `obj_id_col` is passed.
    geom_col : str, optional
        The name of the column in `gdf` that corresponds to geometry. Defaults
        to ``'geometry'``.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        A `geopandas.GeoDataFrame` that's identical to the input, except with
        the multipolygons split into separate rows, and the object ID column
        renumbered (if one exists).

    """
    if obj_id_col and not group_col:
        raise ValueError('group_col must be provided if obj_id_col is used.')
    gdf2 = _check_gdf_load(gdf)
    # drop duplicate columns (happens if loading a csv with geopandas)
    gdf2 = gdf2.loc[:, ~gdf2.columns.duplicated()]
    if len(gdf2) == 0:
        return gdf2
    # check if the values in gdf2[geometry] are polygons; if strings, do loads
    if isinstance(gdf2[geom_col].iloc[0], str):
        gdf2[geom_col] = gdf2[geom_col].apply(loads)
    split_geoms_gdf = pd.concat(
        gdf2.apply(_split_multigeom_row, axis=1, geom_col=geom_col).tolist())
    gdf2 = gdf2.drop(index=split_geoms_gdf.index.unique())  # remove multipolygons
    gdf2 = gpd.GeoDataFrame(pd.concat([gdf2, split_geoms_gdf],
                                      ignore_index=True), crs=gdf2.crs)

    if obj_id_col:
        gdf2[obj_id_col] = gdf2.groupby(group_col).cumcount()+1

    return gdf2


def get_subgraph(G, node_subset):
    """
    Create a subgraph from G. Code almost directly copied from osmnx.

    Arguments
    ---------
    G : :class:`networkx.MultiDiGraph`
        A graph to be subsetted
    node_subset : `list`-like
        The subset of nodes to induce a subgraph of `G`

    Returns
    -------
    G2 : :class:`networkx`.MultiDiGraph
        The subgraph of G that includes node_subset
    """

    node_subset = set(node_subset)

    # copy nodes into new graph
    G2 = G.fresh_copy()
    G2.add_nodes_from((n, G.nodes[n]) for n in node_subset)

    # copy edges to new graph, including parallel edges
    if G2.is_multigraph:
        G2.add_edges_from(
            (n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, keydict in nbrs.items() if nbr in node_subset
            for key, d in keydict.items())
    else:
        G2.add_edges_from(
            (n, nbr, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, d in nbrs.items() if nbr in node_subset)

    # update graph attribute dict, and return graph
    G2.graph.update(G.graph)
    return G2


def _split_multigeom_row(gdf_row, geom_col):
    new_rows = []
    if isinstance(gdf_row[geom_col], MultiPolygon) \
            or isinstance(gdf_row[geom_col], MultiLineString):
        new_polys = _split_multigeom(gdf_row[geom_col])
        for poly in new_polys:
            row_w_poly = gdf_row.copy()
            row_w_poly[geom_col] = poly
            new_rows.append(row_w_poly)
    return pd.DataFrame(new_rows)


def _split_multigeom(multigeom):
    return list(multigeom)


def _reduce_geom_precision(geom, precision=2):
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(np.array(geojson['coordinates']),
                                      precision)
    return shape(geojson)


def latlon_to_utm_epsg(latitude, longitude, return_proj4=False):
    """Get the WGS84 UTM EPSG code based on a latitude and longitude value.

    Arguments
    ---------
    latitude : numeric
        The latitude value for the coordinate.
    longitude : numeric
        The longitude value for the coordinate.
    return_proj4 : bool, optional
        Should the proj4 string be returned as well as the EPSG code? Defaults
        to no (``False``)`

    Returns
    -------
    epsg : int
        The integer corresponding to the EPSG code for the relevant UTM zone
        in WGS 84.
    proj4 : str
        The proj4 string for the CRS. Only returned if ``return_proj4=True``.
    """
    zone_number, zone_letter = _latlon_to_utm_zone(latitude, longitude)

    if return_proj4:
        if zone_letter == 'N':
            direction_indicator = '+north'
        elif zone_letter == 'S':
            direction_indicator = '+south'
        proj = "+proj=utm +zone={} {}".format(zone_number,
                                              direction_indicator)
        proj += " +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

    if zone_letter == 'N':
        epsg = 32600 + zone_number
    elif zone_letter == 'S':
        epsg = 32700 + zone_number

    return (epsg, proj) if return_proj4 else epsg


def _latlon_to_utm_zone(latitude, longitude, ns_only=True):
    """Convert latitude and longitude to a UTM zone ID.

    This function modified from
    `the python utm library <https://github.com/Turbo87/utm>`_.

    Arguments
    ---------
    latitude : numeric or :class:`numpy.ndarray`
        The latitude value of a coordinate.
    longitude : numeric or :class:`numpy.ndarray`
        The longitude value of a coordinate.
    ns_only : bool, optional
        Should the full list of possible zone numbers be used or just the N/S
        options? Defaults to N/S only (``True``).

    Returns
    -------
    zone_number : int
        The numeric portion of the UTM zone ID.
    zone_letter : str
        The string portion of the UTM zone ID. Note that by default this
        function uses only the N/S designation rather than the full range of
        possible letters.
    """

    # If the input is a numpy array, just use the first element
    # User responsibility to make sure that all points are in one zone
    if isinstance(latitude, np.ndarray):
        latitude = latitude.flat[0]
    if isinstance(longitude, np.ndarray):
        longitude = longitude.flat[0]

    utm_val = None

    if 56 <= latitude < 64 and 3 <= longitude < 12:
        utm_val = 32

    elif 72 <= latitude <= 84 and longitude >= 0:
        if longitude < 9:
            utm_val = 31
        elif longitude < 21:
            utm_val = 33
        elif longitude < 33:
            utm_val = 35
        elif longitude < 42:
            utm_val = 37

    if latitude < 0:
        zone_letter = "S"
    else:
        zone_letter = "N"

    if not -80 <= latitude <= 84:
        warn('Warning: UTM projections not recommended for '
             'latitude {}'.format(latitude))
    if utm_val is None:
        utm_val = int((longitude + 180) / 6) + 1

    return utm_val, zone_letter


def _get_coords(geom):
    """Get coordinates from various shapely geometry types."""
    if isinstance(geom, Point) or isinstance(geom, LineString):
        return geom.coords.xy
    elif isinstance(geom, Polygon):
        return geom.exterior.coords.xy


def bbox_corners_to_coco(bbox):
    """Convert bbox from ``[minx, miny, maxx, maxy]`` to coco format.

    COCO formats bounding boxes as ``[minx, miny, width, height]``.

    Arguments
    ---------
    bbox : :class:`list`-like of numerics
        A 4-element list of the form ``[minx, miny, maxx, maxy]``.

    Returns
    -------
    coco_bbox : list
        ``[minx, miny, width, height]`` shape.
    """

    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]


def polygon_to_coco(polygon):
    """Convert a geometry to COCO polygon format."""
    if isinstance(polygon, Polygon):
        coords = polygon.exterior.coords.xy
    elif isinstance(polygon, str):  # assume it's WKT
        coords = loads(polygon).exterior.coords.xy
    elif isinstance(polygon, MultiPolygon):
        raise ValueError("You have MultiPolygon types in your label df. Remove, explode, or fix these to be Polygon geometry types.")
    else:
        raise ValueError('polygon must be a shapely geometry or WKT.')
    # zip together x,y pairs
    coords = list(zip(coords[0], coords[1]))
    coords = [item for coordinate in coords for item in coordinate]

    return coords


def split_geom(geometry, tile_size, resolution=None,
               use_projection_units=False, src_img=None):
    """Splits a vector into approximately equal sized tiles.

    Adapted from @lossyrob's Gist__

    .. Gist: https://gist.github.com/lossyrob/7b620e6d2193cb55fbd0bffacf27f7f2

    The more complex the geometry, the slower this will run, but geometrys with
    around 10000 coordinates run in a few seconds time. You can simplify
    geometries with shapely.geometry.Polygon.simplify if necessary.

    Arguments
    ---------
    geometry : str, optional
        A shapely.geometry.Polygon, path to a single feature geojson,
        or list-like bounding box shaped like [left, bottom, right, top].
        The geometry must be in the projection coordinates corresponding to
        the resolution units.
    tile_size : `tuple` of `int`s
        The size of the input tiles in ``(y, x)`` coordinates. By default,
        this is in pixel units; this can be changed to metric units using the
        `use_metric_size` argument.
    use_projection_units : bool, optional
        Is `tile_size` in pixel units (default) or distance units? To set to distance units
        use ``use_projection_units=True``. If False, resolution must be supplied.
    resolution: `tuple` of `float`s, optional
        (x resolution, y resolution). Used by default if use_metric_size is False.
        Can be acquired from rasterio dataset object's metadata.
    src_img:  `str` or `raster`, optional
        A rasterio raster object or path to a geotiff. The bounds of this raster and the geometry will be
        intersected and the result of the intersection will be tiled. Useful in cases where the extent of
        collected labels and source imagery partially overlap. The src_img must have the same projection units
        as the geometry.

    Returns
    -------
    tile_bounds : list (containing sublists like [left, bottom, right, top])

    """
    if isinstance(geometry, str):
        gj = json.loads(open(geometry).read())

        features = gj['features']
        if not len(features) == 1:
            print('Feature collection must only contain one feature')
            sys.exit(1)

        geometry = shape(features[0]['geometry'])

    elif isinstance(geometry, list) or isinstance(geometry, np.ndarray):
        assert len(geometry) == 4
        geometry = box(*geometry)

    if use_projection_units is False:
        if resolution is None:
            print("Resolution must be specified if use_projection_units is"
                  " False. Access it from src raster meta.")
            return
        # convert pixel units to CRS units to use during image tiling.
        # NOTE: This will be imperfect for large AOIs where there isn't
        # a constant relationship between the src CRS units and src pixel
        # units.
        if isinstance(resolution, (float, int)):
            resolution = (resolution, resolution)
        tmp_tile_size = [tile_size[0]*resolution[0],
                         tile_size[1]*resolution[1]]
    else:
        tmp_tile_size = tile_size
        
    if src_img is not None:
        src_img = _check_rasterio_im_load(src_img)
        geometry = geometry.intersection(box(*src_img.bounds))
        bounds = geometry.bounds
    else:
        bounds = geometry.bounds
        
    xmin = bounds[0]
    xmax = bounds[2]
    ymin = bounds[1]
    ymax = bounds[3]
    x_extent = xmax - xmin
    y_extent = ymax - ymin
    x_steps = np.ceil(x_extent/tmp_tile_size[1])
    y_steps = np.ceil(y_extent/tmp_tile_size[0])
    x_mins = np.arange(xmin, xmin + tmp_tile_size[1]*x_steps,
                       tmp_tile_size[1])
    y_mins = np.arange(ymin, ymin + tmp_tile_size[0]*y_steps,
                       tmp_tile_size[0])
    tile_bounds = [
        (i, j, i+tmp_tile_size[1], j+tmp_tile_size[0])
        for i in x_mins for j in y_mins if not geometry.intersection(
            box(*(i, j, i+tmp_tile_size[1], j+tmp_tile_size[0]))).is_empty
        ]
    return tile_bounds
