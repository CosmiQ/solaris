from .core import _check_gdf_load
import numpy as np
import pandas as pd
from affine import Affine
import geopandas as gpd
import os
import rasterio
from rasterio.enums import Resampling
import ogr
import shapely
from shapely.errors import WKTReadingError
from shapely.wkt import loads
from shapely.geometry import MultiLineString, MultiPolygon, mapping, shape
from shapely.ops import cascaded_union
from warnings import warn


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
    # check if the values in gdf2[geometry] are polygons; if strings, do loads
    if isinstance(gdf2[geom_col][0], str):
        gdf2[geom_col] = gdf2[geom_col].apply(loads)
    split_geoms_gdf = pd.concat(
        gdf2.apply(_split_multigeom_row, axis=1, geom_col=geom_col).tolist())
    gdf2.drop(index=split_geoms_gdf.index.unique())  # remove multipolygons
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


def _check_wkt_load(x):
    """Check if an object is a loaded polygon or not. If not, load it."""
    if isinstance(x, str):
        try:
            x = loads(x)
        except WKTReadingError:
            warn('{} is not a WKT-formatted string.'.format(x))

    return x



# PRETEND THIS ISN'T HERE AT THE MOMENT
# class CoordTransformer(object):
#     """A transformer class to change coordinate space using affine transforms.
#
#     Notes
#     -----
#     This class will take in an image or geometric object (Shapely or GDAL)
#     and transform its coordinate space based on `dest_obj` . `dest_obj`
#     should be an instance of :class:`rasterio.DatasetReader` .
#
#     Arguments
#     ---------
#     src_obj
#         A source image or geometric object to transform. The function will
#         first try to extract georegistration information from this object
#         if it exists; if it doesn't, it will assume unit (pixel) coords.
#     dest_obj
#         Object with a destination coordinate reference system to apply to
#         `src_obj` . This can be in the form of an ``[a, b, d, e, xoff, yoff]``
#         `list` , an :class:`affine.Affine` instance, or a source
#         :class:`geopandas.GeoDataFrame` or geotiff with `crs` metadata to
#         produce the transform from, or even just a crs string.
#     src_crs : optional
#         Source coordinate reference in the form of a :class:`rasterio.crs.CRS`
#         object or an epsg string. Only needed if the source object provided
#         does not have CRS metadata attached to it.
#     src_transform : :class:`affine.Affine` or :class:`list`
#         The source affine transformation matrix as a :class:`affine.Affine`
#         object or in an ``[a, b, c, d, xoff, yoff]`` `list`. Required if
#         `src_obj` is a :class:`numpy.array` .
#     dest_transform : :class:`affine.Affine` or :class:`list`
#         The destination affine transformation matrix as a
#         :class:`affine.Affine` object or in an ``[a, b, c, d, xoff, yoff]``
#         `list` . Required if `dest_obj` is a :class:`numpy.array` .
#     """
#     def __init__(self, src_obj=None, dest_obj=None, src_crs=None,
#                  src_transform=None, dest_transform=None):
#         self.src_obj = src_obj
#         self.src_type = None
#         self.dest_obj = dest_obj
#         self.dest_type = None
#         self.get_obj_types()  # replaces the None values above
#         self.src_crs = src_crs
#         if isinstance(self.src_crs, dict):
#             self.src_crs = self.src_crs['init']
#         if not self.src_crs:
#             self.src_crs = self._get_crs(self.src_obj, self.src_type)
#         self.dest_crs = self._get_crs(self.dest_obj, self.dest_type)
#         self.src_transform = src_transform
#         self.dest_transform = dest_transform
#
#     def __repr__(self):
#         print('CoordTransformer for {}'.format(self.src_obj))
#
#     def load_src_obj(self, src_obj, src_crs=None):
#         """Load in a new source object for transformation."""
#         self.src_obj = src_obj
#         self.src_type = None  # replaced in self._get_src_crs()
#         self.src_type = self._get_type(self.src_obj)
#         self.src_crs = src_crs
#         if self.src_crs is None:
#             self.src_crs = self._get_crs(self.src_obj, self.src_type)
#
#     def load_dest_obj(self, dest_obj):
#         """Load in a new destination object for transformation."""
#         self.dest_obj = dest_obj
#         self.dest_type = None
#         self.dest_type = self._get_type(self.dest_obj)
#         self.dest_crs = self._get_crs(self.dest_obj, self.dest_type)
#
#     def load_src_crs(self, src_crs):
#         """Load in a new source coordinate reference system."""
#         self.src_crs = self._get_crs(src_crs)
#
#     def get_obj_types(self):
#         if self.src_obj is not None:
#             self.src_type = self._get_type(self.src_obj)
#             if self.src_type is None:
#                 warn('The src_obj type is not compatible with this package.')
#         if self.dest_obj is not None:
#             self.dest_type = self._get_type(self.dest_obj)
#             if self.dest_type is None:
#                 warn('The dest_obj type is not compatible with this package.')
#             elif self.dest_type == 'shapely Geometry':
#                 warn('Shapely geometries cannot provide a destination CRS.')
#
#     @staticmethod
#     def _get_crs(obj, obj_type):
#         """Get the destination coordinate reference system."""
#         # get the affine transformation out of dest_obj
#         if obj_type == "transform matrix":
#             return Affine(obj)
#         elif obj_type == 'Affine':
#             return obj
#         elif obj_type == 'GeoTIFF':
#             return rasterio.open(obj).crs
#         elif obj_type == 'GeoDataFrame':
#             if isinstance(obj, str):  # if it's a path to a gdf
#                 return gpd.read_file(obj).crs
#             else:  # assume it's a GeoDataFrame object
#                 return obj.crs
#         elif obj_type == 'epsg string':
#             if obj.startswith('{init'):
#                 return rasterio.crs.CRS.from_string(
#                     obj.lstrip('{init: ').rstrip('}'))
#             elif obj.lower().startswith('epsg'):
#                 return rasterio.crs.CRS.from_string(obj)
#         elif obj_type == 'OGR Geometry':
#             return get_crs_from_ogr(obj)
#         elif obj_type == 'shapely Geometry':
#             raise TypeError('Cannot extract a coordinate system from a ' +
#                             'shapely.Geometry')
#         else:
#             raise TypeError('Cannot extract CRS from this object type.')
#
#     @staticmethod
#     def _get_type(obj):
#         if isinstance(obj, gpd.GeoDataFrame):
#             return 'GeoDataFrame'
#         elif isinstance(obj, str):
#             if os.path.isfile(obj):
#                 if os.path.splitext(obj)[1].lower() in ['tif', 'tiff',
#                                                         'geotiff']:
#                     return 'GeoTIFF'
#                 elif os.path.splitext(obj)[1] in ['csv', 'geojson']:
#                     # assume it can be loaded as a geodataframe
#                     return 'GeoDataFrame'
#             else:  # assume it's a crs string
#                 if obj.startswith('{init'):
#                     return "epsg string"
#                 elif obj.lower().startswith('epsg'):
#                     return "epsg string"
#                 else:
#                     raise ValueError('{} is not an accepted crs type.'.format(
#                         obj))
#         elif isinstance(obj, ogr.Geometry):
#             # ugh. Try to get the EPSG code out.
#             return 'OGR Geometry'
#         elif isinstance(obj, shapely.Geometry):
#             return "shapely Geometry"
#         elif isinstance(obj, list):
#             return "transform matrix"
#         elif isinstance(obj, Affine):
#             return "Affine transform"
#         elif isinstance(obj, np.array):
#             return "numpy array"
#         else:
#             return None
#
#     def transform(self, output_loc):
#         """Transform `src_obj` from `src_crs` to `dest_crs`.
#
#         Arguments
#         ---------
#         output_loc : `str` or `var`
#             Object or location to output transformed src_obj to. If it's a
#             string, it's assumed to be a path.
#         """
#         if not self.src_crs or not self.dest_crs:
#             raise AttributeError('The source or destination CRS is missing.')
#         if not self.src_obj:
#             raise AttributeError('The source object to transform is missing.')
#         if isinstance(output_loc, str):
#             out_file = True
#         if self.src_type == 'GeoTIFF':
#             return rasterio.warp.reproject(rasterio.open(self.src_obj),
#                                            output_loc,
#                                            src_transform=self.src_transform,
#                                            src_crs=self.src_crs,
#                                            dst_trasnform=self.dest_transform,
#                                            dst_crs=self.dest_crs,
#                                            resampling=Resampling.bilinear)
#         elif self.src_type == 'GeoDataFrame':
#             if isinstance(self.src_obj, str):
#                 # load the gdf and transform it
#                 tmp_src = gpd.read_file(self.src_obj).to_crs(self.dest_crs)
#             else:
#                 # just transform it
#                 tmp_src = self.src_obj.to_crs(self.dest_crs)
#             if out_file:
#                 # save to file
#                 if output_loc.lower().endswith('json'):
#                     tmp_src.to_file(output_loc, driver="GeoJSON")
#                 else:
#                     tmp_src.to_file(output_loc)  # ESRI shapefile
#                 return
#             else:
#                 # assign to the variable and return
#                 output_loc = tmp_src
#                 return output_loc
#         elif self.src_type == 'OGR Geometry':
#             dest_sr = ogr.SpatialReference().ImportFromEPSG(
#                 int(self.dest_crs.lstrip('epsg')))
#             output_loc = self.src_obj.TransformTo(dest_sr)
#             return output_loc
#         elif self.src_type == 'shapely Geometry':
#             if self.dest_type not in [
#                     'Affine transform', 'transform matrix'
#                     ] and not self.dest_transform:
#                 raise ValueError('Transforming shapely objects requires ' +
#                                  'an affine transformation matrix.')
#             elif self.dest_type == 'Affine transform':
#                 output_loc = shapely.affinity.affine_transform(
#                     self.src_obj, [self.dest_obj.a, self.dest_obj.b,
#                                    self.dest_obj.d, self.dest_obj.e,
#                                    self.dest_obj.xoff, self.dest_obj.yoff]
#                 )
#                 return output_loc
#             elif self.dest_type == 'transform matrix':
#                 output_loc = shapely.affinity.affine_transform(self.src_obj,
#                                                                self.dest_obj)
#                 return output_loc
#             else:
#                 if isinstance(self.dest_transform, Affine):
#                     xform_mat = [self.dest_transform.a, self.dest_transform.b,
#                                  self.dest_transform.d, self.dest_transform.e,
#                                  self.dest_transform.xoff,
#                                  self.dest_transform.yoff]
#                 else:
#                     xform_mat = self.dest_transform
#                 output_loc = shapely.affinity.affine_transform(self.src_obj,
#                                                                xform_mat)
#                 return output_loc
#         elif self.src_type == 'numpy array':
#             return rasterio.warp.reproject(
#                 self.src_obj, output_loc, src_transform=self.src_transform,
#                 src_crs=self.src_crs, dst_transform=self.dest_transform,
#                 dst_crs=self.dest_crs)
#
#
# def get_crs_from_ogr(annoying_OGR_geometry):
#     """Get a CRS from an :class:`osgeo.ogr.Geometry` object.
#
#     Arguments
#     ---------
#     annoying_OGR_geometry: :class:`osgeo.ogr.Geometry`
#         An OGR object which stores crs information in an annoying fashion.
#
#     Returns
#     -------
#     An extremely clear, easy to work with ``'epsg[number]'`` string.
#     """
#     srs = annoying_OGR_geometry.GetSpatialReference()
#     result_of_ID = srs.AutoIdentifyEPSG()  # if success, returns 0
#     if result_of_ID == 0:
#         return 'epsg:' + str(srs.GetAuthorityCode(None))
#     else:
#         raise ValueError('Could not determine EPSG code.')
