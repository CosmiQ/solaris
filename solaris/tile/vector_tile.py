import os
import numpy as np
from pathlib import Path
from shapely.geometry import box, Polygon
import geopandas as gpd
from ..utils.core import _check_gdf_load, _check_crs
from ..utils.tile import save_empty_geojson
from ..utils.geo import get_projection_unit, split_multi_geometries
from ..utils.geo import reproject_geometry
from tqdm.auto import tqdm


class VectorTiler(object):
    """An object to tile geospatial vector data into smaller pieces.

    Arguments
    ---------


    Attributes
    ----------
    """

    def __init__(self, dest_dir=None, dest_crs=None, output_format='GeoJSON',
                 verbose=False, super_verbose=False):
        if verbose or super_verbose:
            print('Preparing the tiler...')
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(exist_ok=True)
        if dest_crs is not None:
            self.dest_crs = _check_crs(dest_crs)
        self.output_format = output_format
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.tile_paths = [] # retains the paths of the last call to .tile()
        if self.verbose or self.super_verbose:
            print('Initialization done.')

    def tile(self, src, tile_bounds, tile_bounds_crs=None, geom_type='Polygon',
             split_multi_geoms=True, min_partial_perc=0.0,
             dest_fname_base='geoms', obj_id_col=None,
             output_ext='.geojson'):
        """Tile `src` into vector data tiles bounded by `tile_bounds`.

        Arguments
        ---------
        src : `str`, :class:`pathlib.Path` or :class:`geopandas.GeoDataFrame`
            The source vector data to tile. Must either be a path to a GeoJSON
            or a :class:`geopandas.GeoDataFrame`.
        tile_bounds : list
            A :class:`list` made up of ``[left, top, right, bottom] `` sublists
            (this can be extracted from
            :class:`solaris.tile.raster_tile.RasterTiler` after tiling imagery)
        tile_bounds_crs : int, optional
            The EPSG code or rasterio.crs.CRS object for the CRS that the tile
            bounds are in. RasterTiler.tile returns the CRS of the raster tiles
            and can be used here. If not provided, it's assumed that the CRS is the
            same as in `src`. This argument must be provided if the bound
            coordinates and `src` are not in the same CRS, otherwise tiling will
            not occur correctly.
        geom_type : str, optional (default: "Polygon")
            The type of geometries contained within `src`. Defaults to
            ``"Polygon"``, can also be ``"LineString"``.
        split_multi_geoms : bool, optional (default: True)
            Should multi-polygons or multi-linestrings generated by clipping
            a geometry into discontinuous pieces be separated? Defaults to yes
            (``True``).
        min_partial_perc : float, optional (default: 0.0)
            The minimum percentage of a :class:`shapely.geometry.Polygon` 's
            area or :class:`shapely.geometry.LineString` 's length that must
            be retained within a tile's bounds to be included in the output.
            Defaults to ``0.0``, meaning that the contained portion of a
            clipped geometry will be included, no matter how small.
        dest_fname_base : str, optional (default: 'geoms')
            The base filename to use when creating outputs. The lower left
            corner coordinates of the tile's bounding box will be appended
            when saving.
        obj_id_col : str, optional (default: None)
            If ``split_multi_geoms=True``, the name of a column that specifies
            a unique identifier for each geometry (e.g. the ``"BuildingId"``
            column in many SpaceNet datasets.) See
            :func:`solaris.utils.geo.split_multi_geometries` for more.
        output_ext : str, optional, (default: geojson)
            Extension of output files, can be 'geojson' or 'json'.
        """

        if isinstance(src, gpd.GeoDataFrame) and src.crs is None:
            raise ValueError("If the src input is a geopandas.GeoDataFrame, it must have a crs attribute.")

        tile_gen = self.tile_generator(src, tile_bounds, tile_bounds_crs,
                                       geom_type, split_multi_geoms,
                                       min_partial_perc,
                                       obj_id_col=obj_id_col)
        self.tile_paths = []
        for tile_gdf, tb in tqdm(tile_gen):
            if self.proj_unit not in ['meter', 'metre']:
                dest_path = os.path.join(
                    self.dest_dir, '{}_{}_{}{}'.format(dest_fname_base,
                                                       np.round(tb[0], 3),
                                                       np.round(tb[3], 3),
                                                       output_ext))
            else:
                dest_path = os.path.join(
                    self.dest_dir, '{}_{}_{}{}'.format(dest_fname_base,
                                                       int(tb[0]),
                                                       int(tb[3]),
                                                       output_ext))
            self.tile_paths.append(dest_path)
            if len(tile_gdf) > 0:
                tile_gdf.to_file(dest_path, driver='GeoJSON')
            else:
                save_empty_geojson(dest_path, self.dest_crs)

    def tile_generator(self, src, tile_bounds, tile_bounds_crs=None,
                       geom_type='Polygon', split_multi_geoms=True,
                       min_partial_perc=0.0, obj_id_col=None):
        """Generate `src` vector data tiles bounded by `tile_bounds`.

        Arguments
        ---------
        src : `str` :class:`pathlib.Path` or :class:`geopandas.GeoDataFrame`
            The source vector data to tile. Must either be a path to a GeoJSON
            or a :class:`geopandas.GeoDataFrame`.
        tile_bounds : list
            A :class:`list` made up of ``[left, top, right, bottom] `` sublists
            (this can be extracted from
            :class:`solaris.tile.raster_tile.RasterTiler` after tiling imagery)
        tile_bounds_crs : int, optional
            The EPSG code for the CRS that the tile bounds are in. If not
            provided, it's assumed that the CRS is the same as in `src`. This
            argument must be provided if the bound coordinates and `src` are
            not in the same CRS, otherwise tiling will not occur correctly.
        geom_type : str, optional (default: "Polygon")
            The type of geometries contained within `src`. Defaults to
            ``"Polygon"``, can also be ``"LineString"``.
        split_multi_geoms : bool, optional (default: True)
            Should multi-polygons or multi-linestrings generated by clipping
            a geometry into discontinuous pieces be separated? Defaults to yes
            (``True``).
        min_partial_perc : float, optional (default: 0.0)
            The minimum percentage of a :class:`shapely.geometry.Polygon` 's
            area or :class:`shapely.geometry.LineString` 's length that must
            be retained within a tile's bounds to be included in the output.
            Defaults to ``0.0``, meaning that the contained portion of a
            clipped geometry will be included, no matter how small.
        obj_id_col : str, optional (default: None)
            If ``split_multi_geoms=True``, the name of a column that specifies
            a unique identifier for each geometry (e.g. the ``"BuildingId"``
            column in many SpaceNet datasets.) See
            :func:`solaris.utils.geo.split_multi_geometries` for more.

        Yields
        ------
        tile_gdf : :class:`geopandas.GeoDataFrame`
            A tile geodataframe.
        tb : list
            A list with ``[left, top, right, bottom] `` coordinates for the
            boundaries contained by `tile_gdf`.
        """
        self.src = _check_gdf_load(src)
        if self.verbose:
            print("Num tiles:", len(tile_bounds))

        self.src_crs = _check_crs(self.src.crs)
        # check if the tile bounds and vector are in the same crs
        if tile_bounds_crs is not None:
            tile_bounds_crs = _check_crs(tile_bounds_crs)
        else:
            tile_bounds_crs = self.src_crs
        if self.src_crs != tile_bounds_crs:
            reproject_bounds = True  # used to transform tb for clip_gdf()
        else:
            reproject_bounds = False

        self.proj_unit = get_projection_unit(self.src_crs)
        if getattr(self, 'dest_crs', None) is None:
            self.dest_crs = self.src_crs
        for i, tb in enumerate(tile_bounds):
            if self.super_verbose:
                print("\n", i, "/", len(tile_bounds))
            if reproject_bounds:
                tile_gdf = clip_gdf(self.src,
                                    reproject_geometry(box(*tb),
                                                       tile_bounds_crs,
                                                       self.src_crs),
                                    min_partial_perc,
                                    geom_type, verbose=self.super_verbose)
            else:
                tile_gdf = clip_gdf(self.src, tb, min_partial_perc, geom_type,
                                    verbose=self.super_verbose)
            if self.src_crs != self.dest_crs:
                tile_gdf = tile_gdf.to_crs(crs=self.dest_crs.to_wkt())
            if split_multi_geoms:
                split_multi_geometries(tile_gdf, obj_id_col=obj_id_col)
            yield tile_gdf, tb


def search_gdf_polygon(gdf, tile_polygon):
    """Find polygons in a GeoDataFrame that overlap with `tile_polygon` .

    Arguments
    ---------
    gdf : :py:class:`geopandas.GeoDataFrame`
        A :py:class:`geopandas.GeoDataFrame` of polygons to search.
    tile_polygon : :py:class:`shapely.geometry.Polygon`
        A :py:class:`shapely.geometry.Polygon` denoting a tile's bounds.

    Returns
    -------
    precise_matches : :py:class:`geopandas.GeoDataFrame`
        The subset of `gdf` that overlaps with `tile_polygon` . If
        there are no overlaps, this will return an empty
        :py:class:`geopandas.GeoDataFrame`.

    """
    sindex = gdf.sindex
    possible_matches_index = list(sindex.intersection(tile_polygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[
        possible_matches.intersects(tile_polygon)
        ]
    if precise_matches.empty:
        precise_matches = gpd.GeoDataFrame(geometry=[])
    return precise_matches


def clip_gdf(gdf, tile_bounds, min_partial_perc=0.0, geom_type="Polygon",
             use_sindex=True, verbose=False):
    """Clip GDF to a provided polygon.

    Clips objects within `gdf` to the region defined by
    `poly_to_cut`. Also adds several columns to the output::

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
    tile_bounds : `list` or :class:`shapely.geometry.Polygon`
        The geometry to clip objects in `gdf` to. This can either be a
        ``[left, top, right, bottom] `` bounds list or a
        :class:`shapely.geometry.Polygon` object defining the area to keep.
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
    verbose : bool, optional
        Switch to print relevant values.

    Returns
    -------
    cut_gdf : :py:class:`geopandas.GeoDataFrame`
        `gdf` with all contained objects clipped to `poly_to_cut` .
        See notes above for details on additional clipping columns added.

    """
    if isinstance(tile_bounds, tuple):
        tb = box(*tile_bounds)
    elif isinstance(tile_bounds, list):
        tb = box(*tile_bounds)
    elif isinstance(tile_bounds, Polygon):
        tb = tile_bounds
    if use_sindex and (geom_type == "Polygon"):
        gdf = search_gdf_polygon(gdf, tb)

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

    cut_gdf = gdf.copy()
    cut_gdf.geometry = gdf.intersection(tb)

    if geom_type == 'Polygon':
        cut_gdf['partialDec'] = cut_gdf.area / cut_gdf['origarea']
        cut_gdf = cut_gdf.loc[cut_gdf['partialDec'] > min_partial_perc, :]
        cut_gdf['truncated'] = (cut_gdf['partialDec'] != 1.0).astype(int)
    else:
        # assume linestrings
        # remove null
        cut_gdf = cut_gdf[cut_gdf['geometry'].notnull()]
        cut_gdf['partialDec'] = 1
        cut_gdf['truncated'] = 0
        # cut_gdf = cut_gdf[cut_gdf.geom_type != "GeometryCollection"]
        if len(cut_gdf) > 0 and verbose:
            print("clip_gdf() - gdf.iloc[0]:", gdf.iloc[0])
            print("clip_gdf() - tb:", tb)
            print("clip_gdf() - gdf_cut:", cut_gdf)

    # TODO: IMPLEMENT TRUNCATION MEASUREMENT FOR LINESTRINGS

    return cut_gdf
