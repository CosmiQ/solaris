import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from affine import Affine
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio import transform
from rio_tiler.utils import get_vrt_transform, has_alpha_band
from rio_tiler.utils import _requested_tile_aligned_with_internal_tile


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


def read_cog_tile(src,
                  bounds,
                  tile_size,
                  indexes=None,
                  nodata=None,
                  resampling_method="bilinear",
                  tile_edge_padding=2):
    """
    Read cloud-optimized geotiff tile.

    Notes
    -----
    Modified from `rio-tiler <https://github.com/cogeotiff/rio-tiler>`_.
    License included below per terms of use.

        BSD 3-Clause License
        (c) 2017 Mapbox
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this
          list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright notice,
          this list of conditions and the following disclaimer in the documentation
          and/or other materials provided with the distribution.

        * Neither the name of the copyright holder nor the names of its
          contributors may be used to endorse or promote products derived from
          this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Arguments
    ---------
    src : rasterio.io.DatasetReader
        rasterio.io.DatasetReader object
    bounds : list
        Tile bounds (left, bottom, right, top)
    tile_size : list
        Output image size
    indexes : list of ints or a single int, optional, (defaults: None)
        If `indexes` is a list, the result is a 3D array, but is
        a 2D array if it is a band index number.
    nodata: int or float, optional (defaults: None)
    resampling_method : str, optional (default: "bilinear")
         Resampling algorithm
    tile_edge_padding : int, optional (default: 2)
        Padding to apply to each edge of the tile when retrieving data
        to assist in reducing resampling artefacts along edges.

    Returns
    -------
    out : array, int
        returns pixel value.
    """
    if isinstance(indexes, int):
        indexes = [indexes]
    elif isinstance(indexes, tuple):
        indexes = list(indexes)

    vrt_params = dict(
        add_alpha=True, crs='epsg:' + str(src.crs.to_epsg()),
        resampling=Resampling[resampling_method]
    )

    vrt_transform, vrt_width, vrt_height = get_vrt_transform(
        src, bounds, bounds_crs='epsg:' + str(src.crs.to_epsg()))
    out_window = Window(col_off=0, row_off=0,
                        width=vrt_width, height=vrt_height)

    if tile_edge_padding > 0 and not \
            _requested_tile_aligned_with_internal_tile(src, bounds, tile_size):
        vrt_transform = vrt_transform * Affine.translation(
            -tile_edge_padding, -tile_edge_padding
        )
        orig__vrt_height = vrt_height
        orig_vrt_width = vrt_width
        vrt_height = vrt_height + 2 * tile_edge_padding
        vrt_width = vrt_width + 2 * tile_edge_padding
        out_window = Window(
            col_off=tile_edge_padding,
            row_off=tile_edge_padding,
            width=orig_vrt_width,
            height=orig__vrt_height,
        )

    vrt_params.update(dict(transform=vrt_transform,
                           width=vrt_width,
                           height=vrt_height))

    indexes = indexes if indexes is not None else src.indexes
    out_shape = (len(indexes), tile_size[1], tile_size[0])

    nodata = nodata if nodata is not None else src.nodata
    if nodata is not None:
        vrt_params.update(dict(nodata=nodata,
                               add_alpha=False,
                               src_nodata=nodata))

    if has_alpha_band(src):
        vrt_params.update(dict(add_alpha=False))

    with WarpedVRT(src, **vrt_params) as vrt:
        data = vrt.read(
            out_shape=out_shape,
            indexes=indexes,
            window=out_window,
            resampling=Resampling[resampling_method],
        )
        mask = vrt.dataset_mask(out_shape=(tile_size[1], tile_size[0]),
                                window=out_window)

        return data, mask, out_window, vrt_transform
