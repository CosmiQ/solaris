import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.io import DatasetReader
import math
from rio_tiler.errors import TileOutsideBounds
from . import utils
import numpy as np


def tile_utm_source(src, ll_x, ll_y, ur_x, ur_y, indexes=None, tilesize=256,
                    nodata=None, alpha=None, dst_crs='epsg:4326'):
    """
    Create a UTM tile from a :py:class:`rasterio.Dataset` in memory.

    Arguments
    ---------
    src : :py:class:`rasterio.Dataset`
        Source imagery dataset to tile.
    ll_x : int or float
        Lower left x position (i.e. Western bound).
    ll_y : int or f
    loat
        Lower left y position (i.e. Southern bound).
    ur_x : int or float
        Upper right x position (i.e. Eastern bound).
    ur_y : int or float
        Upper right y position (i.e. Northern bound).
    indexes : tuple of 3 ints, optional
        Band indexes for the output. By default, extracts all of the
        indexes from `src`.
    tilesize : int, optional
        Output image X and Y pixel extent. Defaults to ``256``.
    nodata : int or float, optional
        Value to use for `nodata` pixels during tiling. By default, uses
        the existing `nodata` value in `src`.
    alpha : int, optional
        Alpha band index for tiling. By default, uses the same band as
        specified by `src`.
    dst_crs : str, optional
        Coordinate reference system for output. Defaults to ``"epsg:4326"``.

    Returns
    -------
    ``(data, mask, window, window_transform)`` tuple.
        data : :py:class:`numpy.ndarray`
            int pixel values. Shape is ``(C, Y, X)`` if retrieving multiple
            channels, ``(Y, X)`` otherwise.
        mask : :py:class:`numpy.ndarray`
            int mask indicating which pixels contain information and which are
            `nodata`. Pixels containing data have value ``255``, `nodata`
            pixels have value ``0``.
        window : :py:class:`rasterio.windows.Window`
            :py:class:`rasterio.windows.Window` object indicating the raster
            location of the dataset subregion being returned in `data`.
        window_transform : :py:class:`affine.Affine`
            Affine transformation for the window.

    """

    wgs_bounds = transform_bounds(
        *[src.crs, dst_crs] + list(src.bounds), densify_pts=21)

    indexes = indexes if indexes is not None else src.indexes
    tile_bounds = (ll_x, ll_y, ur_x, ur_y)
    if not utils.tile_exists_utm(wgs_bounds, tile_bounds):
        raise TileOutsideBounds(
            'Tile {}/{}/{}/{} is outside image bounds'.format(tile_bounds))

    return utils.tile_read_utm(src, tile_bounds, tilesize, indexes=indexes,
                               nodata=nodata, alpha=alpha, dst_crs=dst_crs)


def tile_utm(source, ll_x, ll_y, ur_x, ur_y, indexes=None, tilesize=256,
             nodata=None, alpha=None, dst_crs='epsg:4326'):
    """
    Create a UTM tile from a file or a :py:class:`rasterio.Dataset` in memory.

    This function is a wrapper around :func:`tile_utm_source` to enable passing
    of file paths instead of pre-loaded :py:class:`rasterio.Dataset` s.

    Arguments
    ---------
    source : :py:class:`rasterio.Dataset`
        Source imagery dataset to tile.
    ll_x : int or float
        Lower left x position (i.e. Western bound).
    ll_y : int or float
        Lower left y position (i.e. Southern bound).
    ur_x : int or float
        Upper right x position (i.e. Eastern bound).
    ur_y : int or float
        Upper right y position (i.e. Northern bound).
    indexes : tuple of 3 ints, optional
        Band indexes for the output. By default, extracts all of the
        indexes from `source` .
    tilesize : :obj:`int`, optional
        Output image X and Y pixel extent. Defaults to ``256``.
    nodata : int or float, optional
        Value to use for ``nodata`` pixels during tiling. By default, uses
        the existing ``nodata`` value in `src`.
    alpha : :obj:`int`, optional
        Alpha band index for tiling. By default, uses the same band as
        specified by `src`.
    dst_crs : str, optional
        Coordinate reference system for output. Defaults to ``"epsg:4326"``.

    Returns
    -------
    ``(data, mask, window, window_transform`` tuple.
        data : :py:class:`numpy.ndarray`
            int pixel values. Shape is ``(C, Y, X)`` if retrieving multiple
            channels, ``(Y, X)`` otherwise.
        mask : :class:`numpy.ndarray`
            int mask indicating which pixels contain information and which are
            `nodata`. Pixels containing data have value ``255``, `nodata` pixels
            have value ``0``.
        window : :py:class:`rasterio.windows.Window`
            :py:class:`rasterio.windows.Window` object indicating the raster
            location of the dataset subregion being returned in `data`.
        window_transform : :py:class:`affine.Affine`
            Affine transformation for the window.

    """

    if isinstance(source, DatasetReader):
        src = source
    elif os.path.exists(source):
        src = rasterio.open(source)  # read in the file
    else:
        raise ValueError('Source is not a rasterio.Dataset or a valid path.')

    return tile_utm_source(src, ll_x, ll_y, ur_x, ur_y, indexes=indexes,
                           tilesize=tilesize, nodata=nodata, alpha=alpha,
                           dst_crs=dst_crs)


def get_chip(source, ll_x, ll_y, gsd,
             utm_crs='',
             indexes=None,
             tilesize=256,
             nodata=None,
             alpha=None):
    """Get an image tile of specific pixel size.

    This wrapper function permits passing of `ll_x`, `ll_y`, `gsd`, and
    `tile_size_pixels` in place of boundary coordinates to extract an image
    region of defined pixel extent.

    Arguments
    ---------
    source : :py:class:`rasterio.Dataset`
        Source imagery dataset to tile.
    ll_x : int or float
        Lower left x position (i.e. Western bound).
    ll_y : int or float
        Lower left y position (i.e. Southern bound).
    gsd : float
        Ground sample distance of the source imagery in meter/pixel units.
    utm_crs : :py:class:`rasterio.crs.CRS`, optional
        UTM coordinate reference system string for the imagery. If not
        provided, this is calculated using
        :func:`cw_tiler.utils.get_wgs84_bounds` and
        :func:`cw_tiler.utils.calculate_UTM_crs` .
    indexes : tuple of 3 ints, optional
        Band indexes for the output. By default, extracts all of the
        indexes from `source`.
    tilesize : int, optional
        Output image X and Y pixel extent. Defaults to ``256`` .
    nodata : int or float, optional
        Value to use for `nodata` pixels during tiling. By default, uses
        the existing `nodata` value in `source`.
    alpha : int, optional
        Alpha band index for tiling. By default, uses the same band as
        specified by `source`.

    Returns
    -------
    ``(data, mask, window, window_transform`` tuple.
        data : :class:`numpy.ndarray`
            int pixel values. Shape is ``(C, Y, X)`` if retrieving multiple
            channels, ``(Y, X)`` otherwise.
        mask : :class:`numpy.ndarray`
            int mask indicating which pixels contain information and which are
            `nodata`. Pixels containing data have value ``255``, `nodata` pixels
            have value ``0``.
        window : :py:class:`rasterio.windows.Window`
            :py:class:`rasterio.windows.Window` object indicating the raster
            location of the dataset subregion being returned in `data`.
        window_transform : :py:class:`affine.Affine`
            Affine transformation for the window.
    """

    ur_x = ll_x + gsd * tilesize
    ur_y = ll_y + gsd * tilesize

    if isinstance(source, DatasetReader):
        src = source
    else:
        src = rasterio.open(source)

    if not utm_crs:
        wgs_bounds = utils.get_wgs84_bounds(src)
        utm_crs = utils.calculate_UTM_crs(wgs_bounds)

    return tile_utm(source, ll_x, ll_y, ur_x, ur_y, indexes=indexes,
                    tilesize=tilesize, nodata=nodata, alpha=alpha,
                    dst_crs=utm_crs)


def calculate_anchor_points(utm_bounds, stride_size_meters=400, extend=False,
                            quad_space=False):
    """Get anchor point (lower left corner of bbox) for chips from a tile.

    Arguments
    ---------
    utm_bounds : tuple of 4 floats
        A :obj:`tuple` of shape ``(min_x, min_y, max_x, max_y)`` that defines
        the spatial extent of the tile to be split. Coordinates should be in
        UTM.
    stride_size_meters : int, optional
        Stride size in both X and Y directions for generating chips. Defaults
        to ``400``.
    extend : bool, optional
        Defines whether UTM boundaries should be rounded to the nearest integer
        outward from `utm_bounds` (`extend` == ``True``) or
        inward from `utm_bounds` (`extend` == ``False``). Defaults
        to ``False`` (inward).
    quad_space : bool, optional
        If tiles will overlap by no more than half their X and/or Y extent in
        each direction, `quad_space` can be used to split chip
        anchors into four non-overlapping subsets. For example, if anchor
        points are 400m apart and each chip will be 800m by 800m, `quad_space`
        will generate four sets which do not internally overlap;
        however, this would fail if tiles are 900m by 900m. Defaults to
        ``False``, in which case the returned ``anchor_point_list_dict`` will
        comprise a single list of anchor points.

    Returns
    -------
    anchor_point_list_dict : dict of list(s) of lists

    If ``quad_space==True`` , `anchor_point_list_dict` is a
    :obj:`dict` with four keys ``[0, 1, 2, 3]`` corresponding to the four
    subsets of chips generated (see `quad_space` ). If
    ``quad_space==False`` , `anchor_point_list_dict` is a
    :obj:`dict` with a single key, ``0`` , that corresponds to a list of all
    of the generated anchor points. Each anchor point in the list(s) is an
    ``[x, y]`` pair of UTM coordinates denoting the SW corner of a chip.

    """
    if extend:
        min_x = math.floor(utm_bounds[0])
        min_y = math.floor(utm_bounds[1])
        max_x = math.ceil(utm_bounds[2])
        max_y = math.ceil(utm_bounds[3])
    else:
        print("NoExtend")
        print('UTM_Bounds: {}'.format(utm_bounds))
        min_x = math.ceil(utm_bounds[0])
        min_y = math.ceil(utm_bounds[1])
        max_x = math.floor(utm_bounds[2])
        max_y = math.floor(utm_bounds[3])

    if quad_space:
        print("quad_space")
        row_cell = np.asarray([[0, 1], [2, 3]])
        anchor_point_list_dict = {0: [], 1: [], 2: [], 3: []}
    else:
        anchor_point_list_dict = {0: []}

    for rowidx, x in enumerate(np.arange(min_x, max_x, stride_size_meters)):
        for colidx, y in enumerate(np.arange(min_y, max_y,
                                             stride_size_meters)):
            if quad_space:
                anchor_point_list_dict[
                    row_cell[rowidx % 2, colidx % 2]].append([x, y])
            else:
                anchor_point_list_dict[0].append([x, y])

    return anchor_point_list_dict


def calculate_cells(anchor_point_list_dict, cell_size_meters, utm_bounds=[]):
    """ Calculate boundaries for image cells (chips) from anchor points.

    This function takes the output from :func:`calculate_anchor_points` as well
    as a desired cell size (`cell_size_meters`) and outputs
    ``(W, S, E, N)`` tuples for generating cells.

    Arguments
    ---------
    anchor_point_list_dict : dict
        Output of :func:`calculate_anchor_points`. See that function for
        details.
    cell_size_meters : int or float
        Desired width and height of each cell in meters.
    utm_bounds : list -like of float s, optional
        A :obj:`list`-like of shape ``(W, S, E, N)`` that defines the limits
        of an input image tile in UTM coordinates to ensure that no cells
        extend beyond those limits. If not provided, all cells will be included
        even if they extend beyond the UTM limits of the source imagery.

    Returns
    -------
    cells_list_dict : dict of list(s) of lists
        A dict whose keys are either ``0`` or ``[0, 1, 2, 3]`` (see
        :func:`calculate_anchor_points` . ``quad_space`` ), and whose values are
        :obj:`list` s of boundaries in the shape ``[W, S, E, N]`` . Boundaries
        are in UTM coordinates.

    """
    cells_list_dict = {}
    for anchor_point_list_id, anchor_point_list in anchor_point_list_dict.items():
        cells_list = []
        for anchor_point in anchor_point_list:
            if utm_bounds:
                if (anchor_point[0] + cell_size_meters < utm_bounds[2]) and (
                        anchor_point[1] + cell_size_meters < utm_bounds[3]):
                    cells_list.append([anchor_point[0], anchor_point[1],
                                       anchor_point[0] + cell_size_meters,
                                       anchor_point[1] + cell_size_meters])
                else:
                    pass
            else:  # include it regardless of extending beyond bounds
                cells_list.append([anchor_point[0], anchor_point[1],
                                   anchor_point[0] + cell_size_meters,
                                   anchor_point[1] + cell_size_meters])

        cells_list_dict[anchor_point_list_id] = cells_list

    return cells_list_dict


def calculate_analysis_grid(utm_bounds, stride_size_meters=300,
                            cell_size_meters=400, quad_space=False,
                            snapToGrid=False):
    """Wrapper for :func:`calculate_anchor_points` and :func:`calculate_cells`.

    Based on UTM boundaries of an image tile, stride size, and cell size,
    output a dictionary of boundary lists for analysis chips.

    Arguments
    ---------
    utm_bounds : list-like of shape ``(W, S, E, N)``
        UTM coordinate limits of the input tile.
    stride_size_meters : int, optional
        Step size in both X and Y directions between cells in units of meters.
        Defaults to ``300`` .
    cell_size_meters : int, optional
        Extent of each cell in both X and Y directions in units of meters.
        Defaults to ``400`` .
    quad_space : bool, optional
        See :func:`calculate_anchor_points` . ``quad_space`` . Defaults to
        ``False`` .
    snapToGrid : bool, optional
        .. :deprecated: 0.2.0
            This argument is deprecated and no longer has any effect.

    Returns
    -------
    cells_list_dict : dict of list(s) of lists
        A dict whose keys are either ``0`` or ``[0, 1, 2, 3]`` (see
        :func:`calculate_anchor_points` . ``quad_space`` ), and whose values are
        :obj:`list` s of boundaries in the shape ``[W, S, E, N]`` . Boundaries
        are in UTM coordinates.

    """
    anchor_point_list_dict = calculate_anchor_points(
        utm_bounds, stride_size_meters=stride_size_meters,
        quad_space=quad_space)
    cells_list_dict = calculate_cells(anchor_point_list_dict, cell_size_meters,
                                      utm_bounds=utm_bounds)
    return cells_list_dict


if __name__ == '__main__':
    utmX, utmY = 658029, 4006947
    cll_x = utmX
    cur_x = utmX + 500
    cll_y = utmY
    cur_y = utmY + 500
    stride_size_meters = 300
    cell_size_meters = 400
    ctile_size_pixels = 1600
    spacenetPath = "s3://spacenet-dataset/AOI_2_Vegas/srcData/rasterData/AOI_2_Vegas_MUL-PanSharpen_Cloud.tif"
    address = spacenetPath

    with rasterio.open(address) as src:
        cwgs_bounds = utils.get_wgs84_bounds(src)
        cutm_crs = utils.calculate_UTM_crs(cwgs_bounds)
        cutm_bounds = utils.get_utm_bounds(src, cutm_crs)

        #ccells_list = calculate_analysis_grid(cutm_bounds, stride_size_meters=stride_size_meters,
        #                                     cell_size_meters=cell_size_meters)

        #random_cell = random.choice(ccells_list)
        #cll_x, cll_y, cur_x, cur_y = random_cell
        tile, mask, window, window_transform = tile_utm(src, cll_x, cll_y, cur_x, cur_y, indexes=None, tilesize=ctile_size_pixels, nodata=None, alpha=None,
                        dst_crs=cutm_crs)
