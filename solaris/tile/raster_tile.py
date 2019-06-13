import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.io import DatasetReader
from shapely.geometry import box
import math
from rio_cogeo.cogeo import cog_validate
from ..utils.core import _check_crs, _check_rasterio_im_load
from ..utils.tile import tile_exists_utm
from ..utils.geo import latlon_to_utm_epsg, reproject_geometry
import numpy as np


class Tiler(object):
    """An object to tile geospatial image strips into smaller pieces.

    Arguments
    ---------
    src : str or :class:`rasterio.io.DatasetReader`
        The source imagery to tile. Can either be a path or URL to an image
        or an image object already loaded using :func:`rasterio.open`.
    dest_dir : str
        Path to save output files to.
    tile_size : `tuple` of `int`s, optional
        The size of the output tiles in ``(y, x)`` coordinates. By default,
        this is in pixel units; this can be changed to metric units using the
        `size_in_meters` argument.
    size_in_meters : bool, optional
        Is `tile_size` in pixel units (default) or metric? To set to metric,
        use ``size_in_meters=True``.
    dest_crs : int, optional
        The EPSG code for the CRS that output tiles are in. If not provided,
        tiles use the crs of `src` by default.
    nodata : int, optional
        The value in `src` that specifies nodata. If this value is not
        provided, solaris will attempt to infer the nodata value from the `src`
        metadata.
    alpha : int, optional
        The band to specify as alpha. If not provided, solaris will attempt to
        infer if an alpha band is present from the `src` metadata.
    force_load_cog : bool, optional
        If `src` is a cloud-optimized geotiff, use this argument to force
        loading in the entire image at once.
    verbose : bool, optional
        Verbose text output. By default, verbose text is not printed.

    Attributes
    ----------
    src : :class:`rasterio.io.DatasetReader`
        The source dataset to tile.
    src_path : str
        The path or URL to the source dataset. Used for calling
        ``rio_cogeo.cogeo.cog_validate()``.
    dest_dir : str
        The directory to save the output tiles to.
    dest_crs : int
        The EPSG code for the output images.
    tile_size: tuple
        A ``(y, x)`` :class:`tuple` storing the dimensions of the output.
        These are in pixel units unless ``size_in_meters=True``.
    size_in_meters : bool
        If ``True``, the units of `tile_size` are in meters instead of pixels.
    is_cog : bool
        Indicates whether or not the image being tiled is a Cloud-Optimized
        GeoTIFF (COG). Determined by checking COG validity using
        `rio-cogeo <https://github.com/cogeotiff/rio-cogeo>`_.
    nodata : `int` or ``None``
        The value for nodata in the outputs. Will be set to zero in outputs if
        ``None``.
    alpha : `int` or ``None``
        The band index corresponding to an alpha channel (if one exists).
        ``None`` if there is no alpha channel.
    tile_bounds : list
        A :class:`list` containing ``[left, bottom, right, top]`` bounds
        sublists for each tile created.
    """

    def __init__(self, src, dest_dir, tile_size=(900, 900),
                 size_in_meters=False,
                 dest_crs=None, nodata=None, alpha=None, force_load_cog=False,
                 verbose=False):
        # set up attributes
        if verbose:
            print("Processing Tiler arguments...")
        if isinstance(src, str):
            self.is_cog = cog_validate(src)
        else:
            self.is_cog = cog_validate(src.name)
        # determine whether or not image needs to be loaded (is it a cog, etc)
        if self.is_cog and not force_load_cog:
            self.src = src
        else:
            self.src = _check_rasterio_im_load(src)

        self.src_path = self.src.name
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        if dest_crs is not None:
            self.dest_crs = _check_crs(dest_crs)
        else:
            self.dest_crs = _check_crs(self.src.crs)
        self.tile_size = tile_size
        self.size_in_meters = size_in_meters

        if nodata is None:
            self.nodata = src.nodata
        else:
            self.nodata = nodata
        # get index of alpha channel
        if alpha is None:
            mf_list = [rasterio.enums.MaskFlags.alpha in i for i in
                       src.mask_flag_enums]  # list with True at idx of alpha c
            try:
                self.alpha = np.where(mf_list)[0] + 1
            except IndexError:  # if there isn't a True
                self.alpha = None
        else:
            self.alpha = alpha
        self.verbose = verbose
        if self.verbose:
            print('Tiler arguments processed.')
            print('is_cog: {}'.format(self.is_cog))
            print('src: {}'.format(self.src))
            print('src_path: {}'.format(self.src_path))
            print('dest_dir: {}'.format(self.dest_dir))
            print('dest_crs: {}'.format(self.dest_crs))
            print('tile_size: {}'.format(self.tile_size))
            print('size_in_meters: {}'.format(self.size_in_meters))
            print('nodata: {}'.format(self.nodata))
            print('alpha: {}'.format(self.alpha))

    def make_tile_images(self):
        """Create the tiled output imagery from input tiles.

        Uses the arguments provided at initialization to generate output tiles.
        First, tile locations are generated based on `Tiler.tile_size` and
        `Tiler.size_in_meters` given the bounds of the input image.

        Arguments
        ---------
        None

        Returns
        -------
        tile_bounds: list
            A list of the bounds of each tile generated in the order they were
            created, to be used in tiling vector data. These data are also
            stored as an attribute of the :class:`Tiler` instance named
            `tile_bounds`.


def tile_utm_source(src, ll_x, ll_y, ur_x, ur_y, indexes=None, tilesize=256,
                    nodata=None, alpha=None, dst_crs=4326):
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
        EPSG code for the output coordinate reference system. Defaults to
        ``4326``.

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
    dst_crs = _check_crs(dst_crs)
    wgs_bounds = transform_bounds(
        *[src.crs,
          rasterio.crs.CRS.from_epsg(dst_crs)] + list(src.bounds),
        densify_pts=21)

    indexes = indexes if indexes is not None else src.indexes
    tile_bounds = (ll_x, ll_y, ur_x, ur_y)
    if not tile_exists_utm(wgs_bounds, tile_bounds):
        raise TileOutsideBounds(
            'Tile {}/{}/{}/{} is outside image bounds'.format(tile_bounds))

    return tile_read_utm(src, tile_bounds, tilesize, indexes=indexes,
                         nodata=nodata, alpha=alpha, dst_crs=dst_crs)


def tile_utm(source, ll_x, ll_y, ur_x, ur_y, indexes=None, tilesize=256,
             nodata=None, alpha=None, dst_crs=4326):
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
    dst_crs : int, optional
        EPSG code for the output CRS. Defaults to ``4326``.

    Returns
    -------
    ``(data, mask, window, window_transform`` tuple.
        data : :py:class:`numpy.ndarray`
            int pixel values. Shape is ``(C, Y, X)`` if retrieving multiple
            channels, ``(Y, X)`` otherwise.
        mask : :class:`numpy.ndarray`
            int mask indicating which pixels contain information and which are
            `nodata`. Pixels containing data have value ``255``, `nodata`
            pixels have value ``0``.
        window : :py:class:`rasterio.windows.Window`
            :py:class:`rasterio.windows.Window` object indicating the raster
            location of the dataset subregion being returned in `data`.
        window_transform : :py:class:`affine.Affine`
            Affine transformation for the indow.

    """
    dst_crs = _check_crs(dst_crs)
    if isinstance(source, DatasetReader):
        src = source
    elif os.path.exists(source):
        src = rasterio.open(source)  # read in the file
    else:
        raise ValueError('Source is not a rasterio.Dataset or a valid path.')

    return tile_utm_source(src, ll_x, ll_y, ur_x, ur_y, indexes=indexes,
                           tilesize=tilesize, nodata=nodata, alpha=alpha,
                           dst_crs=dst_crs)


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
        :func:`solaris.utils.get_wgs84_bounds` and
        :func:`solaris.utils.calculate_utm_crs` .
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

    source = _check_rasterio_im_load(source)

    if not utm_crs:
        src_bounds = source.bounds
        src_bounds = box(src_bounds['left'], src_bounds['bottom'],
                         src_bounds['right'], src_bounds['top'])
        wgs_pt = reproject_geometry(src_bounds,
                                        source.crs, 4326).centroid.coords
        utm_crs = latlon_to_utm_epsg(**wgs_pt)

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
