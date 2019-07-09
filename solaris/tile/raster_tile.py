import os
import rasterio
from rasterio.warp import transform_bounds, Resampling
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.crs import CRS
from rasterio import transform
from shapely.geometry import box
import math
from rio_cogeo.cogeo import cog_validate, cog_translate
from ..utils.core import _check_crs, _check_rasterio_im_load
# removing the following until COG functionality is implemented
# from ..utils.tile import read_cog_tile
from ..utils.geo import latlon_to_utm_epsg, reproject_geometry, reproject
from ..utils.geo import raster_get_projection_unit
from tqdm import tqdm
import numpy as np


class RasterTiler(object):
    """An object to tile geospatial image strips into smaller pieces.

    Arguments
    ---------
    dest_dir : str, optional
        Path to save output files to. If not specified here, this
        must be provided when ``Tiler.tile_generator()`` is called.
    src_tile_size : `tuple` of `int`s, optional
        The size of the output tiles in ``(y, x)`` coordinates. By default,
        this is in pixel units; this can be changed to metric units using the
        `src_metric_size` argument.
    src_metric_size : bool, optional
        Is `src_tile_size` in pixel units (default) or metric? To set to metric
        use ``src_metric_size=True``.
    dest_tile_size : `tuple` of `int`s, optional
        The size of the output tiles in ``(y, x)`` coordinates in pixel units.
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
    aoi_bounds : list, optional
        A :class:`list` -like of shape
        ``[left_bound, bottom_bound, right_bound, top_bound]`` defining the
        extent of the area of interest to be tiled, in the same units as
        defined by `src_metric_size`. If not provided either upon
        initialization or when an image is loaded, the image bounds will be
        used; if provided, this value will override image metadata.
    verbose : bool, optional
        Verbose text output. By default, verbose text is not printed.

    Attributes
    ----------
    src : :class:`rasterio.io.DatasetReader`
        The source dataset to tile.
    src_path : `str`
        The path or URL to the source dataset. Used for calling
        ``rio_cogeo.cogeo.cog_validate()``.
    dest_dir : `str`
        The directory to save the output tiles to. If not
    dest_crs : int
        The EPSG code for the output images. If not provided, outputs will
        keep the same CRS as the source image when ``Tiler.make_tile_images()``
        is called.
    tile_size: tuple
        A ``(y, x)`` :class:`tuple` storing the dimensions of the output.
        These are in pixel units unless ``size_in_meters=True``.
    size_in_meters : bool
        If ``True``, the units of `tile_size` are in meters instead of pixels.
    is_cog : bool
        Indicates whether or not the image being tiled is a Cloud-Optimized
        GeoTIFF (COG). Determined by checking COG validity using
        `rio-cogeo <https://github.com/cogeotiff/rio-cogeo>`_.
    nodata : `int`
        The value for nodata in the outputs. Will be set to zero in outputs if
        ``None``.
    alpha : `int`
        The band index corresponding to an alpha channel (if one exists).
        ``None`` if there is no alpha channel.
    tile_bounds : list
        A :class:`list` containing ``[left, bottom, right, top]`` bounds
        sublists for each tile created.
    resampling : str
        The resampling method for any resizing. Possible values are
        ``['bilinear', 'cubic', 'nearest', 'lanczos', 'average']`` (or any
        other option from :class:`rasterio.warp.Resampling`).
    aoi_bounds : :class:`shapely.geometry.Polygon`
        A :class:`shapely.geometry.Polygon` defining the bounds of the AOI that
        tiles will be created for. If a tile will extend beyond the boundary,
        the "extra" pixels will have the value `nodata`. Can be provided at
        initialization of the :class:`Tiler` instance or when the input is
        loaded.
    """

    def __init__(self, dest_dir=None, dest_crs=None, channel_idxs=None,
                 src_tile_size=(900, 900), src_metric_size=False,
                 dest_tile_size=None, dest_metric_size=False,
                 aoi_bounds=None, nodata=None, alpha=None,
                 force_load_cog=False, resampling='bilinear',
                 verbose=False):
        # set up attributes
        if verbose:
            print("Initializing Tiler...")
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        if dest_crs is not None:
            self.dest_crs = _check_crs(dest_crs)
        else:
            self.dest_crs = None
        self.src_tile_size = src_tile_size
        self.src_metric_size = src_metric_size
        if dest_tile_size is None:
            self.dest_tile_size = src_tile_size
        else:
            self.dest_tile_size = dest_tile_size
        self.resampling = resampling
        self.force_load_cog = force_load_cog
        self.nodata = nodata
        self.alpha = alpha
        self.aoi_bounds = aoi_bounds
#        self.cog_output = cog_output
        self.verbose = verbose
        if self.verbose:
            print('Tiler initialized.')
            print('dest_dir: {}'.format(self.dest_dir))
            if dest_crs is not None:
                print('dest_crs: EPSG:{}'.format(self.dest_crs))
            else:
                print('dest_crs will be inferred from source data.')
            print('src_tile_size: {}'.format(self.src_tile_size))
            print('tile size units metric: {}'.format(self.src_metric_size))

    def tile(self, src, dest_dir=None, channel_idxs=None, nodata=None,
             alpha=None, aoi_bounds=None, restrict_to_aoi=False):

        tile_gen = self.tile_generator(src, dest_dir, channel_idxs, nodata,
                                       alpha, aoi_bounds, restrict_to_aoi)

        if self.verbose:
            print('Beginning tiling...')
        for tile_data, mask, profile in tqdm(tile_gen):
            self.save_tile(tile_data, mask, profile)
        if self.verbose:
            print('Tiling complete. Cleaning up...')
        self.src.close()
        if os.path.exists(os.path.join(self.dest_dir, 'tmp.tif')) and \
                self.src_crs != _check_crs(self.src.crs):
            os.remove(os.path.join(self.dest_dir, 'tmp.tif'))
        if self.verbose:
            print("Done.")

    def tile_generator(self, src, dest_dir=None, channel_idxs=None,
                       nodata=None, alpha=None, aoi_bounds=None,
                       restrict_to_aoi=False):
        """Create the tiled output imagery from input tiles.

        Uses the arguments provided at initialization to generate output tiles.
        First, tile locations are generated based on `Tiler.tile_size` and
        `Tiler.size_in_meters` given the bounds of the input image.

        Arguments
        ---------
        src : `str` or :class:`Rasterio.DatasetReader`
            The source data to tile from. If this is a "classic"
            (non-cloud-optimized) GeoTIFF, the whole image will be loaded in;
            if it's cloud-optimized, only the required portions will be loaded
            during tiling unless ``force_load_cog=True`` was specified upon
            initialization.
        dest_dir : str, optional
            The path to the destination directory to output images to. If the
            path doesn't exist, it will be created. This argument is required
            if it wasn't provided during initialization.
        channel_idxs : list, optional
            The list of channel indices to be included in the output array.
            If not provided, all channels will be included. *Note:* per
            ``rasterio`` convention, indexing starts at ``1``, not ``0``.
        nodata : int, optional
            The value in `src` that specifies nodata. If this value is not
            provided, solaris will attempt to infer the nodata value from the
            `src` metadata.
        alpha : int, optional
            The band to specify as alpha. If not provided, solaris will attempt
            to infer if an alpha band is present from the `src` metadata.
        aoi_bounds : `list`-like or :class:`shapely.geometry.Polygon`, optional
            AOI bounds can be provided either as a
            ``[left, bottom, right, top]`` :class:`list`-like or as a
            :class:`shapely.geometry.Polygon`.
        restrict_to_aoi : bool, optional
            Should output tiles be restricted to the limits of the AOI? If
            ``True``, any tile that partially extends beyond the limits of the
            AOI will not be returned. This is the inverse of the ``boundless``
            argument for :class:`rasterio.io.DatasetReader` 's ``.read()``
            method.

        Yields
        ------
        tile_data, mask, tile_bounds
            tile_data : :class:`numpy.ndarray`
            A list of lists of each tile's bounds in the order they were
            created, to be used in tiling vector data. These data are also
            stored as an attribute of the :class:`Tiler` instance named
            `tile_bounds`.

        """
        # parse arguments
        if self.verbose:
            print("Checking input data...")
        if isinstance(src, str):
            self.is_cog = cog_validate(src)
        else:
            self.is_cog = cog_validate(src.name)
        if self.verbose:
            print('COG: {}'.format(self.is_cog))
        self.src = _check_rasterio_im_load(src)
        if channel_idxs is None:  # if not provided, include them all
            channel_idxs = list(range(1, self.src.count + 1))
            print(channel_idxs)
        self.src_crs = _check_crs(self.src.crs)
        if self.verbose:
            print('Source CRS: EPSG:{}'.format(self.src_crs))
        if self.dest_crs is None:
            self.dest_crs = self.src_crs
        if self.verbose:
            print('Destination CRS: EPSG:{}'.format(self.dest_crs))
        self.src_path = self.src.name
        self.proj_unit = raster_get_projection_unit(
            self.src).strip('"').strip("'")
        if self.verbose:
            print("Inputs OK.")
        if self.src_metric_size:
            if self.verbose:
                print("Checking if inputs are in metric units...")
            if self.proj_unit not in ['meter', 'metre']:
                if self.verbose:
                    print("Input CRS is not metric. "
                          "Reprojecting the input to UTM.")
                self.src = reproject(self.src,
                                     resampling_method=self.resampling,
                                     dest_path=os.path.join(self.dest_dir,
                                                            'tmp.tif'))
                if self.verbose:
                    print('Done reprojecting.')
        if nodata is None and self.nodata is None:
            self.nodata = self.src.nodata
        else:
            self.nodata = nodata
        # get index of alpha channel
        if alpha is None and self.alpha is None:
            mf_list = [rasterio.enums.MaskFlags.alpha in i for i in
                       self.src.mask_flag_enums]  # list with True at idx of alpha c
            try:
                self.alpha = np.where(mf_list)[0] + 1
            except IndexError:  # if there isn't a True
                self.alpha = None
        else:
            self.alpha = alpha

        if getattr(self, 'tile_bounds', None) is None:
            self.get_tile_bounds()

        for tb in self.tile_bounds:
            # removing the following line until COG functionality implemented
            if True:  # not self.is_cog or self.force_load_cog:
                vrt = self.load_src_vrt()
                window = vrt.window(*tb)
                if self.src.count != 1:
                    tile_data = vrt.read(window=window,
                                         resampling=getattr(Resampling,
                                                            self.resampling),
                                         indexes=channel_idxs)
                else:
                    tile_data = vrt.read(window=window,
                                         resampling=getattr(Resampling,
                                                            self.resampling))
                # get the affine xform between src and dest for the tile
                aff_xform = transform.from_bounds(*tb,
                                                  self.dest_tile_size[1],
                                                  self.dest_tile_size[0])
                if self.nodata:
                    mask = np.all(tile_data != nodata,
                                  axis=0).astype(np.uint8) * 255
                elif self.alpha:
                    mask = vrt.read(self.alpha, window=window,
                                    resampling=getattr(Resampling,
                                                       self.resampling))
                else:
                    mask = None  # placeholder

            # else:
            #     tile_data, mask, window, aff_xform = read_cog_tile(
            #         src=self.src,
            #         bounds=tb,
            #         tile_size=self.dest_tile_size,
            #         indexes=channel_idxs,
            #         nodata=self.nodata,
            #         resampling_method=self.resampling
            #         )
            profile = self.src.profile
            profile.update(width=self.dest_tile_size[1],
                           height=self.dest_tile_size[0],
                           crs=CRS.from_epsg(self.dest_crs),
                           transform=aff_xform)
            if len(tile_data.shape) == 2:  # if there's no channel band
                profile.update(count=1)
            else:
                profile.update(count=tile_data.shape[0])

            yield tile_data, mask, profile

    def save_tile(self, tile_data, mask, profile):
        """Save a tile created by ``Tiler.tile_generator()``."""
        dest_fname_root = os.path.splitext(os.path.split(self.src_path)[1])[0]
        if self.proj_unit not in ['meter', 'metre']:
            dest_fname = '{}_{}_{}.tif'.format(
                dest_fname_root,
                np.round(profile['transform'][2], 3),
                np.round(profile['transform'][5], 3))
        else:
            dest_fname = '{}_{}_{}.tif'.format(
                dest_fname_root,
                int(profile['transform'][2]),
                int(profile['transform'][5]))
        # if self.cog_output:
        #     dest_path = os.path.join(self.dest_dir, 'tmp.tif')
        #else:
        dest_path = os.path.join(self.dest_dir, dest_fname)

        with rasterio.open(dest_path, 'w',
                           **profile) as dest:
            if profile['count'] == 1:
                dest.write(tile_data[0, :, :], 1)
            else:
                for band in range(1, profile['count'] + 1):
                    # base-1 vs. base-0 indexing...bleh
                    dest.write(tile_data[band-1, :, :], band)
            if self.alpha:
                # write the mask if there's an alpha band
                dest.write(mask, profile['count'] + 1)

            dest.close()

        # if self.cog_output:
        #     self._create_cog(os.path.join(self.dest_dir, 'tmp.tif'),
        #                      os.path.join(self.dest_dir, dest_fname))
        #     os.remove(os.path.join(self.dest_dir, 'tmp.tif'))

    def _create_cog(self, src_path, dest_path):
        """Overwrite non-cloud-optimized GeoTIFF with a COG."""
        cog_translate(src_path=src_path, dst_path=dest_path,
                      dst_kwargs={'crs': CRS.from_epsg(self.dest_crs)},
                      resampling=self.resampling,
                      latitude_adjustment=False)

    def get_tile_bounds(self):
        """Get tile bounds for each tile to be created in the input CRS."""
        if not self.aoi_bounds:
            if not self.src:
                raise ValueError('aoi_bounds and/or a source file must be '
                                 'provided.')
            else:
                # set to the bounds of the image
                self.aoi_bounds = self.src.bounds
        if not self.src_metric_size:
            xform = self.src.transform
            # convert pixel units to CRS units to use during image tiling.
            # NOTE: This will be imperfect for large AOIs where there isn't
            # a constant relationship between the src CRS units and src pixel
            # units.
            tmp_tile_size = [self.src_tile_size[0]*xform[0],
                             self.src_tile_size[1]*-xform[4]]
        else:
            tmp_tile_size = self.src_tile_size

        x_extent = self.aoi_bounds.right - self.aoi_bounds.left
        y_extent = self.aoi_bounds.top - self.aoi_bounds.bottom
        x_steps = np.ceil(x_extent/tmp_tile_size[1])
        y_steps = np.ceil(y_extent/tmp_tile_size[0])
        x_mins = np.arange(self.aoi_bounds.left,
                           self.aoi_bounds.left + tmp_tile_size[1]*x_steps,
                           tmp_tile_size[1])
        y_mins = np.arange(self.aoi_bounds.bottom,
                           self.aoi_bounds.bottom + tmp_tile_size[0]*y_steps,
                           tmp_tile_size[0])
        self.tile_bounds = [(i,
                             j,
                             i+tmp_tile_size[1],
                             j+tmp_tile_size[0])
                            for i in x_mins for j in y_mins]

    def load_src_vrt(self):
        """Load a source dataset's VRT into the destination CRS."""
        vrt_params = dict(crs=CRS.from_epsg(self.dest_crs),
                          resampling=getattr(Resampling, self.resampling),
                          src_nodata=self.nodata, dst_nodata=self.nodata)
        return WarpedVRT(self.src, **vrt_params)
