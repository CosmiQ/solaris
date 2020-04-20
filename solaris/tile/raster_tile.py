import os
import rasterio
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.vrt import WarpedVRT
from rasterio.mask import mask as rasterio_mask
# from rio_cogeo.cogeo import cog_validate, cog_translate
from ..utils.core import _check_crs, _check_rasterio_im_load
# removing the following until COG functionality is implemented
# from ..utils.tile import read_cog_tile
from ..utils.geo import reproject, split_geom, raster_get_projection_unit
import numpy as np
from shapely.geometry import box
from tqdm import tqdm

class RasterTiler(object):
    """An object to tile geospatial image strips into smaller pieces.

    Arguments
    ---------
    dest_dir : str, optional
        Path to save output files to. If not specified here, this
        must be provided when ``Tiler.tile_generator()`` is called.
    src_tile_size : `tuple` of `int`s, optional
        The size of the input tiles in ``(y, x)`` coordinates. By default,
        this is in pixel units; this can be changed to metric units using the
        `use_src_metric_size` argument.
    use_src_metric_size : bool, optional
        Is `src_tile_size` in pixel units (default) or metric? To set to metric
        use ``use_src_metric_size=True``.
    dest_tile_size : `tuple` of `int`s, optional
        The size of the output tiles in ``(y, x)`` coordinates in pixel units.
    dest_crs : int, optional
        The EPSG code or rasterio.crs.CRS object for the CRS that output tiles are in.
        If not provided, tiles use the crs of `src` by default. Cannot be specified
        along with project_to_meters.
    project_to_meters : bool, optional
        Specifies whether to project to the correct utm zone for the location.
        Cannot be specified along with `dest_crs`.
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
    aoi_boundary : :class:`shapely.geometry.Polygon` or `list`-like [left, bottom, right, top]
        Defines the bounds of the AOI in which tiles will be created. If a
        tile will extend beyond the boundary, the "extra" pixels will have
        the value `nodata`. Can be provided at initialization of the :class:`Tiler`
        instance or when the input is loaded. If not provided either upon
        initialization or when an image is loaded, the image bounds will be
        used; if provided, this value will override image metadata.
    tile_bounds : `list`-like
        A `list`-like of ``[left, bottom, right, top]`` lists of coordinates
        defining the boundaries of the tiles to create. If not provided, they
        will be generated from the `aoi_boundary` based on `src_tile_size`.
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
    aoi_boundary : :class:`shapely.geometry.Polygon`
        A :class:`shapely.geometry.Polygon` defining the bounds of the AOI that
        tiles will be created for. If a tile will extend beyond the boundary,
        the "extra" pixels will have the value `nodata`. Can be provided at
        initialization of the :class:`Tiler` instance or when the input is
        loaded.
    """

    def __init__(self, dest_dir=None, dest_crs=None, project_to_meters=False,
                 channel_idxs=None, src_tile_size=(900, 900), use_src_metric_size=False,
                 dest_tile_size=None, dest_metric_size=False,
                 aoi_boundary=None, nodata=None, alpha=None,
                 force_load_cog=False, resampling=None, tile_bounds=None,
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
        self.use_src_metric_size = use_src_metric_size
        if dest_tile_size is None:
            self.dest_tile_size = src_tile_size
        else:
            self.dest_tile_size = dest_tile_size
        self.resampling = resampling
        self.force_load_cog = force_load_cog
        self.nodata = nodata
        self.alpha = alpha
        self.aoi_boundary = aoi_boundary
        self.tile_bounds = tile_bounds
        self.project_to_meters = project_to_meters
        self.tile_paths = []  # retains the paths of the last call to .tile()
#        self.cog_output = cog_output
        self.verbose = verbose
        if self.verbose:
            print('Tiler initialized.')
            print('dest_dir: {}'.format(self.dest_dir))
            if dest_crs is not None:
                print('dest_crs: {}'.format(self.dest_crs))
            else:
                print('dest_crs will be inferred from source data.')
            print('src_tile_size: {}'.format(self.src_tile_size))
            print('tile size units metric: {}'.format(self.use_src_metric_size))
            if self.resampling is not None:
                print('Resampling is set to {}'.format(self.resampling))
            else:
                print('Resampling is set to None')

    def tile(self, src, dest_dir=None, channel_idxs=None, nodata=None,
             alpha=None, restrict_to_aoi=False,
             dest_fname_base=None, nodata_threshold = None):
        """An object to tile geospatial image strips into smaller pieces.

        Arguments
        ---------
        src : :class:`rasterio.io.DatasetReader` or str
            The source dataset to tile.
        nodata_threshold : float, optional
            Nodata percentages greater than this threshold will not be saved as tiles.
        restrict_to_aoi : bool, optional
            Requires aoi_boundary. Sets all pixel values outside the aoi_boundary to the nodata value of the src image.
        """
        src = _check_rasterio_im_load(src)
        restricted_im_path = os.path.join(self.dest_dir, "aoi_restricted_"+ os.path.basename(src.name))
        self.src_name = src.name # preserves original src name in case restrict is used
        if restrict_to_aoi is True:
            if self.aoi_boundary is None:
                raise ValueError("aoi_boundary must be specified when RasterTiler is called.")
            mask_geometry = self.aoi_boundary.intersection(box(*src.bounds)) # prevents enlarging raster to size of aoi_boundary
            index_lst = list(np.arange(1,src.meta['count']+1))
            # no need to use transform t since we don't crop. cropping messes up transform of tiled outputs
            arr, t = rasterio_mask(src, [mask_geometry], all_touched=False, invert=False, nodata=src.meta['nodata'], 
                         filled=True, crop=False, pad=False, pad_width=0.5, indexes=list(index_lst))
            with rasterio.open(restricted_im_path, 'w', **src.profile) as dest:
                dest.write(arr)
                dest.close()
                src.close()
            src = _check_rasterio_im_load(restricted_im_path) #if restrict_to_aoi, we overwrite the src to be the masked raster
            
        tile_gen = self.tile_generator(src, dest_dir, channel_idxs, nodata,
                                       alpha, self.aoi_boundary, restrict_to_aoi)

        if self.verbose:
            print('Beginning tiling...')   
        self.tile_paths = []
        if nodata_threshold is not None:
            if nodata_threshold > 1:
                raise ValueError("nodata_threshold should be expressed as a float less than 1.")
            print("nodata value threshold supplied, filtering based on this percentage.")
            new_tile_bounds = []
            for tile_data, mask, profile, tb in tqdm(tile_gen):
                nodata_count = np.logical_or.reduce((tile_data == profile['nodata']), axis=0).sum()
                nodata_perc = nodata_count / (tile_data.shape[1] * tile_data.shape[2])
                if nodata_perc < nodata_threshold:
                    dest_path = self.save_tile(
                        tile_data, mask, profile, dest_fname_base)
                    self.tile_paths.append(dest_path)
                    new_tile_bounds.append(tb)
                else:
                    print("{} of nodata is over the nodata_threshold, tile not saved.".format(nodata_perc))
            self.tile_bounds = new_tile_bounds # only keep the tile bounds that make it past the nodata threshold
        else:
            for tile_data, mask, profile, tb in tqdm(tile_gen):
                dest_path = self.save_tile(
                    tile_data, mask, profile, dest_fname_base)
                self.tile_paths.append(dest_path)
        if self.verbose:
            print('Tiling complete. Cleaning up...')
        self.src.close()
        if os.path.exists(os.path.join(self.dest_dir, 'tmp.tif')):
            os.remove(os.path.join(self.dest_dir, 'tmp.tif'))
        if os.path.exists(restricted_im_path):
            os.remove(restricted_im_path)
        if self.verbose:
            print("Done. CRS returned for vector tiling.")
        return _check_crs(profile['crs'])  # returns the crs to be used for vector tiling

    def tile_generator(self, src, dest_dir=None, channel_idxs=None,
                       nodata=None, alpha=None, aoi_boundary=None,
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
        aoi_boundary : `list`-like or :class:`shapely.geometry.Polygon`, optional
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
        # if isinstance(src, str):
        #     self.is_cog = cog_validate(src)
        # else:
        # self.is_cog = cog_validate(src.name)
        # if self.verbose:
        #     print('COG: {}'.format(self.is_cog))
        self.src = _check_rasterio_im_load(src)
        if channel_idxs is None:  # if not provided, include them all
            channel_idxs = list(range(1, self.src.count + 1))
            print(channel_idxs)
        self.src_crs = _check_crs(self.src.crs, return_rasterio=True) # necessary to use rasterio crs for reproject
        if self.verbose:
            print('Source CRS: EPSG:{}'.format(self.src_crs.to_epsg()))
        if self.dest_crs is None:
            self.dest_crs = self.src_crs
        if self.verbose:
            print('Destination CRS: EPSG:{}'.format(self.dest_crs.to_epsg()))
        self.src_path = self.src.name
        self.proj_unit = raster_get_projection_unit(self.src)  # for rounding
        if self.verbose:
            print("Inputs OK.")
        if self.use_src_metric_size:
            if self.verbose:
                print("Checking if inputs are in metric units...")
            if self.project_to_meters:
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
                window = rasterio.windows.from_bounds(
                    *tb, transform=self.src.transform,
                    width=self.src_tile_size[1],
                    height=self.src_tile_size[0])

                if self.src.count != 1:
                    src_data = self.src.read(
                        window=window,
                        indexes=channel_idxs, boundless=True)
                else:
                    src_data = self.src.read(
                        window=window,
                        boundless=True)

                dst_transform, width, height = calculate_default_transform(
                    self.src.crs, self.dest_crs,
                    self.src.width, self.src.height, *tb,
                    dst_height=self.dest_tile_size[0],
                    dst_width=self.dest_tile_size[1])

                if self.dest_crs != self.src_crs and self.resampling_method is not None:
                    tile_data = np.zeros(shape=(src_data.shape[0], height, width),
                                         dtype=src_data.dtype)
                    rasterio.warp.reproject(
                        source=src_data,
                        destination=tile_data,
                        src_transform=self.src.window_transform(window),
                        src_crs=self.src.crs,
                        dst_transform=dst_transform,
                        dst_crs=self.dest_crs,
                        resampling=getattr(Resampling, self.resampling))

                elif self.dest_crs != self.src_crs and self.resampling_method is None:
                    print("Warning: You've set resampling to None but your "
                          "destination projection differs from the source "
                          "projection. Using bilinear resampling by default.")
                    tile_data = np.zeros(shape=(src_data.shape[0], height, width),
                                         dtype=src_data.dtype)
                    rasterio.warp.reproject(
                        source=src_data,
                        destination=tile_data,
                        src_transform=self.src.window_transform(window),
                        src_crs=self.src.crs,
                        dst_transform=dst_transform,
                        dst_crs=self.dest_crs,
                        resampling=getattr(Resampling, "bilinear"))

                else:  # for the case where there is no resampling and no dest_crs specified, no need to reproject or resample

                    tile_data = src_data

                if self.nodata:
                    mask = np.all(tile_data != nodata,
                                  axis=0).astype(np.uint8) * 255
                elif self.alpha:
                    mask = self.src.read(self.alpha, window=window)
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
                           crs=self.dest_crs,
                           transform=dst_transform)
            if len(tile_data.shape) == 2:  # if there's no channel band
                profile.update(count=1)
            else:
                profile.update(count=tile_data.shape[0])

            yield tile_data, mask, profile, tb

    def save_tile(self, tile_data, mask, profile, dest_fname_base=None):
        """Save a tile created by ``Tiler.tile_generator()``."""
        if dest_fname_base is None:
            dest_fname_root = os.path.splitext(
                os.path.split(self.src_path)[1])[0]
        else:
            dest_fname_root = dest_fname_base
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
        # else:
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

        return dest_path

        # if self.cog_output:
        #     self._create_cog(os.path.join(self.dest_dir, 'tmp.tif'),
        #                      os.path.join(self.dest_dir, dest_fname))
        #     os.remove(os.path.join(self.dest_dir, 'tmp.tif'))
        
    def fill_all_nodata(self, nodata_fill):
        """
        Fills all tile nodata values with a fill value.
        
        The standard workflow is to run this function only after generating label masks and using the original output 
        from the raster tiler to filter out label pixels that overlap nodata pixels in a tile. For example, 
        solaris.vector.mask.instance_mask will filter out nodata pixels from a label mask if a reference_im is provided,
        and after this step nodata pixels may be filled by calling this method.
        
        nodata_fill : int, float, or str, optional
            Default is to not fill any nodata values. Otherwise, pixels outside of the aoi_boundary and pixels inside 
            the aoi_boundary with the nodata value will be filled. "mean" will fill pixels with the channel-wise mean. 
            Providing an int or float will fill pixels in all channels with the provided value.
            
        Returns: list
            The fill values, in case the mean of the src image should be used for normalization later.
        """
        src = _check_rasterio_im_load(self.src_name)
        if nodata_fill == "mean":
            arr = src.read()
            arr_nan = np.where(arr!=src.nodata, arr, np.nan)
            fill_values = np.nanmean(arr_nan, axis=tuple(range(1, arr_nan.ndim)))
            print('Fill values set to {}'.format(fill_values))
        elif isinstance(nodata_fill, (float, int)):
            fill_values = src.meta['count'] * [nodata_fill]
            print('Fill values set to {}'.format(fill_values))
        else:
            raise TypeError('nodata_fill must be "mean", int, or float. {} was supplied.'.format(nodata_fill))
        src.close()
        for tile_path in self.tile_paths:
            tile_src = rasterio.open(tile_path, "r+")
            tile_data = tile_src.read()
            for i in np.arange(tile_data.shape[0]):
                tile_data[i,...][tile_data[i,...] == tile_src.nodata] = fill_values[i] # set fill value for each band
            if tile_src.meta['count'] == 1:
                tile_src.write(tile_data[0, :, :], 1)
            else:
                for band in range(1, tile_src.meta['count'] + 1):
                    # base-1 vs. base-0 indexing...bleh
                    tile_src.write(tile_data[band-1, :, :], band)
            tile_src.close()
        return fill_values

    def _create_cog(self, src_path, dest_path):
        """Overwrite non-cloud-optimized GeoTIFF with a COG."""
        cog_translate(src_path=src_path, dst_path=dest_path,
                      dst_kwargs={'crs': self.dest_crs},
                      resampling=self.resampling,
                      latitude_adjustment=False)

    def get_tile_bounds(self):
        """Get tile bounds for each tile to be created in the input CRS."""
        if not self.aoi_boundary:
            if not self.src:
                raise ValueError('aoi_boundary and/or a source file must be '
                                 'provided.')
            else:
                # set to the bounds of the image
                # split_geom can take a list
                self.aoi_boundary = list(self.src.bounds)

        self.tile_bounds = split_geom(geometry=self.aoi_boundary, tile_size=self.src_tile_size, resolution=(
            self.src.transform[0], -self.src.transform[4]), use_projection_units=self.use_src_metric_size, src_img=self.src)

    def load_src_vrt(self):
        """Load a source dataset's VRT into the destination CRS."""
        vrt_params = dict(crs=self.dest_crs,
                          resampling=getattr(Resampling, self.resampling),
                          src_nodata=self.nodata, dst_nodata=self.nodata)
        return WarpedVRT(self.src, **vrt_params)
