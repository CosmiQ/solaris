from ..utils.core import _check_df_load, _check_rasterio_im_load
from ..utils.core import _check_skimage_im_load
from ..utils.geo import geometries_internal_intersection, _check_wkt_load
import numpy as np
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from affine import Affine
from skimage.morphology import square, erosion, dilation


def df_to_px_mask(df, channels=['footprint'], out_file=None, reference_im=None,
                  geom_col='geometry', do_transform=False, affine_obj=None,
                  shape=(900, 900), out_type='int', burn_value=255, **kwargs):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    channels : list, optional
        The mask channels to generate. There are three values that this can
        contain:

        - ``"footprint"``: Create a full footprint mask, with 0s at pixels
            that don't fall within geometries and `burn_value` at pixels that
            do.
        - ``"boundary"``: Create a mask with geometries outlined. Use
            `boundary_width` to set how thick the boundary will be drawn.
        - ``"contact"``: Create a mask with regions between >= 2 closely
            juxtaposed geometries labeled. Use `contact_spacing` to set the
            maximum spacing between polygons to be labeled.

        Each channel correspond to its own `shape` plane in the output.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to no (``False``). If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    burn_value : `int` or `float`
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.
    kwargs
        Additional arguments to pass to `boundary_mask` or `contact_mask`. See
        those functions for requirements.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`. Shape will be
        ``(shape[0], shape[1], len(channels))``, with channels ordered per the
        provided `channels` `list`.

    """
    if isinstance(channels, str):  # e.g. if "contact", not ["contact"]
        channels = [channels]

    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')

    mask_dict = {}
    if 'footprint' in channels:
        mask_dict['footprint'] = footprint_mask(
            df=df, reference_im=reference_im, geom_col=geom_col,
            do_transform=do_transform, affine_obj=affine_obj, shape=shape,
            out_type=out_type, burn_value=burn_value
        )
    if 'boundary' in channels:
        mask_dict['boundary'] = boundary_mask(
            footprint_msk=mask_dict.get('footprint', None),
            reference_im=reference_im, geom_col=geom_col,
            boundary_width=kwargs.get('boundary_width', 3),
            boundary_type=kwargs.get('boundary_type', 'inner'),
            burn_value=burn_value, df=df, affine_obj=affine_obj,
            shape=shape, out_type=out_type
        )
    if 'contact' in channels:
        mask_dict['contact'] = contact_mask(
            df=df, reference_im=reference_im, geom_col=geom_col,
            affine_obj=affine_obj, shape=shape, out_type=out_type,
            contact_spacing=kwargs.get('contact_spacing', 10),
            burn_value=burn_value
        )

    output_arr = np.stack([mask_dict[c] for c in channels], axis=-1)

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=output_arr.shape[-1])
        meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            # I hate band indexing.
            for c in range(1, 1 + output_arr.shape[-1]):
                dst.write(output_arr[:, :, c-1], indexes=c)

    return output_arr


def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   do_transform=False, affine_obj=None, shape=(900, 900),
                   out_type='int', burn_value=255, burn_field=None):
    """Convert a dataframe of geometries to a pixel mask.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to no (``False``). If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided or if
        ``do_transform=False``.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.

    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.

    """

    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)
    df[geom_col] = df[geom_col].apply(_check_wkt_load)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                    transform=affine_obj)
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def boundary_mask(footprint_msk=None, out_file=None, reference_im=None,
                  boundary_width=3, boundary_type='inner', burn_value=255,
                  **kwargs):
    """Convert a dataframe of geometries to a pixel mask.

    Notes
    -----
    This function requires creation of a footprint mask before it can operate;
    therefore, if there is no footprint mask already present, it will create
    one. In that case, additional arguments for :func:`footprint_mask` (e.g.
    ``df``) must be passed.

    Arguments
    ---------
    footprint_msk : :class:`numpy.array`, optional
        A filled in footprint mask created using :func:`footprint_mask`. If not
        provided, one will be made by calling :func:`footprint_mask` before
        creating the boundary mask, and the required arguments for that
        function must be provided as kwargs.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored
    boundary_width : int, optional
        The width of the boundary to be created in pixels. Defaults to 3.
    boundary_type : ``"inner"`` or ``"outer"``, optional
        Where to draw the boundaries: within the object (``"inner"``) or
        outside of it (``"outer"``). Defaults to ``"inner"``.
    burn_value : `int`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    **kwargs : optional
        Additional arguments to pass to :func:`footprint_mask` if one needs to
        be created.

    Returns
    -------
    boundary_mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and the same value as the
        footprint mask `burn_value` for the boundaries of each object.

    Note: This function draws the boundaries within the edge of the object.

    """
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    # need to have a footprint mask for this function, so make it if not given
    if footprint_msk is None:
        footprint_msk = footprint_mask(reference_im=reference_im,
                                       burn_value=burn_value, **kwargs)

    # perform dilation or erosion of `footprint_mask` to get the boundary
    strel = square(boundary_width)
    if boundary_type == 'outer':
        boundary_mask = dilation(footprint_msk, strel)
    elif boundary_type == 'inner':
        boundary_mask = erosion(footprint_msk, strel)
    # use xor operator between border and footprint mask to get _just_ boundary
    boundary_mask = boundary_mask ^ footprint_msk
    # scale the `True` values to burn_value and return
    boundary_mask = boundary_mask > 0  # need to binarize to get burn val right
    output_arr = boundary_mask.astype('uint8')*burn_value

    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def contact_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                 do_transform=False, affine_obj=None, shape=(900, 900),
                 out_type='int', contact_spacing=10, burn_value=255):
    """Create a pixel mask labeling closely juxtaposed objects.

    Notes
    -----
    This function identifies pixels in an image that do not correspond to
    objects, but fall within `contact_spacing` of >1 labeled object.

    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to no (``False``). If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    contact_spacing : `int` or `float`, optional
        The desired maximum distance between adjacent polygons to be labeled
        as contact. `contact_spacing` will be in the same units as `df` 's
        geometries, not necessarily in pixel units.
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`.

    """
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)
    df[geom_col] = df[geom_col].apply(_check_wkt_load)  # load in geoms if wkt
    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
    # grow geometries by half `contact_spacing` to find overlaps
    buffered_geoms = df[geom_col].apply(lambda x: x.buffer(contact_spacing/2))
    # create a single multipolygon that covers all of the intersections
    intersect_poly = geometries_internal_intersection(buffered_geoms)
    # create a small df containing the intersections to make a footprint from
    df_for_footprint = pd.DataFrame({'shape_name': ['overlap'],
                                     'geometry': [intersect_poly]})
    # use `footprint_mask` to create the overlap mask
    contact_msk = footprint_mask(
        df_for_footprint, reference_im=reference_im, geom_col='geometry',
        do_transform=do_transform, affine_obj=affine_obj, shape=shape,
        out_type=out_type, burn_value=burn_value
        )
    footprint_msk = footprint_mask(
        df, reference_im=reference_im, geom_col=geom_col,
        do_transform=do_transform, affine_obj=affine_obj, shape=shape,
        out_type=out_type, burn_value=burn_value
        )
    contact_msk[footprint_msk > 0] = 0
    contact_msk = contact_msk > 0
    output_arr = contact_msk.astype('uint8')*burn_value

    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def mask_to_poly_geojson(mask_arr, reference_im=None, output_path=None,
                         output_type='csv', min_area=40, bg_value=0,
                         do_transform=False, simplify=False,
                         tolerance=0.5, **kwargs):
    """Get polygons from an image mask.

    Arguments
    ---------
    mask_arr : :class:`numpy.ndarray` of ints
        A 2D array of integers. Multi-channel masks are not supported, and must
        be simplified before passing to this function. Can also pass an image
        file path here.
    reference_im : str, optional
        The path to a reference geotiff to use for georeferencing the polygons
        in the mask. Required if saving to a GeoJSON (see the ``output_type``
        argument), otherwise only required if ``do_transform=True``.
    output_path : str, optional
        Path to save the output file to. If not provided, no file is saved.
    output_type : ``'csv'`` or ``'geojson'``, optional
        If ``output_path`` is provided, this argument defines what type of file
        will be generated - a CSV (``output_type='csv'``) or a geojson
        (``output_type='geojson'``).
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    bg_value : int, optional
        The value in ``mask_arr`` that denotes background (non-object).
        Defaults to ``0``.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to 0.5. Only has an effect if ``simplify=True``.

    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.

    """
    mask_arr = _check_skimage_im_load(mask_arr)
    if do_transform and reference_im is None:
        raise ValueError(
            'Coordinate transformation requires a reference image.')

    if do_transform:
        with rasterio.open(reference_im) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    else:
        transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
        crs = None

    mask = mask_arr != bg_value
    mask = mask.astype('uint8')

    polygon_generator = features.shapes(mask_arr,
                                        transform=transform,
                                        mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                   crs=crs)
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=tolerance)
            )

    return polygon_gdf
