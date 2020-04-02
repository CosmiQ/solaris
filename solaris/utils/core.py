import os
import numpy as np
from shapely.wkt import loads
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
import pandas as pd
import geopandas as gpd
import pyproj
import rasterio
from distutils.version import LooseVersion
import skimage
from fiona._err import CPLE_OpenFailedError
from fiona.errors import DriverError
from warnings import warn


def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for rasterio.".format(im))


def _check_skimage_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return skimage.io.imread(im)
    elif isinstance(im, np.ndarray):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for scikit-image.".format(im))


def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")


def _check_gdf_load(gdf):
    """Check if `gdf` is already loaded in, if not, load from geojson."""
    if isinstance(gdf, str):
        # as of geopandas 0.6.2, using the OGR CSV driver requires some add'nal
        # kwargs to create a valid geodataframe with a geometry column. see
        # https://github.com/geopandas/geopandas/issues/1234
        if gdf.lower().endswith('csv'):
            return gpd.read_file(gdf, GEOM_POSSIBLE_NAMES="geometry",
                                 KEEP_GEOM_COLUMNS="NO")
        try:
            return gpd.read_file(gdf)
        except (DriverError, CPLE_OpenFailedError):
            warn(f"GeoDataFrame couldn't be loaded: either {gdf} isn't a valid"
                 " path or it isn't a valid vector file. Returning an empty"
                 " GeoDataFrame.")
            return gpd.GeoDataFrame()
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(f"{gdf} is not an accepted GeoDataFrame format.")


def _check_geom(geom):
    """Check if a geometry is loaded in.

    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)


def _check_crs(input_crs, return_rasterio=False):
    """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs


def get_data_paths(path, infer=False):
    """Get a pandas dataframe of images and labels from a csv.

    This file is designed to parse image:label reference CSVs (or just image)
    for inferencde) as defined in the documentation. Briefly, these should be
    CSVs containing two columns:

    ``'image'``: the path to images.
    ``'label'``: the path to the label file that corresponds to the image.

    Arguments
    ---------
    path : str
        Path to a .CSV-formatted reference file defining the location of
        training, validation, or inference data. See docs for details.
    infer : bool, optional
        If ``infer=True`` , the ``'label'`` column will not be returned (as it
        is unnecessary for inference), even if it is present.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` containing the relevant `image` and `label`
        information from the CSV at `path` (unless ``infer=True`` , in which
        case only the `image` column is returned.)

    """
    df = pd.read_csv(path)
    if infer:
        return df[['image']]  # no labels in those files
    else:
        return df[['image', 'label']]  # remove anything extraneous


def get_files_recursively(path, traverse_subdirs=False, extension='.tif'):
    """Get files from subdirs of `path`, joining them to the dir."""
    if traverse_subdirs:
        walker = os.walk(path)
        path_list = []
        for step in walker:
            if not step[2]:  # if there are no files in the current dir
                continue
            path_list += [os.path.join(step[0], fname)
                          for fname in step[2] if
                          fname.lower().endswith(extension)]
        return path_list
    else:
        return [os.path.join(path, f) for f in os.listdir(path)
                if f.endswith(extension)]
