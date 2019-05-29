import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import skimage


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
        raise ValueError("{} is not an accepted DataFrame format.".format(df))


def _check_gdf_load(gdf):
    """Check if `gdf` is already loaded in, if not, load from geojson."""
    if isinstance(gdf, str):
        return gpd.read_file(gdf)
    elif isinstance(gdf, gpd.GeoDataFrame):
        return gdf
    else:
        raise ValueError(
            "{} is not an accepted GeoDataFrame format.".format(gdf))


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


def get_files_recursively(image_path, traverse_subdirs=False):
    """Get files from subdirs of `path`, joining them to the dir."""
    if traverse_subdirs:
        walker = os.walk(image_path)
        im_path_list = []
        for step in walker:
            if not step[2]:  # if there are no files in the current dir
                continue
            im_path_list += [os.path.join(step[0], fname)
                             for fname in step[2] if
                             fname.endswith('.tif')]
        return im_path_list
    else:
        return [f for f in os.listdir(image_path)
                if f.endswith('.tif')]
