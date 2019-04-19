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
