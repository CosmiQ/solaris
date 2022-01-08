import os

import geopandas as gpd
import pandas as pd
import rasterio

data_dir = os.path.dirname(__file__) + "/../../tests/data"


def load_geojson(gj_fname):
    """Load a geojson into a gdf using GeoPandas."""
    return gpd.read_file(os.path.join(data_dir, gj_fname))


def gt_gdf():
    """Load in a ground truth GDF example."""
    return load_geojson("gt.geojson")


def pred_gdf():
    """Load in an example prediction GDF."""
    return load_geojson("pred.geojson")


def sample_load_rasterio():
    return rasterio.open(os.path.join(data_dir, "sample_geotiff.tif"))


def sample_load_geojson():
    return gpd.read_file(os.path.join(data_dir, "sample.geojson"))


def sample_load_csv():
    return pd.read_file(os.path.join(data_dir, "sample.csv"))
