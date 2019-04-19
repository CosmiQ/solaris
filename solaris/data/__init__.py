import os
import geopandas as gpd

# define the current directory as `data_dir`
data_dir = os.path.abspath(os.path.dirname(__file__))


def load_geojson(gj_fname):
    """Load a geojson into a gdf using GeoPandas."""
    return gpd.read_file(os.path.join(data_dir, gj_fname))


def gt_gdf():
    """Load in a ground truth GDF example."""
    return load_geojson('gt.geojson')


def pred_gdf():
    """Load in an example prediction GDF."""
    return load_geojson('pred.geojson')
