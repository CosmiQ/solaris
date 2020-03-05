import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from solaris.data import data_dir
from solaris.utils.core import _check_df_load, _check_gdf_load
from solaris.utils.core import _check_rasterio_im_load, _check_crs
import pyproj


class TestLoadCheckers(object):
    """Test objects for checking loading of various objects."""

    def test_unloaded_geojson(self):
        geojson_path = os.path.join(data_dir, 'sample.geojson')
        truth_gdf = gpd.read_file(geojson_path)
        test_gdf = _check_gdf_load(geojson_path)

        assert truth_gdf.equals(test_gdf)

    def test_loaded_geojson(self):
        geojson_path = os.path.join(data_dir, 'sample.geojson')
        truth_gdf = gpd.read_file(geojson_path)
        test_gdf = _check_gdf_load(truth_gdf.copy())

        assert truth_gdf.equals(test_gdf)

    def test_unloaded_df(self):
        csv_path = os.path.join(data_dir, 'sample.csv')
        truth_df = pd.read_csv(csv_path)
        test_df = _check_df_load(csv_path)

        assert truth_df.equals(test_df)

    def test_loaded_df(self):
        csv_path = os.path.join(data_dir, 'sample.csv')
        truth_df = pd.read_csv(csv_path)
        test_df = _check_df_load(truth_df.copy())

        assert truth_df.equals(test_df)

    def test_unloaded_image(self):
        im_path = os.path.join(data_dir, 'sample_geotiff.tif')
        truth_im = rasterio.open(im_path)
        test_im = _check_rasterio_im_load(im_path)

        assert truth_im.profile == test_im.profile
        assert np.array_equal(truth_im.read(1), test_im.read(1))

        truth_im.close()  # need to close the rasterio datasetreader objects
        test_im.close()

    def test_loaded_image(self):
        im_path = os.path.join(data_dir, 'sample_geotiff.tif')
        truth_im = rasterio.open(im_path)
        test_im = _check_rasterio_im_load(truth_im)

        assert truth_im.profile == test_im.profile
        assert np.array_equal(truth_im.read(1), test_im.read(1))

        truth_im.close()  # need to close the rasterio datasetreader objects
        test_im.close()


class TestCRS(object):
    """Test CRS parsing."""

    def test_proj_CRS_object(self):
        input_crs = pyproj.crs.CRS.from_epsg(4326)
        crs = _check_crs(input_crs)
        assert crs == rasterio.crs.CRS.from_epsg(4326)
