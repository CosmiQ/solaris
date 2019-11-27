import os
import skimage.io
import numpy as np
from solaris.tile.raster_tile import RasterTiler
from solaris.tile.vector_tile import VectorTiler
from solaris.data import data_dir
import geopandas as gpd
from shapely.ops import cascaded_union


class TestTilers(object):
    def test_tiler(self):
        raster_tiler = RasterTiler(os.path.join(data_dir,
                                                'rastertile_test_result'),
                                   src_tile_size=(90, 90))
        raster_tiler.tile(src=os.path.join(data_dir, 'sample_geotiff.tif'))
        raster_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'rastertile_test_result'))
        assert len(raster_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'rastertile_test_expected')))
        for f in raster_tiling_result_files:
            result = skimage.io.imread(os.path.join(data_dir,
                                                    'rastertile_test_result',
                                                    f))
            expected = skimage.io.imread(
                os.path.join(data_dir, 'rastertile_test_expected', f))
            assert np.array_equal(result, expected)
            os.remove(os.path.join(data_dir, 'rastertile_test_result', f))
        os.rmdir(os.path.join(data_dir, 'rastertile_test_result'))
        vector_tiler = VectorTiler(os.path.join(data_dir,
                                                'vectortile_test_result'))
        vector_tiler.tile(os.path.join(data_dir, 'geotiff_labels.geojson'),
                          raster_tiler.tile_bounds)
        vector_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'vectortile_test_result'))
        assert len(vector_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'vectortile_test_expected')))
        for f in vector_tiling_result_files:
            result = gpd.read_file(os.path.join(data_dir,
                                                'vectortile_test_result',
                                                f))
            expected = gpd.read_file(os.path.join(data_dir,
                                                  'vectortile_test_expected',
                                                  f))
            if len(result) == 0:
                assert len(expected) == 0
            else:
                result = cascaded_union(result.geometry)
                expected = cascaded_union(expected.geometry)
                assert result.intersection(expected).area/result.area > 0.99999
            os.remove(os.path.join(data_dir, 'vectortile_test_result', f))
        os.rmdir(os.path.join(data_dir, 'vectortile_test_result'))

    def test_tiler_custom_proj(self):
        raster_tiler = RasterTiler(os.path.join(data_dir,
                                                'rastertile_test_custom_proj_result'),
                                   src_tile_size=(128, 128))
        raster_tiler.tile(src=os.path.join(data_dir, 'sample_geotiff_custom_proj.tif'))
        raster_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'rastertile_test_custom_proj_result'))
        assert len(raster_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'rastertile_test_custom_proj_expected')))
        for f in raster_tiling_result_files:
            result = skimage.io.imread(os.path.join(data_dir,
                                                    'rastertile_test_custom_proj_result',
                                                    f))
            expected = skimage.io.imread(
                os.path.join(data_dir, 'rastertile_test_custom_proj_expected', f))
            assert np.array_equal(result, expected)
            os.remove(os.path.join(data_dir, 'rastertile_test_custom_proj_result', f))
        os.rmdir(os.path.join(data_dir, 'rastertile_test_custom_proj_result'))
        vector_tiler = VectorTiler(os.path.join(data_dir,
                                                'vectortile_test_result'))
        vector_tiler.tile(os.path.join(data_dir, 'geotiff_labels.geojson'),
                          raster_tiler.tile_bounds)
        vector_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'vectortile_test_result'))
        assert len(vector_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'vectortile_test_expected')))
        for f in vector_tiling_result_files:
            result = gpd.read_file(os.path.join(data_dir,
                                                'vectortile_test_result',
                                                f))
            expected = gpd.read_file(os.path.join(data_dir,
                                                  'vectortile_test_expected',
                                                  f))
            if len(result) == 0:
                assert len(expected) == 0
            else:
                result = cascaded_union(result.geometry)
                expected = cascaded_union(expected.geometry)
                assert result.intersection(expected).area/result.area > 0.99999
            os.remove(os.path.join(data_dir, 'vectortile_test_result', f))
        os.rmdir(os.path.join(data_dir, 'vectortile_test_result'))
