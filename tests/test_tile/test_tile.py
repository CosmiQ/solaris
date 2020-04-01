import os
import skimage.io
import numpy as np
from solaris.tile.raster_tile import RasterTiler
from solaris.tile.vector_tile import VectorTiler
from solaris.data import data_dir
from solaris.vector.mask import geojsons_to_masks_and_fill_nodata
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
                                                'vectortile_test_custom_proj_result'))
        vector_tiler.tile(os.path.join(data_dir, 'geotiff_custom_proj_labels.geojson'),
                          raster_tiler.tile_bounds)
        vector_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'vectortile_test_custom_proj_result'))
        assert len(vector_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'vectortile_test_custom_proj_expected')))
        for f in vector_tiling_result_files:
            result = gpd.read_file(os.path.join(data_dir,
                                                'vectortile_test_custom_proj_result',
                                                f))
            expected = gpd.read_file(os.path.join(data_dir,
                                                  'vectortile_test_custom_proj_expected',
                                                  f))
            if len(result) == 0:
                assert len(expected) == 0
            else:
                result = cascaded_union(result.geometry)
                expected = cascaded_union(expected.geometry)
                assert result.intersection(expected).area/result.area > 0.99999
            os.remove(os.path.join(data_dir, 'vectortile_test_custom_proj_result', f))
        os.rmdir(os.path.join(data_dir, 'vectortile_test_custom_proj_result'))

    def test_tiler_fill_nodata(self):
        # get non filled tiles
        bounds_gdf= gpd.read_file(os.path.join(data_dir, "restrict_aoi_test.geojson"))
        bounds_poly = bounds_gdf['geometry'].iloc[0]
        raster_tiler = RasterTiler(os.path.join(data_dir,
                                                'rastertile_test_fill_nodata_result'),
                                   src_tile_size=(128, 128),
                                   nodata= -9999.0,
                                   aoi_boundary=bounds_poly)
        raster_tiler.tile(src=os.path.join(data_dir, 'nebraska_landsat5_with_nodata_wgs84.tif'), restrict_to_aoi=True)
        
        vector_tiler = VectorTiler(os.path.join(data_dir,
                                                'vectortile_test_nonfilled_result'))

        vector_tiler.tile(os.path.join(data_dir, 'nebraska_wgs84_with_nodata_labels.geojson'),
                          tile_bounds = raster_tiler.tile_bounds)
        vector_tiling_result_files_nonfilled = os.listdir(os.path.join(
            data_dir, 'vectortile_test_nonfilled_result'))

        # fills nodata in imagery and then fills same no data pixels in rasterized labels
        geojsons_to_masks_and_fill_nodata(raster_tiler, vector_tiler, 
            os.path.join(data_dir, "vectortile_test_filled_result"), fill_value=0)

        # list the results
        vector_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'vectortile_test_filled_result'))
        assert len(vector_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'vectortile_test_filled_expected')))

        raster_tiling_result_files = os.listdir(os.path.join(
            data_dir, 'rastertile_test_fill_nodata_result'))
        assert len(raster_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'rastertile_test_fill_nodata_expected')))

        vector_tiling_result_files_nonfilled = os.listdir(os.path.join(
            data_dir, 'vectortile_test_nonfilled_result'))
        assert len(vector_tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'vectortile_test_nonfilled_expected')))
        
        # check if the filling worked for both img and vector tiles
        for f in raster_tiling_result_files:
            result = skimage.io.imread(os.path.join(data_dir,
                                                    'rastertile_test_fill_nodata_result',
                                                    f))
            expected = skimage.io.imread(
                os.path.join(data_dir, 'rastertile_test_fill_nodata_expected', f))
            assert np.array_equal(result, expected)


        for f in vector_tiling_result_files:
            result = skimage.io.imread(os.path.join(data_dir,
                                                'vectortile_test_filled_result',
                                                f))
            expected = skimage.io.imread(os.path.join(data_dir,
                                                  'vectortile_test_filled_expected',
                                                  f))
            assert np.array_equal(result, expected)

        #cleanup
        for f in vector_tiling_result_files_nonfilled:
            os.remove(os.path.join(data_dir, 'vectortile_test_nonfilled_result', f))
        os.rmdir(os.path.join(data_dir, 'vectortile_test_nonfilled_result'))
        for f in vector_tiling_result_files:
            os.remove(os.path.join(data_dir, 'vectortile_test_filled_result', f))
        os.rmdir(os.path.join(data_dir, 'vectortile_test_filled_result'))
        for f in raster_tiling_result_files:
            os.remove(os.path.join(data_dir, 'rastertile_test_fill_nodata_result', f))
        os.rmdir(os.path.join(data_dir, 'rastertile_test_fill_nodata_result'))

