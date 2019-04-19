import os
from cw_geodata.data import data_dir, sample_load_rasterio, sample_load_gdal
from cw_geodata.raster_image.image import get_geo_transform
from affine import Affine


class TestGetGeoTransform(object):
    """Tests for cw_geodata.raster_image.image.get_geo_transform."""

    def test_get_from_file(self):
        affine_obj = get_geo_transform(os.path.join(data_dir,
                                                    'sample_geotiff.tif'))
        assert affine_obj == Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)

    def test_get_from_opened_raster(self):
        src_obj = sample_load_rasterio()
        affine_obj = get_geo_transform(src_obj)
        assert affine_obj == Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
        src_obj.close()

    def test_get_from_gdal(self):
        src_obj = sample_load_gdal()
        affine_obj = get_geo_transform(src_obj)
        assert affine_obj == Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
