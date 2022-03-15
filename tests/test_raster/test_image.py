import os

from affine import Affine

from solaris.data import sample_load_rasterio
from solaris.raster.image import get_geo_transform

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))


class TestGetGeoTransform(object):
    """Tests for solaris.raster.image.get_geo_transform()."""

    def test_get_from_file(self):
        affine_obj = get_geo_transform(os.path.join(data_dir, "sample_geotiff.tif"))
        assert affine_obj == Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)

    def test_get_from_opened_raster(self):
        src_obj = sample_load_rasterio()
        affine_obj = get_geo_transform(src_obj)
        assert affine_obj == Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
        src_obj.close()
