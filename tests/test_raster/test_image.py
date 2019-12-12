import os
import numpy as np
import solaris as sol
from solaris.data import data_dir, sample_load_rasterio, sample_load_gdal
from solaris.raster.image import get_geo_transform, stitch_images
from affine import Affine
import skimage.io


class TestGetGeoTransform(object):
    """Tests for sol.raster.image.get_geo_transform()."""

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


class TestStitchImages(object):
    """Tests for image stitching with sol.raster.image.stitch_images()."""

    def test_stitch_InferenceTiler_output(self):
        inf_tiler = sol.nets.datagen.InferenceTiler('keras',
                                                    width=250, height=250)
        tiles, tile_inds, _ = inf_tiler(os.path.join(data_dir,
                                                     'sample_geotiff.tif'))
        restored_im = stitch_images(tiles, idx_refs=tile_inds,
                                    out_width=900, out_height=900)
        expected_result = sol.utils.io.imread(
            os.path.join(data_dir, 'sample_geotiff.tif'))

        assert np.array_equal(restored_im[:, :, 0], expected_result)

    def test_stitch_firstval(self):
        inf_tiler = sol.nets.datagen.InferenceTiler('keras',
                                                    width=250, height=250)
        tiles, tile_inds, _ = inf_tiler(os.path.join(data_dir,
                                                     'sample_geotiff.tif'))
        tiles[11, :, :, :] = tiles[11, :, :, :] + 10  # to have a diff to check
        result = stitch_images(tiles, idx_refs=tile_inds,
                               out_width=900, out_height=900, method='first')
        expected_result = np.load(os.path.join(data_dir,
                                               'stitching_first_output.npy'))

        assert np.array_equal(result, expected_result)

    def test_stitch_conf(self):
        src_im = skimage.io.imread(
            os.path.join(data_dir, 'sample_fp_mask_from_geojson.tif'))
        src_im[src_im != 0] = 1
        src_im = src_im.astype('float64')
        inf_tiler = sol.nets.datagen.InferenceTiler('keras',
                                                    width=250, height=250)
        tiles, tile_inds, _ = inf_tiler(src_im)
        rands = np.array([0.93794284, 0.88778908, 0.25066594, 0.76800494,
                          0.43465608, 0.69903218, 0.13256956, 0.20246324,
                          0.65134984, 0.98667763, 0.38168734, 0.85653983,
                          0.34337332, 0.75118759, 0.01128917, 0.92725672])
        # scale the values for heterogeneity
        tiles = tiles*rands[:, np.newaxis, np.newaxis, np.newaxis]
        result = stitch_images(tiles, idx_refs=tile_inds,
                               out_width=900, out_height=900,
                               method='confidence')
        expected_result = np.load(os.path.join(data_dir,
                                               'stitching_conf_output.npy'))

        assert np.array_equal(result, expected_result)
