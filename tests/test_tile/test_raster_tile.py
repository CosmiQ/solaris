import os
import skimage
import numpy as np
from solaris.tile.raster_tile import Tiler
from solaris.data import data_dir


class TestTiler(object):
    def test_tiler(self):
        tiler = Tiler(os.path.join(data_dir, 'tile_test_result'),
                      src_tile_size=(90, 90))
        tiler.tile(src=os.path.join(data_dir, 'sample_geotiff.tif'))
        tiling_result_files = os.listdir(os.path.join(data_dir,
                                                      'tile_test_result'))
        assert len(tiling_result_files) == len(os.listdir(os.path.join(
            data_dir, 'tile_test_expected')))
        for f in tiling_result_files:
            result = skimage.io.imread(os.path.join(data_dir,
                                                    'tile_test_result',
                                                    f))
            expected = skimage.io.imread(os.path.join(data_dir,
                                                      'tile_test_expected',
                                                      f))
            assert np.array_equal(result, expected)
            os.remove(os.path.join(data_dir, 'tile_test_result', f))
        os.rmdir(os.path.join(data_dir, 'tile_test_result'))
