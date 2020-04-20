import os
from solaris.nets.datagen import make_data_generator, InferenceTiler
from solaris.data import data_dir
from solaris.utils.io import _check_channel_order
import pandas as pd
import numpy as np
import skimage.io


class TestDataGenerator(object):
    """An object to test datagen creation from sol.nets.datagen."""

    def test_keras_sequence(self):
        """Test creating a keras sequence object for data generation."""

        dataset_csv = os.path.join(data_dir, 'datagen_sample', 'sample_df.csv')
        df = pd.read_csv(dataset_csv)
        # add the path to the directory to the values in df
        df = df.applymap(lambda x: os.path.join(data_dir, 'datagen_sample', x))

        config = {'data_specs':
                  {'height': 30,
                   'width': 30,
                   'channels': 1,
                   'dtype': None,
                   'label_type': 'mask',
                   'mask_channels': 1,
                   'is_categorical': False
                   },
                  'batch_size': 1,
                  'training_augmentation':
                  {'shuffle': True,  # images all same in test, so no effect
                   'augmentations': {}
                   }
                  }

        keras_seq = make_data_generator('keras', config, df, stage='train')
        im, mask = keras_seq.__getitem__(0)
        expected_im = skimage.io.imread(os.path.join(data_dir,
                                                     'datagen_sample',
                                                     'expected_im.tif'))
        expected_mask = skimage.io.imread(os.path.join(data_dir,
                                                       'datagen_sample',
                                                       'sample_mask_1.tif'))
        expected_mask[expected_mask != 0] = 1  # this should be binary

        assert np.array_equal(im, expected_im[np.newaxis, :, :, np.newaxis])
        assert np.array_equal(mask,
                              expected_mask[np.newaxis, :, :, np.newaxis])

    def test_torch_dataset(self):
        """Test creating a torch dataset object for data generation."""

        dataset_csv = os.path.join(data_dir, 'datagen_sample', 'sample_df.csv')
        df = pd.read_csv(dataset_csv)
        # add the path to the directory to the values in df
        df = df.applymap(lambda x: os.path.join(data_dir, 'datagen_sample', x))

        config = {'data_specs':
                  {'height': 30,
                   'width': 30,
                   'channels': 1,
                   'dtype': None,
                   'label_type': 'mask',
                   'mask_channels': 1,
                   'is_categorical': False
                   },
                  'batch_size': 1,
                  'training_augmentation':
                  {'shuffle': True,  # images all same in test, so no effect
                   'augmentations': {}
                   }
                  }

        torch_ds = make_data_generator('torch', config, df, stage='train')
        sample = next(iter(torch_ds))

        expected_im = skimage.io.imread(os.path.join(data_dir,
                                                     'datagen_sample',
                                                     'expected_im.tif'))
        expected_mask = skimage.io.imread(os.path.join(data_dir,
                                                       'datagen_sample',
                                                       'sample_mask_1.tif'))
        expected_mask[expected_mask != 0] = 1  # this should be binary
        print(sample['mask'].shape)
        assert np.array_equal(sample['image'].numpy(),
                              expected_im[np.newaxis, np.newaxis, :, :])
        assert np.array_equal(sample['mask'].numpy(),
                              expected_mask[np.newaxis, np.newaxis, :, :])


class TestInferenceTiler(object):
    """Test image tiling using sol.nets.datagen.InferenceTiler."""

    def test_simple_geotiff_tile(self):
        """Test tiling a geotiff without overlap."""
        inf_tiler = InferenceTiler('keras', 250, 250)
        tiles, tile_inds, _ = inf_tiler(os.path.join(data_dir,
                                                     'sample_geotiff.tif'))

        expected_tiles = np.load(
            os.path.join(data_dir, 'inference_tiler_test_output.npy')
            )

        expected_tile_inds = [(0, 0),
                              (0, 250),
                              (0, 500),
                              (0, 650),
                              (250, 0),
                              (250, 250),
                              (250, 500),
                              (250, 650),
                              (500, 0),
                              (500, 250),
                              (500, 500),
                              (500, 650),
                              (650, 0),
                              (650, 250),
                              (650, 500),
                              (650, 650)]

        assert np.array_equal(expected_tiles, tiles)
        assert expected_tile_inds == tile_inds
