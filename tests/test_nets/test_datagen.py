import os
from solaris.nets.datagen import make_data_generator
from solaris.data import data_dir
import pandas as pd
import numpy as np
import skimage


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
                   'label_type': 'mask',
                   'mask_channels': 1
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
                   'label_type': 'mask',
                   'mask_channels': 1
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

        assert np.array_equal(sample['image'],
                              expected_im[np.newaxis, :, :, np.newaxis])
        assert np.array_equal(sample['label'],
                              expected_mask[np.newaxis, :, :, np.newaxis])
