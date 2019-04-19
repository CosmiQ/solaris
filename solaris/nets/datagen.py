import keras
import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io


def DataGenerator(framework, image_path, mask_path, **kwargs):
    """Create an appropriate data generator based on the framework used.

    Arguments
    ---------
    framework : str
        One of ['keras', 'pytorch', 'simrdwn', 'tf', 'tf_obj_api'], the deep
        learning framework used for the model to be used.
    image_path : str
        Path to the images to feed to the deep learning model.
    mask_path : str
        Path to the masks to feed to the deep learning model.
    **kwargs
        Additional arguments passed to framework-specific generators. See the
        arguments to the framework-specific datasets and data generators for
        details.

    Returns
    -------
    A Keras, PyTorch, TensorFlow, or TensorFlow Object Detection API object
    to feed data during model training or inference.
    """

    if framework not in ['keras', 'pytorch', 'simrdwn', 'tf', 'tf_obj_api']:
        raise ValueError('{} is not an accepted value for `framework`'.format(
            framework))
    if framework == 'keras':
        if 'image_shape' not in kwargs:
            raise ValueError('`image_shape` must be provided as an argument ' +
                             'when using Keras.')
        return FileDataGenerator(image_path, mask_path, **kwargs)


class FileDataGenerator(keras.utils.Sequence):
    # TODO: DOCUMENT!
    def __init__(self, image_paths, mask_path, image_shape,
                 traverse_subdirs=False, chip_subset=[], batch_size=32,
                 crop=False, output_x=256, output_y=256, shuffle=True,
                 flip_x=False, flip_y=False, zoom_range=None,
                 rotate=False, rescale_brightness=None, output_dir=''):
        self.traverse_subdirs = traverse_subdirs
        self.mask_path = mask_path
        self.mask_list = [f for f in os.listdir(mask_path)
                          if f.endswith('.tif')]
        self.image_list = image_paths
        if chip_subset:
            # subset the raw mask and image lists based on a list of chips
            # provided as chip_subset
            self.image_list = [f for f in self.image_list if any(
                chip in f for chip in chip_subset
                )]
            self.mask_list = [os.path.join(self.mask_path, f)
                              for f in self.mask_list if any(
                                  chip in f for chip in chip_subset
                                  )]
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.image_list) /
                                      self.batch_size))
        self.output_x = output_x
        self.output_y = output_y
        self.crop = crop
        self.shuffle = shuffle
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.rotate = rotate
        self.zoom_range = zoom_range
        self.output_dir = output_dir
        self.output_ctr = 0
        self.rescale_brightness = rescale_brightness
        self.on_epoch_end()

    def on_epoch_end(self):
        'Update indices, rotations, etc. after each epoch'
        # reorder images
        self.image_indexes = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.image_indexes)
        if self.crop:
            self.x_mins = np.random.randint(
                0, self.image_shape[1]-self.output_x, size=self.batch_size
            )
            self.y_mins = np.random.randint(
                0, self.image_shape[0] - self.output_y, size=self.batch_size
            )
        if self.flip_x:
            self.x_flips = np.random.choice(
                [False, True], size=self.batch_size
            )
        if self.flip_y:
            self.y_flips = np.random.choice(
                [False, True], size=self.batch_size
            )
        if self.rotate:
            self.n_rotations = np.random.choice(
                [0, 1, 2, 3], size=self.batch_size
            )
        if self.rescale_brightness is not None:
            self.amt_to_scale = np.random.uniform(
                low=self.rescale_brightness[0],
                high=self.rescale_brightness[1],
                size=self.batch_size
            )
        if self.zoom_range is not None:
            if (1-self.zoom_range)*self.image_shape[0] < self.output_y:
                self.zoom_range = self.output_y/self.image_shape[0]
            if (1-self.zoom_range)*self.image_shape[1] < self.output_x:
                self.zoom_range = self.output_x/self.image_shape[1]
            self.zoom_amt_y = np.random.uniform(
                low=1-self.zoom_range,
                high=1+self.zoom_range,
                size=self.batch_size
            )
            self.zoom_amt_x = np.random.uniform(
                low=1-self.zoom_range,
                high=1+self.zoom_range,
                size=self.batch_size
            )

    def _data_generation(self, image_idxs):
        # initialize
        X = np.empty((self.batch_size, self.output_y, self.output_x,
                      self.image_shape[2]))
        # TODO: IMPLEMENT MULTI-CHANNEL MASK FUNCTIONALITY
        y = np.empty((self.batch_size, self.output_y, self.output_x, 1))
        for i in range(self.batch_size):
            im_path = self.image_list[image_idxs[i]]
            # TODO: IMPLEMENT BETTER REGEX-BASED CHIP ID SEARCHING
            if im_path.endswith('_image.tif'):
                chip_id = '_'.join(im_path.rstrip('_image.tif').split('_')[-2:])
            else:
                chip_id = '_'.join(im_path.rstrip('.tif').split('_')[-2:])
            mask_path = [f for f in self.mask_list if chip_id in f][0]
            im_arr = cv2.imread(im_path, cv2.IMREAD_COLOR)
            mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_arr = mask_arr[:, :, np.newaxis] > 0
            if self.zoom_range is not None:
                im_arr = cv2.resize(
                    im_arr,
                    (int(im_arr.shape[1]*self.zoom_amt_x[i]),
                     int(im_arr.shape[0]*self.zoom_amt_y[i])))
                mask_arr = cv2.resize(
                    mask_arr.astype('uint8'),
                    (int(mask_arr.shape[1]*self.zoom_amt_x[i]),
                     int(mask_arr.shape[0]*self.zoom_amt_y[i])))
                if len(mask_arr.shape) < 3:  # add third axis if absent
                    mask_arr = mask_arr[:, :, np.newaxis]
                mask_arr = mask_arr > 0
                pad_amt = [0, 0]
                if self.zoom_amt_y[i] < 1:
                    pad_amt[0] = int(self.image_shape[0] *
                                     self.zoom_amt_y[i]*0.5)
                if self.zoom_amt_x[i] < 1:
                    pad_amt[1] = int(self.image_shape[1] *
                                     self.zoom_amt_x[i]*0.5)
                if pad_amt != [0, 0]:
                    mask_arr = np.pad(
                        mask_arr,
                        pad_width=((pad_amt[0], pad_amt[0]),
                                   (pad_amt[1], pad_amt[1]),
                                   (0, 0)),
                        mode='reflect')
                    im_arr = np.pad(
                        im_arr,
                        pad_width=((pad_amt[0], pad_amt[0]),
                                   (pad_amt[1], pad_amt[1]),
                                   (0, 0)),
                        mode='reflect')
            if self.crop:
                im_arr = im_arr[self.y_mins[i]:self.y_mins[i]+self.output_y,
                                self.x_mins[i]:self.x_mins[i]+self.output_x,
                                :]
                mask_arr = mask_arr[
                    self.y_mins[i]:self.y_mins[i]+self.output_y,
                    self.x_mins[i]:self.x_mins[i]+self.output_x,
                    :]
            else:
                im_arr = cv2.resize(im_arr, (self.output_y, self.output_x,
                                             self.image_shape[2]))
                mask_arr = cv2.resize(im_arr, (self.output_y, self.output_x,
                                               1))
            if self.flip_x:
                if self.x_flips[i]:
                    im_arr = np.flip(im_arr, axis=0)
                    mask_arr = np.flip(mask_arr, axis=0)
            if self.flip_y:
                if self.y_flips[i]:
                    im_arr = np.flip(im_arr, axis=1)
                    mask_arr = np.flip(mask_arr, axis=1)
            if self.rotate:
                to_go = 0
                while to_go < self.n_rotations[i]:
                    im_arr = np.rot90(im_arr)
                    mask_arr = np.rot90(mask_arr)
                    to_go += 1
            if self.rescale_brightness is not None:
                hsv = cv2.cvtColor(im_arr, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]*self.amt_to_scale[i]
                v = np.clip(v, 0, 255).astype('uint8')
                hsv[:, :, 2] = v
                im_arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            X[i, :, :, :] = im_arr
            y[i, :, :, :] = mask_arr
        X = X/255.
        return X, y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        im_inds = self.image_indexes[index*self.batch_size:
                                     (index+1)*self.batch_size]

        # Generate data
        X, y = self._data_generation(image_idxs=im_inds)
        if self.output_dir:
            np.save(os.path.join(
                self.output_dir, 'images_{}.npy'.format(self.output_ctr)),
                    X)
            np.save(os.path.join(
                self.output_dir, 'masks_{}.npy'.format(self.output_ctr)),
                    y)
            self.output_ctr += 1
        return X, y


class TorchDataset(Dataset):
    """A PyTorch dataset object for segmentation/object detection.

    Arguments
    ---------
    reference_csv : str
        Path to a csv file with at least two columns: ``'image_path'``
        that denotes the paths to the source imagery and ``'label_path'``
        that denotes the paths to their corresponding labels.
    aug_pipeline : :py:class:`torchvision.transforms` or :py:class:`albumentations.core.transforms_interface.Compose`
        An augmentation pipeline object compatible with PyTorch. Not required
         if not performing any augmentation or post-processing on the image.
    """
    def __init__(self, reference_csv, aug_pipeline=None):
        super(self, TorchDataset).__init__()
        self.dataset_ref = pd.read_csv(reference_csv)
        self.aug_pipeline = aug_pipeline

    def __len__(self):
        return len(self.dataset_ref)

    def __getitem__(self, idx):
        img_fname = self.dataset_ref['image_path'].iloc[idx]
        label_fname = self.dataset_ref['label_path'].iloc[idx]
        image = io.imread(img_fname)
        if os.path.splitext(label_fname)[1].lower() in ['gif', 'tiff', 'tif',
                                                        'geotiff' 'png',
                                                        'jpg']:
            label = io.imread(label_fname)
        else:
            raise NotImplementedError(
                'Non-image labels are not currently implemented.')
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_files_recursively(image_path, traverse_subdirs=False):
    """Get files from subdirs of `path`, joining them to the dir."""
    if traverse_subdirs:
        walker = os.walk(image_path)
        im_path_list = []
        for step in walker:
            if not step[2]:  # if there are no files in the current dir
                continue
            im_path_list += [os.path.join(step[0], fname)
                             for fname in step[2] if
                             fname.endswith('.tif')]
        return im_path_list
    else:
        return [f for f in os.listdir(image_path)
                if f.endswith('.tif')]
