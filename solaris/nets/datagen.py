from tensorflow import keras
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .transform import process_aug_dict
from ..utils.core import _check_df_load
from ..utils.io import imread, scale_for_model, _check_channel_order


def make_data_generator(framework, config, df, stage='train'):
    """Create an appropriate data generator based on the framework used.

    Arguments
    ---------
    framework : str
        One of ['keras', 'pytorch', 'simrdwn', 'tf', 'tf_obj_api'], the deep
        learning framework used for the model to be used.
    config : dict
        The config dictionary for the entire pipeline.
    df : :class:`pandas.DataFrame` or :class:`str`
        A :class:`pandas.DataFrame` containing two columns: ``'image'``, with
        the path to images for training, and ``'label'``, with the path to the
        label file corresponding to each image.

    Returns
    -------
    A Keras, PyTorch, TensorFlow, or TensorFlow Object Detection API object
    to feed data during model training or inference.
    """

    if framework.lower() not in ['keras', 'pytorch', 'torch',
                                 'simrdwn', 'tf', 'tf_obj_api']:
        raise ValueError('{} is not an accepted value for `framework`'.format(
            framework))

    # make sure the df is loaded
    df = _check_df_load(df)

    if framework.lower() == 'keras':
        return KerasSegmentationSequence(config, df, stage=stage)

    elif framework in ['torch', 'pytorch']:
        dataset = TorchDataset(config, df, stage)
        # set up workers for DataLoader for pytorch
        data_workers = config['data_specs'].get('data_workers')
        if data_workers == 1 or data_workers is None:
            data_workers = 0  # for DataLoader to run in main process
        return DataLoader(dataset, batch_size=config['batch_size'],
                          shuffle=config['training_augmentation']['shuffle'],
                          num_workers=data_workers)


class KerasSegmentationSequence(keras.utils.Sequence):
    # TODO: DOCUMENT!
    def __init__(self, config, df, stage='train'):
        self.config = config
        # TODO: IMPLEMENT LOADING IN AUGMENTATION PIPELINE HERE!
        # TODO: IMPLEMENT GETTING INPUT FILE LISTS HERE!
        self.batch_size = self.config['batch_size']
        self.df = df
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        if stage == 'train':
            self.aug = process_aug_dict(self.config['training_augmentation'])
        elif stage == 'validate':
            self.aug = process_aug_dict(self.config['validation_augmentation'])
        self.on_epoch_end()

    def on_epoch_end(self):
        'Update indices, rotations, etc. after each epoch'
        # reorder images
        self.image_indexes = np.arange(len(self.df))
        if self.config['training_augmentation']['shuffle']:
            np.random.shuffle(self.image_indexes)
    #     if self.crop:
    #         self.x_mins = np.random.randint(
    #             0, self.image_shape[1]-self.output_x, size=self.batch_size
    #         )
    #         self.y_mins = np.random.randint(
    #             0, self.image_shape[0] - self.output_y, size=self.batch_size
    #         )
    #     if self.flip_x:
    #         self.x_flips = np.random.choice(
    #             [False, True], size=self.batch_size
    #         )
    #     if self.flip_y:
    #         self.y_flips = np.random.choice(
    #             [False, True], size=self.batch_size
    #         )
    #     if self.rotate:
    #         self.n_rotations = np.random.choice(
    #             [0, 1, 2, 3], size=self.batch_size
    #         )
    #     if self.rescale_brightness is not None:
    #         self.amt_to_scale = np.random.uniform(
    #             low=self.rescale_brightness[0],
    #             high=self.rescale_brightness[1],
    #             size=self.batch_size
    #         )
    #     if self.zoom_range is not None:
    #         if (1-self.zoom_range)*self.image_shape[0] < self.output_y:
    #             self.zoom_range = self.output_y/self.image_shape[0]
    #         if (1-self.zoom_range)*self.image_shape[1] < self.output_x:
    #             self.zoom_range = self.output_x/self.image_shape[1]
    #         self.zoom_amt_y = np.random.uniform(
    #             low=1-self.zoom_range,
    #             high=1+self.zoom_range,
    #             size=self.batch_size
    #         )
    #         self.zoom_amt_x = np.random.uniform(
    #             low=1-self.zoom_range,
    #             high=1+self.zoom_range,
    #             size=self.batch_size
    #         )

    def _data_generation(self, image_idxs):
        # initialize the output array
        X = np.empty((self.batch_size,
                      self.config['data_specs']['height'],
                      self.config['data_specs']['width'],
                      self.config['data_specs']['channels']))
        if self.config['data_specs']['label_type'] == 'mask':
            y = np.empty((self.batch_size,
                          self.config['data_specs']['height'],
                          self.config['data_specs']['width'],
                          self.config['data_specs']['mask_channels']))
        else:
            pass  # TODO: IMPLEMENT BBOX LABEL SETUP HERE!
        for i in range(self.batch_size):
            im = imread(self.df['image'].iloc[image_idxs[i]])
            im = _check_channel_order(im, 'keras')
            if self.config['data_specs']['label_type'] == 'mask':
                label = imread(self.df['label'].iloc[image_idxs[i]])
                if not self.config['data_specs']['is_categorical']:
                    label[label != 0] = 1
                aug_result = self.aug(image=im, mask=label)
                # if image shape is 2D, convert to 3D
                scaled_im = scale_for_model(
                    aug_result['image'],
                    self.config['data_specs'].get('image_type')
                    )
                if len(scaled_im.shape) == 2:
                    scaled_im = scaled_im[:, :, np.newaxis]
                X[i, :, :, :] = scaled_im
                if len(aug_result['mask'].shape) == 2:
                    aug_result['mask'] = aug_result['mask'][:, :, np.newaxis]
                y[i, :, :, :] = aug_result['mask']
            else:
                pass  # TODO: IMPLEMENT BBOX LABEL LOADING HERE!

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
        return X, y


class TorchDataset(Dataset):
    """A PyTorch dataset object for segmentation/object detection.

    Arguments
    ---------
    config : dict
        The configuration dictionary for the model run.
    stage : str
        The stage of model training/inference the `TorchDataset` will be used
        for. Options are ``['train', 'validate', 'infer']``.
    """

    def __init__(self, config, df, stage='train'):
        super().__init__()
        self.df = df
        self.config = config
        self.batch_size = self.config['batch_size']
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        if stage == 'train':
            self.aug = process_aug_dict(self.config['training_augmentation'])
        elif stage == 'validate':
            self.aug = process_aug_dict(self.config['validation_augmentation'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        'Get one image:mask pair'
        # Generate indexes of the batch
        image = imread(self.df['image'].iloc[idx])
        mask = imread(self.df['label'].iloc[idx])
        if not self.config['data_specs']['is_categorical']:
            mask[mask != 0] = 1
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        sample = {'image': image, 'mask': mask}

        if self.aug:
            sample = self.aug(**sample)
        # add in additional inputs (if applicable)
        additional_inputs = self.config['data_specs'].get('additional_inputs',
                                                          None)
        if additional_inputs is not None:
            for input in additional_inputs:
                sample[input] = self.df[input].iloc[idx]

        sample['image'] = _check_channel_order(sample['image'],
                                               'torch').astype(np.float32)
        sample['mask'] = _check_channel_order(sample['mask'],
                                              'torch').astype(np.float32)
        return sample


class InferenceTiler(object):
    """An object to tile fragments of images for inference."""

    def __init__(self, framework, width, height, x_step=None, y_step=None,
                 augmentations=None):
        """Create the tiler instance."""
        self.framework = framework
        self.width = width
        self.height = height
        if x_step is None:
            self.x_step = self.width
        else:
            self.x_step = x_step
        if y_step is None:
            self.y_step = self.height
        else:
            self.y_step = y_step
        self.aug = augmentations

    def __call__(self, im):
        """Create an inference array along with an indexing reference list.

        Arguments
        ---------
        im : :class:`str` or :class:`numpy.array`
            An image to perform inference on.

        Returns
        -------
        output_arr, top_left_corner_idxs
            output_arr : ``[N, Y, X, C]`` :class:`numpy.array`
                A :class:`numpy.array` for use in model inferencing. Each
                item along the first axis corresponds to a single sample for
                the model.
            top_left_corner_idxs : :class:`list` of :class:`tuple` s of :class:`int` s
                A :class:`list` of ``(top, left)`` tuples corresponding to the
                top left corner indices of each sample along the first axis of
                ``inference_arr`` . These values can be used to stitch the
                inferencing result back together.
        """
        # read in the image if it's a path
        if isinstance(im, str):
            im = imread(im)
        # determine how many samples will be generated with the sliding window
        src_im_height = im.shape[0]
        src_im_width = im.shape[1]
        y_steps = int(1+np.ceil((src_im_height-self.height)/self.y_step))
        x_steps = int(1+np.ceil((src_im_width-self.width)/self.x_step))
        if len(im.shape) == 2:  # if there's no channel axis
            im = im[:, :, np.newaxis]  # create one - will be needed for model
        top_left_corner_idxs = []
        output_arr = []
        for y in range(y_steps):
            if self.y_step*y + self.height > im.shape[0]:
                y_min = im.shape[0] - self.height
            else:
                y_min = self.y_step*y

            for x in range(x_steps):
                if self.x_step*x + self.width > im.shape[1]:
                    x_min = im.shape[1] - self.width
                else:
                    x_min = self.x_step*x

                subarr = im[y_min:y_min + self.height,
                            x_min:x_min + self.width,
                            :]
                if self.aug is not None:
                    subarr = self.aug(image=subarr)['image']
                output_arr.append(subarr)
                top_left_corner_idxs.append((y_min, x_min))
        output_arr = np.stack(output_arr).astype(np.float32)
        if self.framework in ['torch', 'pytorch']:
            output_arr = np.moveaxis(output_arr, 3, 1)
        return output_arr, top_left_corner_idxs, (src_im_height, src_im_width)
