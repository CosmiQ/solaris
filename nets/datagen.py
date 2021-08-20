from tensorflow import keras
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from .transform import _check_augs, process_aug_dict
from ..utils.core import _check_df_load
from ..utils.geo import split_geom
from ..utils.io import imread, _check_channel_order


def make_data_generator(framework, config, df, stage='train'):
    """Create an appropriate data generator based on the framework used.

    A wrapper for the high-end ``solaris`` API to create data generators.
    Using the ``config`` dictionary, this function creates an instance of
    either :class:`KerasSegmentationSequence` or :class:`TorchDataset`
    (depending on the framework used for the pipeline). If using Torch, this
    instance is then wrapped in a :class:`torch.utils.data.DataLoader` and
    returned; if Keras, the sequence object is directly returned.

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
    stage : str, optional
        Either ``'train'`` or ``'validate'``, indicates whether the object
        created is being used for training or validation. This determines which
        augmentations from the config file are applied within the returned
        object.

    Returns
    -------
    data_gen : :class:`KerasSegmentationSequence` or :class:`torch.utils.data.DataLoader`
        An object to pass data into the :class:`solaris.nets.train.Trainer`
        instance during model training.

    See Also
    --------
    :class:`KerasSegmentationSequence`
    :class:`TorchDataset`
    :class:`InferenceTiler`
    """

    if framework.lower() not in ['keras', 'pytorch', 'torch']:
        raise ValueError('{} is not an accepted value for `framework`'.format(
            framework))

    # make sure the df is loaded
    df = _check_df_load(df)

    if stage == 'train':
        augs = config['training_augmentation']
        shuffle = config['training_augmentation']['shuffle']
    elif stage == 'validate':
        augs = config['validation_augmentation']
        shuffle = False
    try:
        num_classes = config['data_specs']['num_classes']
    except KeyError:
        num_classes = 1

    if framework.lower() == 'keras':
        data_gen = KerasSegmentationSequence(
            df,
            height=config['data_specs']['height'],
            width=config['data_specs']['width'],
            input_channels=config['data_specs']['channels'],
            output_channels=config['data_specs']['mask_channels'],
            augs=augs,
            batch_size=config['batch_size'],
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            shuffle=shuffle)

    elif framework in ['torch', 'pytorch']:
        dataset = TorchDataset(
            df,
            augs=augs,
            batch_size=config['batch_size'],
            label_type=config['data_specs']['label_type'],
            is_categorical=config['data_specs']['is_categorical'],
            num_classes=num_classes,
            dtype=config['data_specs']['dtype'])
        # set up workers for DataLoader for pytorch
        data_workers = config['data_specs'].get('data_workers')
        if data_workers == 1 or data_workers is None:
            data_workers = 0  # for DataLoader to run in main process
        data_gen = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=config['training_augmentation']['shuffle'],
            num_workers=data_workers)

    return data_gen


class KerasSegmentationSequence(keras.utils.Sequence):
    """An object to stream images from files into a Keras model in solaris.


    Attributes
    ----------
    df : :class:`pandas.DataFrame`
        The :class:`pandas.DataFrame` specifying where inputs are stored.
    height : int
        The height of generated images.
    width : int
        The width of generated images.
    input_channels : int
        The number of channels in generated inputs.
    output_channels : int
        The number of channels in target masks created.
    aug : :class:`albumentations.core.composition.Compose`
        An albumentations Compose object to pass imagery through before
        passing it into the neural net. If an augmentation config subdict
        was provided during initialization, this is created by parsing the
        dict with :func:`solaris.nets.transform.process_aug_dict`.
    batch_size : int
        The batch size generated.
    n_batches : int
        The number of batches per epoch. Inferred based on the number of
        input files in `df` and `batch_size`.
    label_type : str
        Type of labels. Currently always ``"mask"``.
    is_categorical : bool
        Indicates whether masks output are boolean or categorical labels.
    num_classes: int
        Indicates the number of classes in the dataset
    shuffle : bool
        Indicates whether or not input order is shuffled for each epoch.
    """

    def __init__(self, df, height, width, input_channels, output_channels,
                 augs, batch_size, label_type='mask', is_categorical=False,
                 num_classes=1, shuffle=True):
        """Create an instance of KerasSegmentationSequence.

        Arguments
        ---------
        df : :class:`pandas.DataFrame`
            A pandas DataFrame specifying images and label files to read into
            the model. See `the reference file creation tutorial`_ for more.
        height : int
            The height of model inputs in pixels.
        width : int
            The width of model inputs in pixels.
        input_channels : int
            The number of channels in model input imagery.
        output_channels : int
            The number of channels in the model output.
        augs : :class:`dict` or :class:`albumentations.core.composition.Compose`
            Either the config subdict specifying augmentations to apply, or
            a pre-created :class:`albumentations.core.composition.Compose` object
            containing all of the augmentations to apply.
        batch_size : int
            The number of samples in a training batch.
        label_type : str, optional
            The type of labels to be used. At present, only ``"mask"`` is
            supported.
        is_categorical : bool, optional
            Is the data categorical or boolean (default)?
        num_classes: int
            Indicates the number of classes in the dataset
        shuffle : bool, optional
            Should image order be shuffled in each epoch?


        .. _the reference file creation tutorial: https://solaris.readthedocs.io/en/latest/tutorials/notebooks/creating_im_reference_csvs.html
        """

        # TODO: IMPLEMENT GETTING INPUT FILE LISTS HERE!
        self.df = df
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.aug = _check_augs(augs)  # checks if they're loaded; loads if not
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.label_type = label_type
        self.is_categorical = is_categorical
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        """Update indices after each epoch."""
        # reorder images
        self.image_indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.image_indexes)

    def _data_generation(self, image_idxs):
        # initialize the output array
        X = np.empty((self.batch_size,
                      self.height,
                      self.width,
                      self.input_channels))
        if self.label_type == 'mask':
            y = np.empty((self.batch_size,
                          self.height,
                          self.width,
                          self.output_channels))
        else:
            pass  # TODO: IMPLEMENT BBOX LABEL SETUP HERE!
        for i in range(self.batch_size):
            im = imread(self.df['image'].iloc[image_idxs[i]])
            im = _check_channel_order(im, 'keras')
            if self.label_type == 'mask':
                label = imread(self.df['label'].iloc[image_idxs[i]])
                if not self.is_categorical:
                    label[label != 0] = 1
                aug_result = self.aug(image=im, mask=label)
                # if image shape is 2D, convert to 3D
                if len(aug_result['image'].shape) == 2:
                    aug_result['image'] = aug_result['image'][:, :, np.newaxis]
                X[i, :, :, :] = aug_result['image']
                if len(aug_result['mask'].shape) == 2:
                    aug_result['mask'] = aug_result['mask'][:, :, np.newaxis]
                y[i, :, :, :] = aug_result['mask']
            else:
                raise NotImplementedError(
                    'Usage of non-mask labels is not implemented yet.')

        return X, y

    def __len__(self):
        """Denotes the number of batches per epoch.

        This is a required method for Keras Sequence objects.
        """
        return self.n_batches

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        im_inds = self.image_indexes[index*self.batch_size:
                                     (index+1)*self.batch_size]

        # Generate data
        X, y = self._data_generation(image_idxs=im_inds)
        return X, y


class TorchDataset(Dataset):
    """A PyTorch dataset object for solaris.

    Note that this object is wrapped in a :class:`torch.utils.data.DataLoader`
    before being passed to the :class:solaris.nets.train.Trainer` instance.

    Attributes
    ----------
    df : :class:`pandas.DataFrame`
        The :class:`pandas.DataFrame` specifying where inputs are stored.
    aug : :class:`albumentations.core.composition.Compose`
        An albumentations Compose object to pass imagery through before
        passing it into the neural net. If an augmentation config subdict
        was provided during initialization, this is created by parsing the
        dict with :func:`solaris.nets.transform.process_aug_dict`.
    batch_size : int
        The batch size generated.
    n_batches : int
        The number of batches per epoch. Inferred based on the number of
        input files in `df` and `batch_size`.
    dtype : :class:`numpy.dtype`
        The numpy dtype that image inputs should be when passed to the model.
    is_categorical : bool
        Indicates whether masks output are boolean or categorical labels.
    num_classes: int
        Indicates the number of classes in the dataset
    dtype : class:`numpy.dtype`
        The data type images should be converted to before being passed to
        neural nets.
    """

    def __init__(self, df, augs, batch_size, label_type='mask',
                 is_categorical=False, num_classes=1, dtype=None):
        """
        Create an instance of TorchDataset for use in model training.

        Arguments
        ---------
        df : :class:`pandas.DataFrame`
            A pandas DataFrame specifying images and label files to read into
            the model. See `the reference file creation tutorial`_ for more.
        augs : :class:`dict` or :class:`albumentations.core.composition.Compose`
            Either the config subdict specifying augmentations to apply, or
            a pre-created :class:`albumentations.core.composition.Compose`
            object containing all of the augmentations to apply.
        batch_size : int
            The number of samples in a training batch.
        label_type : str, optional
            The type of labels to be used. At present, only ``"mask"`` is
            supported.
        is_categorical : bool, optional
            Is the data categorical or boolean (default)?
        num_classes: int
            Indicates the number of classes in the dataset
        dtype : str, optional
            The dtype that image arrays should be converted to before being
            passed to the neural net. If not provided, defaults to
            ``"float32"``. Must be one of the `numpy dtype options`_.

        .. _numpy dtype options: https://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        super().__init__()

        self.df = df
        self.batch_size = batch_size
        self.n_batches = int(np.floor(len(self.df)/self.batch_size))
        self.aug = _check_augs(augs)
        self.is_categorical = is_categorical
        self.num_classes = num_classes

        if dtype is None:
            self.dtype = np.float32  # default
        # if it's a string, get the appropriate object
        elif isinstance(dtype, str):
            try:
                self.dtype = getattr(np, dtype)
            except AttributeError:
                raise ValueError(
                    'The data type {} is not supported'.format(dtype))
        # lastly, check if it's already defined in the right format for use
        elif issubclass(dtype, np.number) or isinstance(dtype, np.dtype):
            self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get one image, mask pair"""
        # Generate indexes of the batch
        image = imread(self.df['image'].iloc[idx])
        mask = imread(self.df['label'].iloc[idx])
        if not self.is_categorical:
            mask[mask != 0] = 1
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        sample = {'image': image, 'mask': mask}

        if self.aug:
            sample = self.aug(**sample)
        # add in additional inputs (if applicable)
        # additional_inputs = self.config['data_specs'].get('additional_inputs',
        #                                                   None)
        # if additional_inputs is not None:
        #     for input in additional_inputs:
        #         sample[input] = self.df[input].iloc[idx]

        sample['image'] = _check_channel_order(sample['image'],
                                               'torch').astype(self.dtype)
        sample['mask'] = _check_channel_order(sample['mask'],
                                              'torch').astype(np.float32)
        return sample


class InferenceTiler(object):
    """An object to tile fragments of images for inference.

    This object allows you to pass images of arbitrary size into Solaris for
    inference, similar to the pre-existing CosmiQ Works tool, BASISS_. The
    object will step across an input image creating tiles of size
    ``[height, width]``, taking steps of size ``[y_step, x_step]`` as it goes.
    When it reaches an edge, it will take tiles from ``-height`` or ``-width``
    to the edge. Clearly, these can overlap with one another; the intention
    is that overlaps will be resolved using
    :func:`solaris.raster.image.stitch_images` when re-creating the output.

    .. _BASISS: https://github.com/cosmiq/basiss

    Attributes
    ----------
    framework : str
        The deep learning framework used. Can be one of ``"torch"``,
        ``"pytorch"``, or ``"keras"``.
    width : int
        The width of images to load into the neural net.
    height : int
        The height of images to load into the neural net.
    x_step : int, optional
        The step size taken in the x direction when sampling for new images.
    y_step : int, optional
        The step size taken in the y direction when sampling for new images.
    aug : :class:`albumentations.core.composition.Compose`
        Augmentations to apply before passing to a neural net. Generally used
        for pre-processing.

    See Also
    --------
    :func:`solaris.raster.image.stitch_images`
    :func:`make_data_generator`
    """

    def __init__(self, framework, width, height, x_step=None, y_step=None,
                 augmentations=None):
        """Create the tiler instance.

        Arguments
        ---------
        framework : str
            The deep learning framework used. Can be one of ``"torch"``,
            ``"pytorch"``, or ``"keras"``.
        width : int
            The width of images to load into the neural net.
        height : int
            The height of images to load into the neural net.
        x_step : int, optional
            The step size taken in the x direction when sampling for new
            images. If not provided, defaults to `width`.
        y_step : int, optional
            The step size taken in the y direction when sampling for new images.
            If not provided, defaults to `height`.
        aug : :class:`albumentations.core.composition.Compose`
            Augmentations to apply before passing to a neural net. Generally used
            for pre-processing.
        """
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
        self.aug = _check_augs(augmentations)

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
