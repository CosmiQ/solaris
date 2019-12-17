import os
import pandas as pd
from .log import _get_logging_level
from .core import get_files_recursively
import logging


def make_dataset_csv(im_dir, im_ext='tif', label_dir=None, label_ext='json',
                     output_path='dataset.csv', stage='train', match_re=None,
                     recursive=False, ignore_mismatch=None, verbose=0):
    """Automatically generate dataset CSVs for training.

    This function creates basic CSVs for training and inference automatically.
    See `the documentation tutorials <https://solaris.readthedocs.io/en/latest/tutorials/notebooks/creating_im_reference_csvs.html>`_
    for details on the specification. A regular expression string can be
    provided to extract substrings for matching images to labels; if not
    provided, it's assumed that the filename for the image and label files is
    identical once extensions are stripped. By default, this function will
    raise an exception if there are multiple label files that match to a given
    image file, or if no label file matches an image file; see the
    `ignore_mismatch` argument for alternatives.

    Arguments
    ---------
    im_dir : str
        The path to the directory containing images to be used by your model.
        Images in sub-directories can be included by setting
        ``recursive=True``.
    im_ext : str, optional
        The file extension used by your images. Defaults to ``"tif"``. Not case
        sensitive.
    label_dir : str, optional
        The path to the directory containing images to be used by your model.
        Images in sub-directories can be included by setting
        ``recursive=True``. This argument is required if `stage` is ``"train"``
        (default) or ``"val"``, but has no effect if `stage` is ``"infer"``.
    output_path : str, optional
        The path to save the generated CSV to. Defaults to ``"dataset.csv"``.
    stage : str, optional
        The stage that the csv is generated for. Can be ``"train"`` (default),
        ``"val"``, or ``"infer"``. If set to ``"train"`` or ``"val"``,
        `label_dir` must be provided or an error will occur.
    match_re : str, optional
        A regular expression pattern to extract substrings from image and
        label filenames for matching. If not provided and labels must be
        matched to images, it's assumed that image and label filenames are
        identical after stripping directory and extension. Has no effect if
        ``stage="infer"``. The pattern must contain at least one capture group
        for compatibility with :func:`pandas.Series.str.extract`.
    recursive : bool, optional
        Should sub-directories in `im_dir` and `label_dir` be traversed to
        find images and label files? Defaults to no (``False``).
    ignore_mismatch : str, optional
        Dictates how mismatches between image files and label files should be
        handled. By default, having != 1 label file per image file will raise
        a ``ValueError``. If ``ignore_mismatch="skip"``, any image files with
        != 1 matching label will be skipped.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.

    Returns
    -------
    output_df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` with one column titled ``"image"`` and
        a second titled ``"label"`` (if ``stage != "infer"``). The function
        also saves a CSV at `output_path`.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Checking arguments.')

    if stage != 'infer' and label_dir is None:
        raise ValueError("label_dir must be provided if stage is not infer.")
    logger.info('Matching images to labels.')
    logger.debug('Getting image file paths.')
    im_fnames = get_files_recursively(im_dir, traverse_subdirs=recursive,
                                      extension=im_ext)
    logger.debug(f"Got {len(im_fnames)} image file paths.")
    temp_im_df = pd.DataFrame({'image_path': im_fnames})

    if stage != 'infer':
        logger.debug('Preparing training or validation set.')
        logger.debug('Getting label file paths.')
        label_fnames = get_files_recursively(label_dir,
                                             traverse_subdirs=recursive,
                                             extension=label_ext)
        logger.debug(f"Got {len(label_fnames)} label file paths.")
        if len(im_fnames) != len(label_fnames):
            logger.warn('The number of images and label files is not equal.')

        logger.debug("Matching image files to label files.")
        logger.debug("Extracting image filename substrings for matching.")
        temp_label_df = pd.DataFrame({'label_path': label_fnames})
        temp_im_df['image_fname'] = temp_im_df['image_path'].apply(
            lambda x: os.path.split(x)[1])
        temp_label_df['label_fname'] = temp_label_df['label_path'].apply(
            lambda x: os.path.split(x)[1])
        if match_re:
            logger.debug('match_re is True, extracting regex matches')
            im_match_strs = temp_im_df['image_fname'].str.extract(match_re)
            label_match_strs = temp_label_df['label_fname'].str.extract(
                match_re)
            if len(im_match_strs.columns) > 1 or \
                    len(label_match_strs.columns) > 1:
                raise ValueError('Multiple regex matches occurred within '
                                 'individual filenames.')
            else:
                temp_im_df['match_str'] = im_match_strs
                temp_label_df['match_str'] = label_match_strs
        else:
            logger.debug('match_re is False, will match by fname without ext')
            temp_im_df['match_str'] = temp_im_df['image_fname'].apply(
                lambda x: os.path.splitext(x)[0])
            temp_label_df['match_str'] = temp_label_df['label_fname'].apply(
                lambda x: os.path.splitext(x)[0])

        logger.debug('Aligning label and image dataframes by'
                     ' match_str.')
        temp_join_df = pd.merge(temp_im_df, temp_label_df, on='match_str',
                                how='inner')
        logger.debug(f'Length of joined dataframe: {len(temp_join_df)}')
        if len(temp_join_df) < len(temp_im_df) and \
                ignore_mismatch is None:
            raise ValueError('There is not a perfect 1:1 match of images '
                             'to label files. To allow this behavior, see '
                             'the make_dataset_csv() ignore_mismatch '
                             'argument.')
        elif len(temp_join_df) > len(temp_im_df) and ignore_mismatch is None:
            raise ValueError('There are multiple label files matching at '
                             'least one image file.')
        elif len(temp_join_df) > len(temp_im_df) and ignore_mismatch == 'skip':
            logger.info('ignore_mismatch="skip", so dropping any images with '
                        f'duplicates. Original images: {len(temp_im_df)}')
            dup_rows = temp_join_df.duplicated(subset='match_str', keep=False)
            temp_join_df = temp_join_df.loc[~dup_rows, :]
            logger.info('Remaining images after dropping duplicates: '
                        f'{len(temp_join_df)}')
        logger.debug('Dropping extra columns from output dataframe.')
        output_df = temp_join_df[['image_path', 'label_path']].rename(
            columns={'image_path': 'image', 'label_path': 'label'})

    elif stage == 'infer':
        logger.debug('Preparing inference dataset dataframe.')
        output_df = temp_im_df.rename(columns={'image_path': 'image'})

    logger.debug(f'Saving output dataframe to {output_path} .')
    output_df.to_csv(output_path, index=False)

    return output_df
