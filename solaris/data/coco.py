from ..utils.core import _check_df_load, _check_geom, get_files_recursively
from ..utils.geo import bbox_corners_to_coco, polygon_to_coco, split_multi_geometries
from ..utils.log import _get_logging_level
from ..vector.polygon import geojson_to_px_gdf, remove_multipolygons
import numpy as np
import rasterio
from tqdm.auto import tqdm
import json
import os
import pandas as pd
import geopandas as gpd
import logging


def geojson2coco(image_src, label_src, output_path=None, image_ext='.tif',
                 matching_re=None, category_attribute=None, score_attribute=None,
                 preset_categories=None, include_other=True, info_dict=None,
                 license_dict=None, recursive=False, override_crs=False,
                 explode_all_multipolygons=False, remove_all_multipolygons=False,
                 verbose=0):
    """Generate COCO-formatted labels from one or multiple geojsons and images.

    This function ingests optionally georegistered polygon labels in geojson
    format alongside image(s) and generates .json files per the
    `COCO dataset specification`_ . Some models, like
    many Mask R-CNN implementations, require labels to be in this format. The
    function assumes you're providing image file(s) and geojson file(s) to
    create the dataset. If the number of images and geojsons are both > 1 (e.g.
    with a SpaceNet dataset), you must provide a regex pattern to extract
    matching substrings to match images to label files.

    .. _COCO dataset specification: http://cocodataset.org/

    Arguments
    ---------
    image_src : :class:`str` or :class:`list` or :class:`dict`
        Source image(s) to use in the dataset. This can be::

            1. a string path to an image,
            2. the path to a directory containing a bunch of images,
            3. a list of image paths,
            4. a dictionary corresponding to COCO-formatted image records, or
            5. a string path to a COCO JSON containing image records.

        If a directory, the `recursive` flag will be used to determine whether
        or not to descend into sub-directories.
    label_src : :class:`str` or :class:`list`
        Source labels to use in the dataset. This can be a string path to a
        geojson, the path to a directory containing multiple geojsons, or a
        list of geojson file paths. If a directory, the `recursive` flag will
        determine whether or not to descend into sub-directories.
    output_path : str, optional
        The path to save the JSON-formatted COCO records to. If not provided,
        the records will only be returned as a dict, and not saved to file.
    image_ext : str, optional
        The string to use to identify images when searching directories. Only
        has an effect if `image_src` is a directory path. Defaults to
        ``".tif"``.
    matching_re : str, optional
        A regular expression pattern to match filenames between `image_src`
        and `label_src` if both are directories of multiple files. This has
        no effect if those arguments do not both correspond to directories or
        lists of files. Will raise a ``ValueError`` if multiple files are
        provided for both `image_src` and `label_src` but no `matching_re` is
        provided.
    category_attribute : str, optional
        The name of an attribute in the geojson that specifies which category
        a given instance corresponds to. If not provided, it's assumed that
        only one class of object is present in the dataset, which will be
        termed ``"other"`` in the output json.
    score_attribute : str, optional
        The name of an attribute in the geojson that specifies the prediction
        confidence of a model
    preset_categories : :class:`list` of :class:`dict`s, optional
        A pre-set list of categories to use for labels. These categories should
        be formatted per
        `the COCO category specification`_.
        example:
        [{'id': 1, 'name': 'Fighter Jet', 'supercategory': 'plane'},
        {'id': 2, 'name': 'Military Bomber', 'supercategory': 'plane'}, ... ]
    include_other : bool, optional
        If set to ``True``, and `preset_categories` is provided, objects that
        don't fall into the specified categories will not be removed from the
        dataset. They will instead be passed into a category named ``"other"``
        with its own associated category ``id``. If ``False``, objects whose
        categories don't match a category from `preset_categories` will be
        dropped.
    info_dict : dict, optional
        A dictonary with the following key-value pairs::

            - ``"year"``: :class:`int` year of creation
            - ``"version"``: :class:`str` version of the dataset
            - ``"description"``: :class:`str` string description of the dataset
            - ``"contributor"``: :class:`str` who contributed the dataset
            - ``"url"``: :class:`str` URL where the dataset can be found
            - ``"date_created"``: :class:`datetime.datetime` when the dataset
                was created

    license_dict : dict, optional
        A dictionary containing the licensing information for the dataset, with
        the following key-value pairs::

            - ``"name": :class:`str` the name of the license.
            -  ``"url": :class:`str` a link to the dataset's license.

        *Note*: This implementation assumes that all of the data uses one
        license. If multiple licenses are provided, the image records will not
        be assigned a license ID.
    recursive : bool, optional
        If `image_src` and/or `label_src` are directories, setting this flag
        to ``True`` will induce solaris to descend into subdirectories to find
        files. By default, solaris does not traverse the directory tree.
    explode_all_multipolygons : bool, optional
        Explode the multipolygons into individual geometries using sol.utils.geo.split_multi_geometries.
        Be sure to inspect which geometries are multigeometries, each individual geometries within these
        may represent artifacts rather than true labels.
    remove_all_multipolygons : bool, optional
        Filters MultiPolygons and GeometryCollections out of each tile geodataframe. Alternatively you
        can edit each polygon manually to be a polygon before converting to COCO format.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.

    Returns
    -------
    coco_dataset : dict
        A dictionary following the `COCO dataset specification`_ . Depending
        on arguments provided, it may or may not include license and info
        metadata.
    """

    # first, convert both image_src and label_src to lists of filenames
    logger = logging.getLogger(__name__)
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Preparing image filename: image ID dict.')
    # pdb.set_trace()
    if isinstance(image_src, str):
        if image_src.endswith('json'):
            logger.debug('COCO json provided. Extracting fname:id dict.')
            with open(image_src, 'r') as f:
                image_ref = json.load(f)
                image_ref = {image['file_name']: image['id']
                             for image in image_ref['images']}
        else:
            image_list = _get_fname_list(image_src, recursive=recursive,
                                         extension=image_ext)
            image_ref = dict(zip(image_list,
                                 list(range(1, len(image_list) + 1))
                                 ))
    elif isinstance(image_src, dict):
        logger.debug('image COCO dict provided. Extracting fname:id dict.')
        if 'images' in image_src.keys():
            image_ref = image_src['images']
        else:
            image_ref = image_src
        image_ref = {image['file_name']: image['id']
                     for image in image_ref}
    else:
        logger.debug('Non-COCO formatted image set provided. Generating '
                     'image fname:id dict with arbitrary ID integers.')
        image_list = _get_fname_list(image_src, recursive=recursive,
                                     extension=image_ext)
        image_ref = dict(zip(image_list, list(range(1, len(image_list) + 1))))

    logger.debug('Preparing label filename list.')
    label_list = _get_fname_list(label_src, recursive=recursive,
                                 extension='json')

    logger.debug('Checking if images and vector labels must be matched.')
    do_matches = len(image_ref) > 1 and len(label_list) > 1
    if do_matches:
        logger.info('Matching images to label files.')
        im_names = pd.DataFrame({'image_fname': list(image_ref.keys())})
        label_names = pd.DataFrame({'label_fname': label_list})
        logger.debug('Getting substrings for matching from image fnames.')
        if matching_re is not None:
            im_names['match_substr'] = im_names['image_fname'].str.extract(
                matching_re)
            logger.debug('Getting substrings for matching from label fnames.')
            label_names['match_substr'] = label_names[
                'label_fname'].str.extract(matching_re)
        else:
            logger.debug('matching_re is none, getting full filenames '
                         'without extensions for matching.')
            im_names['match_substr'] = im_names['image_fname'].apply(
                lambda x: os.path.splitext(os.path.split(x)[1])[0])
            im_names['match_substr'] = im_names['match_substr'].astype(
                str)
            label_names['match_substr'] = label_names['label_fname'].apply(
                lambda x: os.path.splitext(os.path.split(x)[1])[0])
            label_names['match_substr'] = label_names['match_substr'].astype(
                str)
        match_df = im_names.merge(label_names, on='match_substr', how='inner')

    logger.info('Loading labels.')
    label_df = pd.DataFrame({'label_fname': [],
                             'category_str': [],
                             'geometry': []})
    for gj in tqdm(label_list):
        logger.debug('Reading in {}'.format(gj))
        curr_gdf = gpd.read_file(gj)

        if remove_all_multipolygons is True and explode_all_multipolygons is True:
            raise ValueError("Only one of remove_all_multipolygons or explode_all_multipolygons can be set to True.")
        if remove_all_multipolygons is True and explode_all_multipolygons is False:
            curr_gdf = remove_multipolygons(curr_gdf)
        elif explode_all_multipolygons is True:
            curr_gdf = split_multi_geometries(curr_gdf)

        curr_gdf['label_fname'] = gj
        curr_gdf['image_fname'] = ''
        curr_gdf['image_id'] = np.nan
        if category_attribute is None:
            logger.debug('No category attribute provided. Creating a default '
                         '"other" category.')
            curr_gdf['category_str'] = 'other'  # add arbitrary value
            tmp_category_attribute = 'category_str'
        else:
            tmp_category_attribute = category_attribute
        if do_matches:  # multiple images: multiple labels
            logger.debug('do_matches is True, finding matching image')
            logger.debug('Converting to pixel coordinates.')
            if len(curr_gdf) > 0:  # if there are geoms, reproj to px coords
                curr_gdf = geojson_to_px_gdf(
                    curr_gdf,
                    override_crs=override_crs,
                    im_path=match_df.loc[match_df['label_fname'] == gj,
                                         'image_fname'].values[0])
                curr_gdf['image_id'] = image_ref[match_df.loc[
                    match_df['label_fname'] == gj, 'image_fname'].values[0]]
        # handle case with multiple images, one big geojson
        elif len(image_ref) > 1 and len(label_list) == 1:
            logger.debug('do_matches is False. Many images:1 label detected.')
            raise NotImplementedError('one label file: many images '
                                      'not implemented yet.')
        elif len(image_ref) == 1 and len(label_list) == 1:
            logger.debug('do_matches is False. 1 image:1 label detected.')
            logger.debug('Converting to pixel coordinates.')
            # match the two images
            curr_gdf = geojson_to_px_gdf(curr_gdf,
                                         override_crs=override_crs,
                                         im_path=list(image_ref.keys())[0])
            curr_gdf['image_id'] = list(image_ref.values())[0]
        curr_gdf = curr_gdf.rename(
            columns={tmp_category_attribute: 'category_str'})
        if score_attribute is not None:
            curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str',
                                 score_attribute, 'geometry']]
        else:
            curr_gdf = curr_gdf[['image_id', 'label_fname', 'category_str',
                                 'geometry']]
        label_df = pd.concat([label_df, curr_gdf], axis='index',
                             ignore_index=True, sort=False)

    logger.info('Finished loading labels.')
    logger.info('Generating COCO-formatted annotations.')
    coco_dataset = df_to_coco_annos(label_df,
                                    geom_col='geometry',
                                    image_id_col='image_id',
                                    category_col='category_str',
                                    score_col=score_attribute,
                                    preset_categories=preset_categories,
                                    include_other=include_other,
                                    verbose=verbose)

    logger.info('Generating COCO-formatted image and license records.')
    if license_dict is not None:
        logger.debug('Getting license ID.')
        if len(license_dict) == 1:
            logger.debug('Only one license present; assuming it applies to '
                         'all images.')
            license_id = 1
        else:
            logger.debug('Zero or multiple licenses present. Not trying to '
                         'match to images.')
            license_id = None
        logger.info('Adding licenses to dataset.')
        coco_licenses = []
        license_idx = 1
        for license_name, license_url in license_dict.items():
            coco_licenses.append({'name': license_name,
                                  'url': license_url,
                                  'id': license_idx})
            license_idx += 1
        coco_dataset['licenses'] = coco_licenses
    else:
        logger.debug('No license information provided, skipping for image '
                     'COCO records.')
        license_id = None
    coco_image_records = make_coco_image_dict(image_ref, license_id)
    coco_dataset['images'] = coco_image_records

    logger.info('Adding any additional information provided as arguments.')
    if info_dict is not None:
        coco_dataset['info'] = info_dict

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            json.dump(coco_dataset, outfile)

    return coco_dataset


def df_to_coco_annos(df, output_path=None, geom_col='geometry',
                     image_id_col=None, category_col=None, score_col=None,
                     preset_categories=None, supercategory_col=None,
                     include_other=True, starting_id=1, verbose=0):
    """Extract COCO-formatted annotations from a pandas ``DataFrame``.

    This function assumes that *annotations are already in pixel coordinates.*
    If this is not the case, you can transform them using
    :func:`solaris.vector.polygon.geojson_to_px_gdf`.

    Note that this function generates annotations formatted per the COCO object
    detection specification. For additional information, see
    `the COCO dataset specification`_.

    .. _the COCO dataset specification: http://cocodataset.org/#format-data

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` containing geometries to store as annos.
    image_id_col : str, optional
        The column containing image IDs. If not provided, it's assumed that
        all are in the same image, which will be assigned the ID of ``1``.
    geom_col : str, optional
        The name of the column in `df` that contains geometries. The geometries
        should either be shapely :class:`shapely.geometry.Polygon` s or WKT
        strings. Defaults to ``"geometry"``.
    category_col : str, optional
        The name of the column that specifies categories for each object. If
        not provided, all objects will be placed in a single category named
        ``"other"``.
    score_col : str, optional
        The name of the column that specifies the ouptut confidence of a model.
        If not provided, will not be output.
    preset_categories : :class:`list` of :class:`dict`s, optional
        A pre-set list of categories to use for labels. These categories should
        be formatted per
        `the COCO category specification`_.
    starting_id : int, optional
        The number to start numbering annotation IDs at. Defaults to ``1``.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.

    .. _the COCO category specification: http://cocodataset.org/#format-data

    Returns
    -------
    output_dict : dict
        A dictionary containing COCO-formatted annotation and category entries
        per the `COCO dataset specification`_
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Checking that df is loaded.')
    df = _check_df_load(df)
    temp_df = df.copy()  # for manipulation
    if preset_categories is not None and category_col is None:
        logger.debug('preset_categories has a value, category_col is None.')
        raise ValueError('category_col must be specified if using'
                         ' preset_categories.')
    elif preset_categories is not None and category_col is not None:
        logger.debug('Both preset_categories and category_col have values.')
        logger.debug('Getting list of category names.')
        category_dict = _coco_category_name_id_dict_from_list(
            preset_categories)
        category_names = list(category_dict.keys())
        if not include_other:
            logger.info('Filtering out objects not contained in '
                        ' preset_categories')
            temp_df = temp_df.loc[temp_df[category_col].isin(category_names),
                                  :]
        else:
            logger.info('Setting category to "other" for objects outside of '
                        'preset category list.')
            temp_df.loc[~temp_df[category_col].isin(category_names),
                        category_col] = 'other'
            if 'other' not in category_dict.keys():
                logger.debug('Adding "other" to category_dict.')
                other_id = np.array(list(category_dict.values())).max() + 1
                category_dict['other'] = other_id
                preset_categories.append({'id': other_id,
                                          'name': 'other',
                                          'supercategory': 'other'})
    elif preset_categories is None and category_col is not None:
        logger.debug('No preset_categories, have category_col.')
        logger.info(f'Collecting unique category names from {category_col}.')
        category_names = list(temp_df[category_col].unique())
        logger.info('Generating category ID numbers arbitrarily.')
        category_dict = {k: v for k, v in zip(category_names,
                                              range(1, len(category_names)+1))}
    else:
        logger.debug('No category column or preset categories.')
        logger.info('Setting category to "other" for all objects.')
        category_col = 'category_col'
        temp_df[category_col] = 'other'
        category_names = ['other']
        category_dict = {'other': 1}

    if image_id_col is None:
        temp_df['image_id'] = 1
    else:
        temp_df.rename(columns={image_id_col: 'image_id'})
    logger.debug('Checking geometries.')
    temp_df[geom_col] = temp_df[geom_col].apply(_check_geom)
    logger.info('Getting area of geometries.')
    temp_df['area'] = temp_df[geom_col].apply(lambda x: x.area)
    logger.info('Getting geometry bounding boxes.')
    temp_df['bbox'] = temp_df[geom_col].apply(
        lambda x: bbox_corners_to_coco(x.bounds))
    temp_df['category_id'] = temp_df[category_col].map(category_dict)
    temp_df['annotation_id'] = list(range(starting_id,
                                          starting_id + len(temp_df)))
    if score_col is not None:
        temp_df['score'] = df[score_col]

    def _row_to_coco(row, geom_col, category_id_col, image_id_col, score_col):
        "get a single annotation record from a row of temp_df."
        if score_col is None:

            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}
        else:
            return {'id': row['annotation_id'],
                    'image_id': int(row[image_id_col]),
                    'category_id': int(row[category_id_col]),
                    'segmentation': [polygon_to_coco(row[geom_col])],
                    'score': float(row[score_col]),
                    'area': row['area'],
                    'bbox': row['bbox'],
                    'iscrowd': 0}

    coco_annotations = temp_df.apply(_row_to_coco, axis=1, geom_col=geom_col,
                                     category_id_col='category_id',
                                     image_id_col=image_id_col,
                                     score_col=score_col).tolist()
    coco_categories = coco_categories_dict_from_df(
        temp_df, category_id_col='category_id',
        category_name_col=category_col,
        supercategory_col=supercategory_col)

    output_dict = {'annotations': coco_annotations,
                   'categories': coco_categories}

    if output_path is not None:
        with open(output_path, 'w') as outfile:
            json.dump(output_dict, outfile)

    return output_dict


def coco_categories_dict_from_df(df, category_id_col, category_name_col,
                                 supercategory_col=None):
    """Extract category IDs, category names, and supercat names from df.

    Arguments
    ---------
    df : :class:`pandas.DataFrame`
        A :class:`pandas.DataFrame` of records to filter for category info.
    category_id_col : str
        The name for the column in `df` that contains category IDs.
    category_name_col : str
        The name for the column in `df` that contains category names.
    supercategory_col : str, optional
        The name for the column in `df` that contains supercategory names,
        if one exists. If not provided, supercategory will be left out of the
        output.

    Returns
    -------
    :class:`list` of :class:`dict` s
        A :class:`list` of :class:`dict` s that contain category records per
        the `COCO dataset specification`_ .
    """
    cols_to_keep = [category_id_col, category_name_col]
    rename_dict = {category_id_col: 'id',
                   category_name_col: 'name'}
    if supercategory_col is not None:
        cols_to_keep.append(supercategory_col)
        rename_dict[supercategory_col] = 'supercategory'
    coco_cat_df = df[cols_to_keep]
    coco_cat_df = coco_cat_df.rename(columns=rename_dict)
    coco_cat_df = coco_cat_df.drop_duplicates()

    return coco_cat_df.to_dict(orient='records')


def make_coco_image_dict(image_ref, license_id=None):
    """Take a dict of ``image_fname: image_id`` pairs and make a coco dict.

    Note that this creates a relatively limited version of the standard
    `COCO image record format`_ record, which only contains the following
    keys::

        * id ``(int)``
        * width ``(int)``
        * height ``(int)``
        * file_name ``(str)``
        * license ``(int)``, optional

    .. _COCO image record format: http://cocodataset.org/#format-data

    Arguments
    ---------
    image_ref : dict
        A dictionary of ``image_fname: image_id`` key-value pairs.
    license_id : int, optional
        The license ID number for the relevant license. If not provided, no
        license information will be included in the output.

    Returns
    -------
    coco_images : list
        A list of COCO-formatted image records ready for export to json.
    """

    image_records = []
    for image_fname, image_id in image_ref.items():
        with rasterio.open(image_fname) as f:
            width = f.width
            height = f.height
        im_record = {'id': image_id,
                     'file_name': os.path.split(image_fname)[1],
                     'width': width,
                     'height': height}
        if license_id is not None:
            im_record['license'] = license_id
        image_records.append(im_record)

    return image_records


def _coco_category_name_id_dict_from_list(category_list):
    """Extract ``{category_name: category_id}`` from a list."""
    # check if this is a full annotation json or just the categories
    category_dict = {category['name']: category['id']
                     for category in category_list}
    return category_dict


def _get_fname_list(p, recursive=False, extension='.tif'):
    """Get a list of filenames from p, which can be a dir, fname, or list."""
    if isinstance(p, list):
        return p
    elif isinstance(p, str):
        if os.path.isdir(p):
            return get_files_recursively(p, traverse_subdirs=recursive,
                                         extension=extension)
        elif os.path.isfile(p):
            return [p]
        else:
            raise ValueError("If a string is provided, it must be a valid"
                             " path.")
    else:
        raise ValueError("{} is not a string or list.".format(p))
