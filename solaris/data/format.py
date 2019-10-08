from ..utils.core import get_fname_list, _check_df_load, _check_geom
from ..utils.geo import bbox_corners_to_coco
from ..utils.log import _get_logging_level
from ..vector.polygon import geojson_to_px_gdf
import numpy as np
from tqdm import tqdm
import json
import os
import re
import pandas as pd
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)


def geojson2coco(image_src, label_src, image_ext='.tif', matching_re=None,
                 category_attribute=None, category_dict=None,
                 other_category=True, info_dict=None, license_dict=None,
                 recursive=False, verbose=0):
    """Generate COCO-formatted labels from one or multiple geojsons and images.

    This function ingests optionally georegistered polygon labels in geojson
    format alongside image(s) and generates .json files per the
    `COCO dataset specification <http://cocodataset.org/>`. Some models, like
    many Mask R-CNN implementations, require labels to be in this format. The
    function assumes you're providing image file(s) and geojson file(s) to
    create the dataset. If the number of images and geojsons are both > 1 (e.g.
    with a SpaceNet dataset), you must provide a regex pattern to extract
    matching substrings to match images to label files.

    Arguments
    ---------
    image_src : :class:`str` or :class:`list`
        Source image(s) to use in the dataset. This can be a string path to an
        image, the path to a directory containing a bunch of images, or a list
        of image paths. If a directory, the `recursive` flag will be used to
        determine whether or not to descend into sub-directories.
    label_src : :class:`str` or :class:`list`
        Source labels to use in the dataset. This can be a string path to a
        geojson, the path to a directory containing multiple geojsons, or a
        list of geojson file paths. If a directory, the `recursive` flag will
        determine whether or not to descend into sub-directories.
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
    category_dict : dict, optional
        A dictionary if ``id: category`` pairs that specifies which category
        ID number corresponds to which object type. If provided, only the
        categories defined in the dict will be included, plus an additional
        ``"other"`` category depending on the value of `other_category`. If not
        provided, all values present for `category_attribute` are used.
    other_category :
    info_dict : dict, optional
        A dictonary with the following key-value pairs::

            - ``"year"``: :class:`int` year of creation
            - ``"version"``: :class:`str` version of the dataset
            - ``"description"``: :class:`str` string description of the dataset
            - ``"contributor"``: :class:`str` who contributed the dataset
            - ``"url"``: :class:`str` URL where the dataset can be found
            - ``"date_created"``: :class:`datetime.datetime` when the dataset
                was created

        If not provided, those values are all filled with ``None``.

    license_dict : dict, optional
        A dictionary containing the licensing information for the dataset, with
        the following key-value pairs::

            - ``"name": :class:`str` the name of the license.
            -  ``"url": :class:`str` a link to the dataset's license.

        *Note*: This implementation assumes that all of the data uses one
        license.
    recursive : bool, optional
        If `image_src` and/or `label_src` are directories, setting this flag
        to ``True`` will induce solaris to descend into subdirectories to find
        files. By default, solaris does not traverse the directory tree.
    verbose : int, optional
        Verbose text output. By default, none is provided; if ``True`` or
        ``1``, information-level outputs are provided; if ``2``, extremely
        verbose text is output.
    """

    # first, convert both image_src and label_src to lists of filenames
    logger.setLevel(_get_logging_level(int(verbose)))
    logger.debug('Preparing image and label filename lists.')
    image_list = get_fname_list(image_src, recursive=recursive,
                                extension=image_ext)
    label_list = get_fname_list(label_src, recursive=recursive,
                                extension='json')
    logger.debug('Checking if images and vector labels must be matched.')
    do_matches = len(image_list) > 1 and len(label_list) > 1
    if do_matches:
        logger.info('Matching images to label files.')
        im_names = pd.DataFrame({'image_fname': image_list})
        label_names = pd.DataFrame({'label_fname': label_list})
        logger.debug('Getting substrings for matching from image fnames.')
        im_names['match_substr'] = im_names['image_fname'].extract(
            matching_re)
        logger.debug('Getting substrings for matching from label fnames.')
        label_names['match_substr'] = label_names['label_fname'].extract(
            matching_re)
        match_df = im_names.join(label_names, on='match_substr', how='inner')

    logger.info('Loading labels.')
    label_df = pd.DataFrame({'label_fname': [],
                             'category_str': []},
                            geometry=[])
    for gj in tqdm(label_list):
        logger.debug('Reading in {}'.format(gj))
        curr_gdf = gpd.read_file(gj)
        curr_gdf['label_fname'] = gj
        if do_matches:
            logger.debug('do_matches is True, finding matching image')
            logger.debug('Converting to pixel coordinates.')
            curr_gdf = geojson_to_px_gdf(
                curr_gdf,
                im_path=match_df.loc[match_df['label_fname'] == gj,
                                     'image_fname'])
        # handle case with multiple images, one big geojson
        elif len(image_list) > 1 and len(label_list) == 1:
            logger.debug('do_matches is False. Many images:1 label detected.')
            pass
        elif len(image_list) == 1 and len(label_list) == 1:
            logger.debug('do_matches is False. 1 image:1 label detected.')
            logger.info('Converting to pixel coordinates.')
            # match the two images
            curr_gdf = geojson_to_px_gdf(curr_gdf, im_path=image_list[0])
        curr_gdf.rename(columns={'category_attribute': 'category_str'})
        curr_gdf = curr_gdf[['label_fname', 'category_str', 'geometry']]
        label_df = pd.concat([label_df, curr_gdf], axis='index',
                             ignore_index=True)


def df_to_coco_annos(df, output_path=None, geom_col='geometry',
                     image_id_col=None, category_col=None,
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

    """
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
        category_dict = _coco_category_name_id_dict_from_json(
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
                                              range(1, len(category_names)))}
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

    def _row_to_coco(row, geom_col, category_col, image_id_col):
        "get a single annotation record from a row of temp_df."
        return {'id': row['annotation_id'],
                'image_id': row[image_id_col],
                'category_id': row[category_col],
                'segmentation': row[geom_col].exterior.coords.xy,
                'area': row['area'],
                'bbox': row['bbox'],
                'iscrowd': 0}

    coco_annotations = temp_df.apply(_row_to_coco, axis=1)
    coco_categories = coco_categories_dict_from_df(
        df, category_id_col='category_id',
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
    """Extract category IDs, category names, and supercat names from df."""
    cols_to_keep = [category_id_col, category_name_col]
    if supercategory_col is not None:
        cols_to_keep.append(supercategory_col)
    coco_cat_df = df[cols_to_keep]
    coco_cat_df = coco_cat_df.drop_duplicates()

    return coco_cat_df.to_dict(orient='records')


def _coco_category_name_id_dict_from_json(category_json):
    """Extract ``{category_name: category_id}`` from the COCO JSON."""
    if isinstance(category_json, str):  # if it's a filepath
        with open(category_json, "r") as f:
            category_json = json.load(f)
    # check if this is a full annotation json or just the categories
    if 'categories' in category_json.keys():
        category_json = category_json['categories']
    category_dict = {category['name']: category['id']
                     for category in category_json}
    return category_dict
