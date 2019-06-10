import geopandas as gpd


def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.

    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.

    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.

    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for _, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
        else:
            iou_score = 0
        row['iou_score'] = iou_score
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF


def process_iou(pred_poly, test_data_GDF, remove_matching_element=True):
    """Get the maximum intersection over union score for a predicted polygon.

    Arguments
    ---------
    pred_poly : :py:class:`shapely.geometry.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    remove_matching_element : bool, optional
        Should the maximum IoU row be dropped from ``test_data_GDF``? Defaults
        to ``True``.

    Returns
    -------
    *This function doesn't currently return anything.*

    """

    iou_GDF = calculate_iou(pred_poly, test_data_GDF)

    max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]

    if remove_matching_element:
        test_data_GDF.drop(max_iou_row.name, axis=0, inplace=True)

    # Prediction poly had no overlap with anything
    # if not iou_list:
    #     return max_iou_row, 0, test_data_DF
    # else:
    #     max_iou_idx, max_iou = max(iou_list, key=lambda x: x[1])
    #     # Remove ground truth polygon from tree
    #     test_tree.delete(max_iou_idx, Polygon(test_data[max_iou_idx]['geometry']['coordinates'][0]).bounds)
    #     return max_iou_row['iou_score'], iou_GDF, test_data_DF
