from __future__ import print_function, with_statement, division

import pandas as pd
from .. import baseeval as bF
import re

# Note, for mac osx compatability import something from shapely.geometry before
# importing fiona or geopandas
# https://github.com/Toblerity/Shapely/issues/553  * Import shapely before
# rasterio or fiona


def eval_spacenet_buildings2(prop_csv, truth_csv, miniou=0.5, minArea=20):
    """Evaluate a SpaceNet2 building footprint competition proposal csv.

    Uses :class:`EvalBase` to evaluate SpaceNet2 challenge proposals.

    Arguments
    ---------
    prop_csv : str
        Path to the proposal polygon CSV file.
    truth_csv : str
        Path to the ground truth polygon CSV file.
    miniou : float, optional
        Minimum IoU score between a region proposal and ground truth to define
        as a successful identification. Defaults to 0.5.
    minArea : float or int, optional
        Minimum area of ground truth regions to include in scoring calculation.
        Defaults to ``20``.

    Returns
    -------

    results_DF, results_DF_Full

        results_DF : :py:class:`pd.DataFrame`
            Summary :py:class:`pd.DataFrame` of score outputs grouped by nadir
            angle bin, along with the overall score.

        results_DF_Full : :py:class:`pd.DataFrame`
            :py:class:`pd.DataFrame` of scores by individual image chip across
            the ground truth and proposal datasets.

    """

    evalObject = bF.EvalBase(ground_truth_vector_file=truth_csv)
    evalObject.load_proposal(prop_csv,
                             conf_field_list=['Confidence'],
                             proposalCSV=True
                             )
    results = evalObject.eval_iou_spacenet_csv(miniou=miniou,
                                               iou_field_prefix="iou_score",
                                               imageIDField="ImageId",
                                               minArea=minArea
                                               )
    results_DF_Full = pd.DataFrame(results)

    results_DF_Full['AOI'] = [get_aoi(imageID) for imageID
                              in results_DF_Full['imageID'].values]

    results_DF = results_DF_Full.groupby(['AOI']).sum()

    # Recalculate Values after Summation of AOIs
    for indexVal in results_DF.index:
        rowValue = results_DF[results_DF.index == indexVal]
        # Precision = TruePos / float(TruePos + FalsePos)
        if float(rowValue['TruePos'] + rowValue['FalsePos']) > 0:
            Precision = float(
                rowValue['TruePos'] / float(rowValue['TruePos'] +
                                            rowValue['FalsePos'])
                )
        else:
            Precision = 0
        # Recall    = TruePos / float(TruePos + FalseNeg)
        if float(rowValue['TruePos'] + rowValue['FalseNeg']) > 0:
            Recall = float(rowValue['TruePos'] / float(rowValue['TruePos'] +
                                                       rowValue['FalseNeg']))
        else:
            Recall = 0
        if Recall * Precision > 0:
            F1Score = 2 * Precision * Recall / (Precision + Recall)
        else:
            F1Score = 0
        results_DF.loc[results_DF.index == indexVal, 'Precision'] = Precision
        results_DF.loc[results_DF.index == indexVal, 'Recall'] = Recall
        results_DF.loc[results_DF.index == indexVal, 'F1Score'] = F1Score

    return results_DF, results_DF_Full


def get_aoi(imageID):
    """Get the AOI from an image name"""
    return '_'.join(imageID.split('_')[:-1])
