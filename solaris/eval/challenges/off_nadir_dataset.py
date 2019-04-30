from __future__ import print_function, with_statement, division

import pandas as pd
from .. import baseeval as bF
import re

# Note, for mac osx compatability import something from shapely.geometry before
# importing fiona or geopandas
# https://github.com/Toblerity/Shapely/issues/553  * Import shapely before
# rasterio or fiona


def eval_off_nadir(prop_csv, truth_csv, imageColumns={}, miniou=0.5,
                   minArea=20):
    """Evaluate an off-nadir competition proposal csv.

    Uses :class:`EvalBase` to evaluate off-nadir challenge proposals. See
    ``imageColumns`` in the source code for how collects are broken into
    Nadir, Off-Nadir, and Very-Off-Nadir bins.

    Arguments
    ---------
    prop_csv : str
        Path to the proposal polygon CSV file.
    truth_csv : str
        Path to the ground truth polygon CSV file.
    imageColumns : dict, optional
        dict of ``(collect: nadir bin)`` pairs used to separate collects into
        sets. Nadir bin values must be one of
        ``["Nadir", "Off-Nadir", "Very-Off-Nadir"]`` . See source code for
        collect name options.
    miniou : float, optional
        Minimum IoU score between a region proposal and ground truth to define
        as a successful identification. Defaults to 0.5.
    minArea : float or int, optional
        Minimum area of ground truth regions to include in scoring calculation.
        Defaults to ``20``.

    Returnss
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

    if not imageColumns:
        imageColumns = {
            'Atlanta_nadir7_catid_1030010003D22F00': "Nadir",
            'Atlanta_nadir8_catid_10300100023BC100': "Nadir",
            'Atlanta_nadir10_catid_1030010003993E00': "Nadir",
            'Atlanta_nadir10_catid_1030010003CAF100': "Nadir",
            'Atlanta_nadir13_catid_1030010002B7D800': "Nadir",
            'Atlanta_nadir14_catid_10300100039AB000': "Nadir",
            'Atlanta_nadir16_catid_1030010002649200': "Nadir",
            'Atlanta_nadir19_catid_1030010003C92000': "Nadir",
            'Atlanta_nadir21_catid_1030010003127500': "Nadir",
            'Atlanta_nadir23_catid_103001000352C200': "Nadir",
            'Atlanta_nadir25_catid_103001000307D800': "Nadir",
            'Atlanta_nadir27_catid_1030010003472200': "Off-Nadir",
            'Atlanta_nadir29_catid_1030010003315300': "Off-Nadir",
            'Atlanta_nadir30_catid_10300100036D5200': "Off-Nadir",
            'Atlanta_nadir32_catid_103001000392F600': "Off-Nadir",
            'Atlanta_nadir34_catid_1030010003697400': "Off-Nadir",
            'Atlanta_nadir36_catid_1030010003895500': "Off-Nadir",
            'Atlanta_nadir39_catid_1030010003832800': "Off-Nadir",
            'Atlanta_nadir42_catid_10300100035D1B00': "Very-Off-Nadir",
            'Atlanta_nadir44_catid_1030010003CCD700': "Very-Off-Nadir",
            'Atlanta_nadir46_catid_1030010003713C00': "Very-Off-Nadir",
            'Atlanta_nadir47_catid_10300100033C5200': "Very-Off-Nadir",
            'Atlanta_nadir49_catid_1030010003492700': "Very-Off-Nadir",
            'Atlanta_nadir50_catid_10300100039E6200': "Very-Off-Nadir",
            'Atlanta_nadir52_catid_1030010003BDDC00': "Very-Off-Nadir",
            'Atlanta_nadir53_catid_1030010003193D00': "Very-Off-Nadir",
            'Atlanta_nadir53_catid_1030010003CD4300': "Very-Off-Nadir",
        }

    results_DF_Full['nadir-category'] = [
        imageColumns[get_collect_id(imageID)]
        for imageID in results_DF_Full['imageID'].values]

    results_DF = results_DF_Full.groupby(['nadir-category']).sum()

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


def get_collect_id(imageID):
    """Get the collect ID for an image name using a regex."""
    collect_id = re.findall('Atlanta_nadir[0-9]{1,2}_catid_[0-9A-Z]{16}',
                            imageID)[0]

    return collect_id
