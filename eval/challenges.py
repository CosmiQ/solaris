import pandas as pd
import geopandas as gpd
from .base import Evaluator
from .scot import scot_multi_aoi
import re


def spacenet_buildings_2(prop_csv, truth_csv, miniou=0.5, min_area=20, challenge='spacenet_2'):
    """Evaluate a SpaceNet building footprint competition proposal csv.

    Uses :class:`Evaluator` to evaluate SpaceNet challenge proposals.

    Arguments
    ---------
    prop_csv : str
        Path to the proposal polygon CSV file.
    truth_csv : str
        Path to the ground truth polygon CSV file.
    miniou : float, optional
        Minimum IoU score between a region proposal and ground truth to define
        as a successful identification. Defaults to 0.5.
    min_area : float or int, optional
        Minimum area of ground truth regions to include in scoring calculation.
        Defaults to ``20``.
    challenge: str, optional
        The challenge id for evaluation.
        One of
        ``['spacenet_2', 'spacenet_3', 'spacenet_off_nadir', 'spacenet_6']``.
        The name of the challenge that `chip_name` came from. Defaults to
        ``'spacenet_2'``.

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

    evaluator = Evaluator(ground_truth_vector_file=truth_csv)
    evaluator.load_proposal(prop_csv,
                            conf_field_list=['Confidence'],
                            proposalCSV=True
                            )
    results = evaluator.eval_iou_spacenet_csv(miniou=miniou,
                                              iou_field_prefix="iou_score",
                                              imageIDField="ImageId",
                                              min_area=min_area
                                              )
    results_DF_Full = pd.DataFrame(results)

    results_DF_Full['AOI'] = [get_chip_id(imageID, challenge=challenge)
                              for imageID in results_DF_Full['imageID'].values]

    results_DF = results_DF_Full.groupby(['AOI']).sum()

    # Recalculate Values after Summation of AOIs
    for indexVal in results_DF.index:
        rowValue = results_DF[results_DF.index == indexVal]
        # Precision = TruePos / float(TruePos + FalsePos)
        if float(rowValue['TruePos'] + rowValue['FalsePos']) > 0:
            Precision = float(
                rowValue['TruePos'] / float(
                    rowValue['TruePos'] + rowValue['FalsePos']))
        else:
            Precision = 0
        # Recall    = TruePos / float(TruePos + FalseNeg)
        if float(rowValue['TruePos'] + rowValue['FalseNeg']) > 0:
            Recall = float(rowValue['TruePos'] / float(
                rowValue['TruePos'] + rowValue['FalseNeg']))
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


def off_nadir_buildings(prop_csv, truth_csv, image_columns={}, miniou=0.5,
                        min_area=20, verbose=False):
    """Evaluate an off-nadir competition proposal csv.

    Uses :class:`Evaluator` to evaluate off-nadir challenge proposals. See
    ``image_columns`` in the source code for how collects are broken into
    Nadir, Off-Nadir, and Very-Off-Nadir bins.

    Arguments
    ---------
    prop_csv : str
        Path to the proposal polygon CSV file.
    truth_csv : str
        Path to the ground truth polygon CSV file.
    image_columns : dict, optional
        dict of ``(collect: nadir bin)`` pairs used to separate collects into
        sets. Nadir bin values must be one of
        ``["Nadir", "Off-Nadir", "Very-Off-Nadir"]`` . See source code for
        collect name options.
    miniou : float, optional
        Minimum IoU score between a region proposal and ground truth to define
        as a successful identification. Defaults to 0.5.
    min_area : float or int, optional
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

    evaluator = Evaluator(ground_truth_vector_file=truth_csv)
    evaluator.load_proposal(prop_csv,
                            conf_field_list=['Confidence'],
                            proposalCSV=True
                            )
    results = evaluator.eval_iou_spacenet_csv(miniou=miniou,
                                              iou_field_prefix="iou_score",
                                              imageIDField="ImageId",
                                              min_area=min_area
                                              )
    results_DF_Full = pd.DataFrame(results)

    if not image_columns:
        image_columns = {
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
        image_columns[get_chip_id(imageID, challenge='spacenet_off_nadir')]
        for imageID in results_DF_Full['imageID'].values]

    results_DF = results_DF_Full.groupby(['nadir-category']).sum()

    # Recalculate Values after Summation of AOIs
    for indexVal in results_DF.index:
        if verbose:
            print(indexVal)
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


def multi_temporal_buildings(prop_csv, truth_csv, miniou=0.25, min_area=4.,
                             beta=2., stats=False, verbose=False):
    """
    Evaluate submissions to SpaceNet 7: Multi-Temporal Urban Development
    Input CSV files should have "filename", "id", and "geometry" columns.
    """

    # Load dataframes
    grnd_df = gpd.read_file(truth_csv, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
    prop_df = gpd.read_file(prop_csv, GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO")
    if verbose:
        print("init len grnd_df:", len(grnd_df))
        print("init len prop_df:", len(prop_df))

    # Filter out small buildings from ground truth
    if min_area is not None:
        grnd_df['area'] = grnd_df.area
        grnd_df = grnd_df[grnd_df['area'] >= min_area]
        grnd_df = grnd_df.drop(columns=['area'])
        if verbose:
            print("filtered len grnd_df:", len(grnd_df))

    # Extract place (aoi) and time (timestep) from the "filename" column
    grnd_df['aoi'] = grnd_df['filename'].str.slice(30, 58)
    prop_df['aoi'] = prop_df['filename'].str.slice(30, 58)
    grnd_df['timestep'] = grnd_df['filename'].str.slice(15, 22)
    prop_df['timestep'] = prop_df['filename'].str.slice(15, 22)
    aois = sorted(list(grnd_df.aoi.drop_duplicates()))
    if verbose:
        print("Number of AOIS:", len(aois))

    # Compute the score for this proposal
    score, all_stats = scot_multi_aoi(grnd_df, prop_df,
                                      threshold=miniou, base_reward=100.,
                                      beta=beta,
                                      stats=True, verbose=verbose)
    if verbose:
        print('The submission "%s" receives a score of %f'
              % (prop_csv, score))
    if stats:
        return (score, all_stats)
    else:
        return score


def get_chip_id(chip_name, challenge="spacenet_2"):
    """Get the unique identifier for a chip location from SpaceNet images.

    Arguments
    ---------
    chip_name: str
        The name of the chip to extract the identifier from.
    challenge: str, optional
        One of
        ``['spacenet_2', 'spacenet_3', 'spacenet_off_nadir', 'spacenet_6']``.
        The name of the challenge that `chip_name` came from. Defaults to
        ``'spacenet_2'``.

    Returns
    -------
    chip_id : str
        The unique identifier for the chip location.
    """
    # AS NEW CHALLENGES ARE ADDED, ADD THE CHIP MATCHING FUNCTIONALITY WITHIN
    # THIS FUNCTION.
    if challenge in ['spacenet_2', 'spacenet_3']:
        chip_id = '_'.join(chip_name.split('_')[:-1])
    elif challenge == 'spacenet_off_nadir':
        chip_id = re.findall('Atlanta_nadir[0-9]{1,2}_catid_[0-9A-Z]{16}',
                             chip_name)[0]
    elif challenge == 'spacenet_6':
        chip_id = '_'.join(chip_name.split('_')[-4:]).split(".")[0]

    return chip_id
