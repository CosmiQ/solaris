import shapely.wkt
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import os
from . import iou
from fiona.errors import DriverError
from fiona._err import CPLE_OpenFailedError


class Evaluator():
    """Object to test IoU for predictions and ground truth polygons.

    Attributes
    ----------
    ground_truth_fname : str
        The filename for the ground truth CSV or JSON.
    ground_truth_GDF : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` containing the ground truth vector
        labels.
    ground_truth_GDF_Edit : :class:`geopandas.GeoDataFrame`
        A copy of ``ground_truth_GDF`` which will be manipulated during
        processing.
    proposal_GDF : :class:`geopandas.GeoDataFrame`
        The proposal :class:`geopandas.GeoDataFrame`, added using
        ``load_proposal()``.

    Arguments
    ---------
    ground_truth_vector_file : str
        Path to .geojson file for ground truth.

    """

    def __init__(self, ground_truth_vector_file):
        # Load Ground Truth : Ground Truth should be in geojson or shape file
        try:
            if ground_truth_vector_file.lower().endswith('json'):
                self.load_truth(ground_truth_vector_file)
            elif ground_truth_vector_file.lower().endswith('csv'):
                self.load_truth(ground_truth_vector_file, truthCSV=True)
            self.ground_truth_fname = ground_truth_vector_file
        except AttributeError:  # handles passing gdf instead of path to file
            self.ground_truth_GDF = ground_truth_vector_file
            self.ground_truth_fname = 'GeoDataFrame variable'
        self.ground_truth_sindex = self.ground_truth_GDF.sindex  # get sindex
        # create deep copy of ground truth file for calculations
        self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(deep=True)
        self.proposal_GDF = gpd.GeoDataFrame([])  # initialize proposal GDF

    def __repr__(self):
        return 'Evaluator {}'.format(os.path.split(
            self.ground_truth_fname)[-1])

    def get_iou_by_building(self):
        """Returns a copy of the ground truth table, which includes a
        per-building IoU score column after eval_iou_spacenet_csv() has run.
        """

        output_ground_truth_GDF = self.ground_truth_GDF.copy(deep=True)
        return output_ground_truth_GDF

    def eval_iou_spacenet_csv(self, miniou=0.5, iou_field_prefix="iou_score",
                              imageIDField="ImageId", debug=False, min_area=0):
        """Evaluate IoU between the ground truth and proposals in CSVs.

        Arguments
        ---------
        miniou : float , optional
            Minimum intersection over union score to qualify as a successful
            object detection event. Defaults to ``0.5``.
        iou_field_prefix : str , optional
            The name of the IoU score column in ``self.proposal_GDF``. Defaults
            to ``"iou_score"`` .
        imageIDField : str , optional
            The name of the column corresponding to the image IDs in the
            ground truth data. Defaults to ``"ImageId"``.
        debug : bool , optional
            Argument for verbose execution during debugging. Defaults to
            ``False`` (silent execution).
        min_area : float  or int , optional
            Minimum area of a ground truth polygon to be considered during
            evaluation. Often set to ``20`` in SpaceNet competitions. Defaults
            to ``0``  (consider all ground truth polygons).

        Returns
        -------
        scoring_dict_list : list
            list  of score output dicts for each image in the ground
            truth and evaluated image datasets. The dicts contain
            the following keys: ::

                ('imageID', 'iou_field', 'TruePos', 'FalsePos', 'FalseNeg',
                'Precision', 'Recall', 'F1Score')

        """
        # Get List of all ImageID in both ground truth and proposals
        imageIDList = []
        imageIDList.extend(list(self.ground_truth_GDF[imageIDField].unique()))
        if not self.proposal_GDF.empty:
            imageIDList.extend(list(self.proposal_GDF[imageIDField].unique()))
        imageIDList = list(set(imageIDList))
        iou_field = iou_field_prefix
        scoring_dict_list = []
        self.ground_truth_GDF[iou_field] = 0.
        iou_index = self.ground_truth_GDF.columns.get_loc(iou_field)
        id_cols = 2
        ground_truth_ids = self.ground_truth_GDF.iloc[:, :id_cols]

        for imageID in tqdm(imageIDList):
            self.ground_truth_GDF_Edit = self.ground_truth_GDF[
                self.ground_truth_GDF[imageIDField] == imageID
                ].copy(deep=True)
            self.ground_truth_GDF_Edit = self.ground_truth_GDF_Edit[
                self.ground_truth_GDF_Edit.area >= min_area
                ]
            proposal_GDF_copy = self.proposal_GDF[self.proposal_GDF[
                imageIDField] == imageID].copy(deep=True)
            proposal_GDF_copy = proposal_GDF_copy[proposal_GDF_copy.area
                                                  > min_area]
            if debug:
                print(iou_field)
            for _, pred_row in proposal_GDF_copy.iterrows():
                if debug:
                    print(pred_row.name)
                if pred_row.geometry.area > 0:
                    pred_poly = pred_row.geometry
                    iou_GDF = iou.calculate_iou(pred_poly,
                                                self.ground_truth_GDF_Edit)
                    # Get max iou
                    if not iou_GDF.empty:
                        max_index = iou_GDF['iou_score'].idxmax(axis=0,
                                                                skipna=True)
                        max_iou_row = iou_GDF.loc[max_index]
                        # Update entry in full ground truth table
                        previous_iou = self.ground_truth_GDF.iloc[
                            max_index, iou_index]
                        new_iou = max_iou_row[iou_field]
                        if new_iou > previous_iou:
                            self.ground_truth_GDF.iloc[max_index, iou_index] \
                                = new_iou
                        if max_iou_row['iou_score'] > miniou:
                            self.proposal_GDF.loc[pred_row.name, iou_field] \
                                = max_iou_row['iou_score']
                            self.ground_truth_GDF_Edit \
                                = self.ground_truth_GDF_Edit.drop(
                                    max_iou_row.name, axis=0)
                        else:
                            self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                    else:
                        self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                else:
                    self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                if debug:
                    print(self.proposal_GDF.loc[pred_row.name])

            if self.proposal_GDF.empty:
                TruePos = 0
                FalsePos = 0
            else:
                proposal_GDF_copy = self.proposal_GDF[
                    self.proposal_GDF[imageIDField] == imageID].copy(deep=True)
                proposal_GDF_copy = proposal_GDF_copy[
                    proposal_GDF_copy.area > min_area]
                if not proposal_GDF_copy.empty:
                    if iou_field in proposal_GDF_copy.columns:
                        TruePos = proposal_GDF_copy[
                            proposal_GDF_copy[iou_field] >= miniou].shape[0]
                        FalsePos = proposal_GDF_copy[
                            proposal_GDF_copy[iou_field] < miniou].shape[0]
                    else:
                        print("iou field {} missing".format(iou_field))
                        TruePos = 0
                        FalsePos = 0
                else:
                    print("Empty Proposal Id")
                    TruePos = 0
                    FalsePos = 0

            # false negatives is the number of objects remaining in ground
            # truth after pulling out matched objects
            FalseNeg = self.ground_truth_GDF_Edit[
                self.ground_truth_GDF_Edit.area > 0].shape[0]
            if float(TruePos+FalsePos) > 0:
                Precision = TruePos / float(TruePos + FalsePos)
            else:
                Precision = 0
            if float(TruePos + FalseNeg) > 0:
                Recall = TruePos / float(TruePos + FalseNeg)
            else:
                Recall = 0
            if Recall * Precision > 0:
                F1Score = 2*Precision*Recall/(Precision+Recall)
            else:
                F1Score = 0

            score_calc = {'imageID': imageID,
                          'iou_field': iou_field,
                          'TruePos': TruePos,
                          'FalsePos': FalsePos,
                          'FalseNeg': FalseNeg,
                          'Precision': Precision,
                          'Recall':  Recall,
                          'F1Score': F1Score
                          }
            scoring_dict_list.append(score_calc)

        return scoring_dict_list

    def eval_iou(self, miniou=0.5, iou_field_prefix='iou_score',
                 ground_truth_class_field='', calculate_class_scores=True,
                 class_list=['all']):
        """Evaluate IoU between the ground truth and proposals.

        Arguments
        ---------
        miniou : float, optional
            Minimum intersection over union score to qualify as a successful
            object detection event. Defaults to ``0.5``.
        iou_field_prefix : str, optional
            The name of the IoU score column in ``self.proposal_GDF``. Defaults
            to ``"iou_score"``.
        ground_truth_class_field : str, optional
            The column in ``self.ground_truth_GDF`` that indicates the class of
            each polygon. Required if using ``calculate_class_scores``.
        calculate_class_scores : bool, optional
            Should class-by-class scores be calculated? Defaults to ``True``.
        class_list : list, optional
            List of classes to be scored. Defaults to ``['all']`` (score all
            classes).

        Returns
        -------
        scoring_dict_list : list
            list of score output dicts for each image in the ground
            truth and evaluated image datasets. The dicts contain
            the following keys: ::

                ('class_id', 'iou_field', 'TruePos', 'FalsePos', 'FalseNeg',
                'Precision', 'Recall', 'F1Score')

        """

        scoring_dict_list = []

        if calculate_class_scores:
            if not ground_truth_class_field:
                raise ValueError('Must provide ground_truth_class_field '
                                 'if using calculate_class_scores.')
            if class_list == ['all']:
                class_list = list(
                    self.ground_truth_GDF[ground_truth_class_field].unique())
                if not self.proposal_GDF.empty:
                    class_list.extend(
                        list(self.proposal_GDF['__max_conf_class'].unique()))
                class_list = list(set(class_list))

        for class_id in class_list:
            iou_field = "{}_{}".format(iou_field_prefix, class_id)
            if class_id is not 'all':  # this is probably unnecessary now
                self.ground_truth_GDF_Edit = self.ground_truth_GDF[
                    self.ground_truth_GDF[
                        ground_truth_class_field] == class_id].copy(deep=True)
            else:
                self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(
                    deep=True)

            for _, pred_row in tqdm(self.proposal_GDF.iterrows()):
                if pred_row['__max_conf_class'] == class_id \
                   or class_id == 'all':
                    pred_poly = pred_row.geometry
                    iou_GDF = iou.calculate_iou(pred_poly,
                                                self.ground_truth_GDF_Edit)
                    # Get max iou
                    if not iou_GDF.empty:
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(
                            axis=0, skipna=True)]
                        if max_iou_row['iou_score'] > miniou:
                            self.proposal_GDF.loc[pred_row.name, iou_field] \
                                = max_iou_row['iou_score']
                            self.ground_truth_GDF_Edit \
                                = self.ground_truth_GDF_Edit.drop(
                                    max_iou_row.name, axis=0)
                        else:
                            self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                    else:
                        self.proposal_GDF.loc[pred_row.name, iou_field] = 0

            if self.proposal_GDF.empty:
                TruePos = 0
                FalsePos = 0
            else:
                try:
                    TruePos = self.proposal_GDF[
                        self.proposal_GDF[iou_field] >= miniou].shape[0]
                    FalsePos = self.proposal_GDF[
                        self.proposal_GDF[iou_field] < miniou].shape[0]
                except KeyError:  # handle missing iou_field
                    print("iou field {} missing")
                    TruePos = 0
                    FalsePos = 0

            # number of remaining rows in ground_truth_gdf_edit after removing
            # matches is number of false negatives
            FalseNeg = self.ground_truth_GDF_Edit.shape[0]
            if float(TruePos+FalsePos) > 0:
                Precision = TruePos / float(TruePos + FalsePos)
            else:
                Precision = 0
            if float(TruePos + FalseNeg) > 0:
                Recall = TruePos / float(TruePos + FalseNeg)
            else:
                Recall = 0
            if Recall*Precision > 0:
                F1Score = 2*Precision*Recall/(Precision+Recall)
            else:
                F1Score = 0

            score_calc = {'class_id': class_id,
                          'iou_field': iou_field,
                          'TruePos': TruePos,
                          'FalsePos': FalsePos,
                          'FalseNeg': FalseNeg,
                          'Precision': Precision,
                          'Recall':  Recall,
                          'F1Score': F1Score
                          }
            scoring_dict_list.append(score_calc)

        return scoring_dict_list

    def eval_iou_return_GDFs(self, miniou=0.5, iou_field_prefix='iou_score',
                 ground_truth_class_field='', calculate_class_scores=True,
                 class_list=['all']):
        """Evaluate IoU between the ground truth and proposals.
        Arguments
        ---------
        miniou : float, optional
            Minimum intersection over union score to qualify as a successful
            object detection event. Defaults to ``0.5``.
        iou_field_prefix : str, optional
            The name of the IoU score column in ``self.proposal_GDF``. Defaults
            to ``"iou_score"``.
        ground_truth_class_field : str, optional
            The column in ``self.ground_truth_GDF`` that indicates the class of
            each polygon. Required if using ``calculate_class_scores``.
        calculate_class_scores : bool, optional
            Should class-by-class scores be calculated? Defaults to ``True``.
        class_list : list, optional
            List of classes to be scored. Defaults to ``['all']`` (score all
            classes).
        Returns
        -------
        scoring_dict_list : list
            list of score output dicts for each image in the ground
            truth and evaluated image datasets. The dicts contain
            the following keys: ::
                ('class_id', 'iou_field', 'TruePos', 'FalsePos', 'FalseNeg',
                'Precision', 'Recall', 'F1Score')
        True_Pos_gdf : gdf
            A geodataframe containing only true positive predictions
        False_Neg_gdf : gdf
            A geodataframe containing only false negative predictions
        False_Pos_gdf : gdf
            A geodataframe containing only false positive predictions
        """

        scoring_dict_list = []

        if calculate_class_scores:
            if not ground_truth_class_field:
                raise ValueError('Must provide ground_truth_class_field if using calculate_class_scores.')
            if class_list == ['all']:
                class_list = list(
                    self.ground_truth_GDF[ground_truth_class_field].unique())
                if not self.proposal_GDF.empty:
                    class_list.extend(
                        list(self.proposal_GDF['__max_conf_class'].unique()))
                class_list = list(set(class_list))

        for class_id in class_list:
            iou_field = "{}_{}".format(iou_field_prefix, class_id)
            if class_id is not 'all':  # this is probably unnecessary now
                self.ground_truth_GDF_Edit = self.ground_truth_GDF[
                    self.ground_truth_GDF[
                        ground_truth_class_field] == class_id].copy(deep=True)
            else:
                self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(
                    deep=True)

            for _, pred_row in tqdm(self.proposal_GDF.iterrows()):
                if pred_row['__max_conf_class'] == class_id or class_id == 'all':
                    pred_poly = pred_row.geometry
                    iou_GDF = iou.calculate_iou(pred_poly,
                                                self.ground_truth_GDF_Edit)
                    # Get max iou
                    if not iou_GDF.empty:
                        max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(
                            axis=0, skipna=True)]
                        if max_iou_row['iou_score'] > miniou:
                            self.proposal_GDF.loc[pred_row.name, iou_field] = max_iou_row['iou_score']
                            self.ground_truth_GDF_Edit = self.ground_truth_GDF_Edit.drop(max_iou_row.name, axis=0)
                        else:
                            self.proposal_GDF.loc[pred_row.name, iou_field] = 0
                    else:
                        self.proposal_GDF.loc[pred_row.name, iou_field] = 0

            if self.proposal_GDF.empty:
                TruePos = 0
                FalsePos = 0
            else:
                try:
                    True_Pos_gdf = self.proposal_GDF[
                        self.proposal_GDF[iou_field] >= miniou]
                    TruePos = True_Pos_gdf.shape[0]
                    if TruePos == 0:
                        True_Pos_gdf = None
                    False_Pos_gdf = self.proposal_GDF[
                        self.proposal_GDF[iou_field] < miniou]
                    FalsePos = False_Pos_gdf.shape[0]
                    if FalsePos == 0:
                        False_Pos_gdf = None
                except KeyError:  # handle missing iou_field
                    print("iou field {} missing")
                    TruePos = 0
                    FalsePos = 0
                    False_Pos_gdf = None
                    True_Pos_gdf = None

            # number of remaining rows in ground_truth_gdf_edit after removing
            # matches is number of false negatives
            False_Neg_gdf = self.ground_truth_GDF_Edit
            FalseNeg = False_Neg_gdf.shape[0]
            if FalseNeg == 0:
                False_Neg_gdf = None
            if float(TruePos + FalsePos) > 0:
                Precision = TruePos / float(TruePos + FalsePos)
            else:
                Precision = 0
            if float(TruePos + FalseNeg) > 0:
                Recall = TruePos / float(TruePos + FalseNeg)
            else:
                Recall = 0
            if Recall * Precision > 0:
                F1Score = 2 * Precision * Recall / (Precision + Recall)
            else:
                F1Score = 0

            score_calc = {'class_id': class_id,
                          'iou_field': iou_field,
                          'TruePos': TruePos,
                          'FalsePos': FalsePos,
                          'FalseNeg': FalseNeg,
                          'Precision': Precision,
                          'Recall': Recall,
                          'F1Score': F1Score
                          }
            scoring_dict_list.append(score_calc)

        return scoring_dict_list, True_Pos_gdf, False_Neg_gdf, False_Pos_gdf

    def load_proposal(self, proposal_vector_file, conf_field_list=['conf'],
                      proposalCSV=False, pred_row_geo_value='PolygonWKT_Pix',
                      conf_field_mapping=None):
        """Load in a proposal geojson or CSV.

        Arguments
        ---------
        proposal_vector_file : str
            Path to the file containing proposal vector objects. This can be
            a .geojson or a .csv.
        conf_field_list : list, optional
            List of columns corresponding to confidence value(s) in the
            proposal vector file. Defaults to ``['conf']``.
        proposalCSV : bool, optional
            Is the proposal file a CSV? Defaults to no (``False``), in which
            case it's assumed to be a .geojson.
        pred_row_geo_value : str, optional
            The name of the geometry-containing column in the proposal vector
            file. Defaults to ``'PolygonWKT_Pix'``. Note: this method assumes
            the geometry is in WKT format.
        conf_field_mapping : dict, optional
            ``'__max_conf_class'`` column value:class ID mapping dict for
            multiclass use. Only required in multiclass cases.

        Returns
        -------
        ``0`` upon successful completion.

        Notes
        -----
        Loads in a .geojson or .csv-formatted file of proposal polygons for
        comparison to the ground truth and stores it as part of the
        ``Evaluator`` instance. This method assumes the geometry contained in
        the proposal file is in WKT format.

        """

        # Load Proposal if proposal_vector_file is a path to a file
        if os.path.isfile(proposal_vector_file):
            # if it's a CSV format, first read into a pd df and then convert
            # to gpd gdf by loading in geometries using shapely
            if proposalCSV:
                pred_data = pd.read_csv(proposal_vector_file)
                self.proposal_GDF = gpd.GeoDataFrame(
                    pred_data, geometry=[
                        shapely.wkt.loads(pred_row[pred_row_geo_value])
                        for idx, pred_row in pred_data.iterrows()
                        ]
                    )
            else:  # if it's a .geojson
                try:
                    self.proposal_GDF = gpd.read_file(
                        proposal_vector_file).dropna()
                except (CPLE_OpenFailedError, DriverError):
                    self.proposal_GDF = gpd.GeoDataFrame(geometry=[])

            if conf_field_list:
                self.proposal_GDF['__total_conf'] = self.proposal_GDF[
                    conf_field_list].max(axis=1)
                self.proposal_GDF['__max_conf_class'] = self.proposal_GDF[
                    conf_field_list].idxmax(axis=1)
            else:
                # set arbitrary (meaningless) values otherwise
                self.proposal_GDF['__total_conf'] = 1.0
                self.proposal_GDF['__max_conf_class'] = 1

            if conf_field_mapping is not None:
                self.proposal_GDF['__max_conf_class'] = [
                    conf_field_mapping[item] for item in
                    self.proposal_GDF['__max_conf_class'].values]
            self.proposal_GDF = self.proposal_GDF.sort_values(
                by='__total_conf', ascending=False)
        else:
            self.proposal_GDF = gpd.GeoDataFrame(geometry=[])

    def load_truth(self, ground_truth_vector_file, truthCSV=False,
                   truth_geo_value='PolygonWKT_Pix'):
        """Load in the ground truth geometry data.

        Arguments
        ---------
        ground_truth_vector_file : str
            Path to the ground truth vector file. Must be either .geojson or
            .csv format.
        truthCSV : bool, optional
            Is the ground truth a CSV? Defaults to ``False``, in which case
            it's assumed to be a .geojson.
        truth_geo_value : str, optional
            Column of the ground truth vector file that corresponds to
            geometry.

        Returns
        -------
        Nothing.

        Notes
        -----
        Loads the ground truth vector data into the ``Evaluator`` instance.

        """
        if truthCSV:
            truth_data = pd.read_csv(ground_truth_vector_file)
            self.ground_truth_GDF = gpd.GeoDataFrame(
                truth_data, geometry=[
                    shapely.wkt.loads(truth_row[truth_geo_value])
                    for idx, truth_row in truth_data.iterrows()])
        else:
            try:
                self.ground_truth_GDF = gpd.read_file(ground_truth_vector_file)
            except (CPLE_OpenFailedError, DriverError):  # empty geojson
                self.ground_truth_GDF = gpd.GeoDataFrame({'sindex': [],
                                                          'condition': [],
                                                          'geometry': []})
        # force calculation of spatialindex
        self.ground_truth_sindex = self.ground_truth_GDF.sindex
        # create deep copy of ground truth file for calculations
        self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(deep=True)

    def eval(self, type='iou'):
        pass


def eval_base(ground_truth_vector_file, csvFile=False,
              truth_geo_value='PolygonWKT_Pix'):
    """Deprecated API to Evaluator.

    .. deprecated:: 0.3
        Use :class:`Evaluator` instead."""

    return Evaluator(ground_truth_vector_file)
