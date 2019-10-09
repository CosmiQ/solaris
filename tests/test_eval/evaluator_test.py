import os
from solaris.eval.base import Evaluator
import solaris
import geopandas as gpd
import pandas as pd


class TestEvaluator(object):
    def test_init_from_file(self):
        """Test instantiation of an Evaluator instance from a file."""
        base_instance = Evaluator(os.path.join(solaris.data.data_dir,
                                               'gt.geojson'))
        gdf = solaris.data.gt_gdf()
        assert base_instance.ground_truth_sindex.bounds == gdf.sindex.bounds
        assert base_instance.proposal_GDF.equals(gpd.GeoDataFrame([]))
        assert base_instance.ground_truth_GDF.equals(
            base_instance.ground_truth_GDF_Edit)

    def test_init_from_gdf(self):
        """Test instantiation of an Evaluator from a pre-loaded GeoDataFrame."""
        gdf = solaris.data.gt_gdf()
        base_instance = Evaluator(gdf)
        assert base_instance.ground_truth_sindex.bounds == gdf.sindex.bounds
        assert base_instance.proposal_GDF.equals(gpd.GeoDataFrame([]))
        assert base_instance.ground_truth_GDF.equals(
            base_instance.ground_truth_GDF_Edit)

    def test_init_empty_geojson(self):
        """Test instantiation of Evaluator with an empty geojson file."""
        base_instance = Evaluator(os.path.join(solaris.data.data_dir,
                                               'empty.geojson'))
        expected_gdf = gpd.GeoDataFrame({'sindex': [],
                                         'condition': [],
                                         'geometry': []})
        assert base_instance.ground_truth_GDF.equals(expected_gdf)

    def test_score_proposals(self):
        """Test reading in a proposal GDF from a geojson and scoring it."""
        eb = Evaluator(os.path.join(solaris.data.data_dir, 'gt.geojson'))
        eb.load_proposal(os.path.join(solaris.data.data_dir, 'pred.geojson'))
        pred_gdf = solaris.data.pred_gdf()
        assert eb.proposal_GDF.iloc[:, 0:3].sort_index().equals(pred_gdf)
        expected_score = [{'class_id': 'all',
                           'iou_field': 'iou_score_all',
                           'TruePos': 8,
                           'FalsePos': 20,
                           'FalseNeg': 20,
                           'Precision': 0.2857142857142857,
                           'Recall': 0.2857142857142857,
                           'F1Score': 0.2857142857142857}]
        scores = eb.eval_iou(calculate_class_scores=False)
        assert scores == expected_score

    def test_score_proposals_return_gdfs(self):
        eb = Evaluator(os.path.join(solaris.data.data_dir, 'gt.geojson'))
        eb.load_proposal(os.path.join(solaris.data.data_dir, 'pred.geojson'))
        expected_score = [{'class_id': 'all',
                           'iou_field': 'iou_score_all',
                           'TruePos': 8,
                           'FalsePos': 20,
                           'FalseNeg': 20,
                           'Precision': 0.2857142857142857,
                           'Recall': 0.2857142857142857,
                           'F1Score': 0.2857142857142857}]
        scores, tp_gdf, fn_gdf, fp_gdf = eb.eval_iou_return_GDFs(
            calculate_class_scores=False)
        assert scores == expected_score
        assert len(tp_gdf) == expected_score[0]['TruePos']
        assert len(fp_gdf) == expected_score[0]['FalsePos']
        assert len(fn_gdf) == expected_score[0]['FalseNeg']

    def test_iou_by_building(self):
        """Test output of ground truth table with per-building IoU scores"""
        data_folder = solaris.data.data_dir
        path_truth = os.path.join(data_folder, 'SN2_sample_truth.csv')
        path_pred = os.path.join(data_folder, 'SN2_sample_preds.csv')
        path_ious = os.path.join(data_folder, 'SN2_sample_iou_by_building.csv')
        path_temp = './temp.pd'
        eb = Evaluator(path_truth)
        eb.load_proposal(path_pred, conf_field_list=['Confidence'],
                         proposalCSV=True)
        eb.eval_iou_spacenet_csv(miniou=0.5, imageIDField='ImageId',
                                 min_area=20)
        output = eb.get_iou_by_building()
        result_actual = pd.DataFrame(output)
        result_actual.sort_values(by=['ImageId', 'BuildingId'], inplace=True)
        ious_actual = list(result_actual['iou_score'])
        result_expected = pd.read_csv(path_ious, index_col=0)
        result_expected.sort_values(by=['ImageId', 'BuildingId'], inplace=True)
        ious_expected = list(result_expected['iou_score'])
        maxdifference = max([abs(x-y) for x, y in zip(ious_actual,
                                                      ious_expected)])
        epsilon = 1E-9
        assert maxdifference < epsilon
