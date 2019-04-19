import os
from cw_eval.baseeval import EvalBase
import cw_eval
import geopandas as gpd


class TestEvalBase(object):
    def test_init_from_file(self):
        """Test instantiation of an EvalBase instance from a file."""
        base_instance = EvalBase(os.path.join(cw_eval.data.data_dir,
                                              'gt.geojson'))
        gdf = cw_eval.data.gt_gdf()
        assert base_instance.ground_truth_sindex.bounds == gdf.sindex.bounds
        assert base_instance.proposal_GDF.equals(gpd.GeoDataFrame([]))
        assert base_instance.ground_truth_GDF.equals(base_instance.ground_truth_GDF_Edit)

    def test_init_from_gdf(self):
        """Test instantiation of an EvalBase from a pre-loaded GeoDataFrame."""
        gdf = cw_eval.data.gt_gdf()
        base_instance = EvalBase(gdf)
        assert base_instance.ground_truth_sindex.bounds == gdf.sindex.bounds
        assert base_instance.proposal_GDF.equals(gpd.GeoDataFrame([]))
        assert base_instance.ground_truth_GDF.equals(base_instance.ground_truth_GDF_Edit)

    def test_init_empty_geojson(self):
        """Test instantiation of EvalBase with an empty geojson file."""
        base_instance = EvalBase(os.path.join(cw_eval.data.data_dir,
                                              'empty.geojson'))
        expected_gdf = gpd.GeoDataFrame({'sindex': [],
                                         'condition': [],
                                         'geometry': []})
        assert base_instance.ground_truth_GDF.equals(expected_gdf)

    def test_score_proposals(self):
        """Test reading in a proposal GDF from a geojson and scoring it."""
        eb = EvalBase(os.path.join(cw_eval.data.data_dir, 'gt.geojson'))
        eb.load_proposal(os.path.join(cw_eval.data.data_dir, 'pred.geojson'))
        pred_gdf = cw_eval.data.pred_gdf()
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
