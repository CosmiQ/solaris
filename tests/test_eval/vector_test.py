import os
from solaris.data import data_dir
from solaris.eval import vector


class TestVectorMetrics(object):
    """Test the vector metrics."""

    def test_vector_metrics(self):
        proposal_polygons_dir = os.path.join(data_dir, "eval_vector/preds/")
        gt_polygons_dir = os.path.join(data_dir, "eval_vector/gt/")
        mAP, APs_by_class, mF1_score, f1s_by_class, precision_iou_by_obj, precision_by_class, mPrecision, recall_iou_by_obj, recall_by_class, mRecall, object_subset, confidences = vector.mAP_score(proposal_polygons_dir, gt_polygons_dir, prediction_cat_attrib="class", gt_cat_attrib='make')
        assert mAP.round(2) == 0.85
