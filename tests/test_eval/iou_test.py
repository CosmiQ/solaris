from solaris.eval.iou import calculate_iou, process_iou
from solaris import data
from shapely.geometry import Polygon


class TestEvalFuncs(object):
    def test_overlap(self):
        gt_gdf = data.gt_gdf()
        pred_poly = Polygon(((736348.0, 3722762.5),
                             (736353.0, 3722762.0),
                             (736354.0, 3722759.0),
                             (736352.0, 3722755.5),
                             (736348.5, 3722755.5),
                             (736346.0, 3722757.5),
                             (736348.0, 3722762.5)))
        overlap_pred_gdf = calculate_iou(pred_poly, gt_gdf)
        assert overlap_pred_gdf.index[0] == 27
        assert overlap_pred_gdf.iou_score.iloc[0] == 0.073499798744833519

    def test_process_iou(self):
        gt_gdf = data.gt_gdf()
        pred_poly = Polygon(((736414.0, 3722573.0),
                             (736417.5, 3722572.5),
                             (736420.0, 3722568.0),
                             (736421.0, 3722556.0),
                             (736418.5, 3722538.0),
                             (736424.0, 3722532.5),
                             (736424.0, 3722527.0),
                             (736422.5, 3722525.5),
                             (736412.0, 3722524.0),
                             (736410.5, 3722521.5),
                             (736407.0, 3722520.5),
                             (736383.5, 3722521.0),
                             (736376.5, 3722528.5),
                             (736378.0, 3722532.5),
                             (736402.0, 3722532.0),
                             (736410.0, 3722539.0),
                             (736411.0, 3722544.0),
                             (736408.5, 3722553.5),
                             (736409.0, 3722569.0),
                             (736414.0, 3722573.0)))
        assert 21 in gt_gdf.index
        process_iou(pred_poly, gt_gdf)
        assert 21 not in gt_gdf.index
