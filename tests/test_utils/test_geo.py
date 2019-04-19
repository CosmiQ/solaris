import os
import pandas as pd
import geopandas as gpd
import shapely
from affine import Affine
from shapely.wkt import loads
from cw_geodata.data import data_dir
from cw_geodata.utils.geo import list_to_affine, split_multi_geometries, \
    geometries_internal_intersection


class TestCoordTransformer(object):
    """Tests for the utils.geo.CoordTransformer."""

    def test_convert_image_crs(self):
        pass


class TestListToAffine(object):
    """Tests for utils.geo.list_to_affine()."""

    def test_rasterio_order_list(self):
        truth_affine_obj = Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
        affine_list = [0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0]
        test_affine = list_to_affine(affine_list)

        assert truth_affine_obj == test_affine

    def test_gdal_order_list(self):
        truth_affine_obj = Affine(0.5, 0.0, 733601.0, 0.0, -0.5, 3725139.0)
        gdal_affine_list = [733601.0, 0.5, 0.0, 3725139.0, 0.0, -0.5]
        test_affine = list_to_affine(gdal_affine_list)

        assert truth_affine_obj == test_affine


class TestGeometriesInternalIntersection(object):
    """Tests for utils.geo.geometries_internal_intersection."""

    def test_no_overlap(self):
        """Test creation of an overlap object with no intersection."""
        poly_df = pd.read_csv(os.path.join(data_dir, 'sample.csv'))
        polygons = poly_df['PolygonWKT_Pix'].apply(loads).values
        preds = geometries_internal_intersection(polygons)
        # there's no overlap, so result should be an empty GeometryCollection
        assert preds == shapely.geometry.collection.GeometryCollection()

    def test_with_overlap(self):
        poly_df = pd.read_csv(os.path.join(data_dir, 'sample.csv'))
        # expand the polygons to generate some overlap
        polygons = poly_df['PolygonWKT_Pix'].apply(
            lambda x: loads(x).buffer(15)).values
        preds = geometries_internal_intersection(polygons)
        with open(os.path.join(data_dir, 'test_overlap_output.txt'), 'r') as f:
            truth = f.read()
            f.close()
        truth = loads(truth)
        # set a threshold for how good overlap with truth has to be in case of
        # rounding errors
        assert truth.intersection(preds).area/truth.area > 0.99


class TestSplitMultiGeometries(object):
    """Test for splittling MultiPolygons."""

    def test_simple_split_multipolygon(self):
        output = split_multi_geometries(os.path.join(data_dir,
                                                     'w_multipolygon.csv'))
        expected = gpd.read_file(os.path.join(
            data_dir, 'split_multi_result.json')).drop(columns='id')

        assert expected.equals(output)

    def test_grouped_split_multipolygon(self):
        output = split_multi_geometries(
            os.path.join(data_dir, 'w_multipolygon.csv'), obj_id_col='field_1',
            group_col='truncated')
        expected = gpd.read_file(os.path.join(
            data_dir, 'split_multi_grouped_result.json')).drop(columns='id')

        assert expected.equals(output)
