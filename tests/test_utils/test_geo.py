import os
import pandas as pd
import geopandas as gpd
import shapely
from affine import Affine
from shapely.wkt import loads
from shapely.ops import cascaded_union
from solaris.data import data_dir
from solaris.utils.core import _check_gdf_load
from solaris.utils.geo import list_to_affine, split_multi_geometries
from solaris.utils.geo import geometries_internal_intersection, split_geom
from solaris.utils.geo import reproject, reproject_geometry
import rasterio


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
            data_dir, 'split_multi_result.json'))

        assert expected.equals(output)

    def test_grouped_split_multipolygon(self):
        output = split_multi_geometries(
            os.path.join(data_dir, 'w_multipolygon.csv'), obj_id_col='field_1',
            group_col='truncated')
        expected = gpd.read_file(os.path.join(
            data_dir, 'split_multi_grouped_result.json'))

        assert expected.equals(output)


class TestReproject(object):
    """Test reprojection functionality."""

    def test_reproject_rasterio_dataset(self):
        input_data = os.path.join(data_dir, 'sample_geotiff.tif')
        output = reproject(input_data, target_crs=4326,
                           dest_path=os.path.join(data_dir, 'tmp.tiff'))
        with rasterio.open(input_data) as input_rio:
            input_bounds = input_rio.bounds
            expected_bounds = rasterio.warp.transform_bounds(input_rio.crs,
                                                             'EPSG:4326',
                                                             *input_bounds)
        expected_bounds = tuple([round(i, 4) for i in tuple(expected_bounds)])
        output_bounds = tuple([round(i, 4) for i in tuple(output.bounds)])

        assert expected_bounds == output_bounds
        assert output.crs.to_epsg() == 4326

        os.remove(os.path.join(data_dir, 'tmp.tiff'))

    def test_reproject_gdf(self):
        input_data = os.path.join(data_dir, 'gt.geojson')
        output = reproject(input_data, target_crs=4326,
                           dest_path=os.path.join(data_dir, 'tmp.json'))
        expected_result = gpd.read_file(os.path.join(data_dir,
                                                     'gt_epsg4326.json'))
        out_geoms = cascaded_union(output.geometry)
        exp_geoms = cascaded_union(expected_result.geometry)

        assert out_geoms.intersection(exp_geoms).area/out_geoms.area > 0.99999
        os.remove(os.path.join(data_dir, 'tmp.json'))

    def test_reproject_gdf_utm_default(self):
        input_data = os.path.join(data_dir, 'gt_epsg4326.json')
        output = reproject(input_data)
        expected_result = gpd.read_file(os.path.join(data_dir, 'gt.geojson'))
        out_geoms = cascaded_union(output.geometry)
        exp_geoms = cascaded_union(expected_result.geometry)

        assert out_geoms.intersection(exp_geoms).area/out_geoms.area > 0.99999


class TestReprojectGeometry(object):
    """Test reprojection of single geometries."""

    def test_reproject_from_wkt(self):
        input_str = "POLYGON ((736687.5456353347 3722455.06780279, 736686.9301210654 3722464.96326352, 736691.6397869177 3722470.9059681, 736705.5443059544 3722472.614050498, 736706.8992101226 3722462.858909504, 736704.866059878 3722459.457111885, 736713.1443474176 3722452.103498172, 736710.0312805283 3722447.309985571, 736700.3886167214 3722454.263705271, 736698.4577440721 3722451.98534527, 736690.1272768064 3722451.291527834, 736689.4108667439 3722455.113813923, 736687.5456353347 3722455.06780279))"
        result_str = "POLYGON ((-84.4487639 33.6156071, -84.44876790000001 33.6156964, -84.4487156 33.61574889999999, -84.44856540000001 33.6157612, -84.44855339999999 33.61567300000001, -84.44857620000001 33.6156428, -84.448489 33.6155747, -84.4485238 33.6155322, -84.4486258 33.615597, -84.4486472 33.61557689999999, -84.4487371 33.6155725, -84.4487438 33.6156071, -84.4487639 33.6156071))"
        result_geom = loads(result_str)
        reproj_geom = reproject_geometry(input_str, input_crs=32616,
                                         target_crs=4326)
        area_sim = result_geom.intersection(reproj_geom).area/result_geom.area

        assert area_sim > 0.99999

    def test_reproject_from_wkt_to_utm(self):
        result_str = "POLYGON ((736687.5456353347 3722455.06780279, 736686.9301210654 3722464.96326352, 736691.6397869177 3722470.9059681, 736705.5443059544 3722472.614050498, 736706.8992101226 3722462.858909504, 736704.866059878 3722459.457111885, 736713.1443474176 3722452.103498172, 736710.0312805283 3722447.309985571, 736700.3886167214 3722454.263705271, 736698.4577440721 3722451.98534527, 736690.1272768064 3722451.291527834, 736689.4108667439 3722455.113813923, 736687.5456353347 3722455.06780279))"
        input_str = "POLYGON ((-84.4487639 33.6156071, -84.44876790000001 33.6156964, -84.4487156 33.61574889999999, -84.44856540000001 33.6157612, -84.44855339999999 33.61567300000001, -84.44857620000001 33.6156428, -84.448489 33.6155747, -84.4485238 33.6155322, -84.4486258 33.615597, -84.4486472 33.61557689999999, -84.4487371 33.6155725, -84.4487438 33.6156071, -84.4487639 33.6156071))"
        result_geom = loads(result_str)
        reproj_geom = reproject_geometry(input_str, input_crs=4326,
                                         target_crs=32616)
        area_sim = result_geom.intersection(reproj_geom).area/result_geom.area

        assert area_sim > 0.99999


class TestSplitGeometry(object):
    """Test splitting of single geometries. Used in RasterTiler"""

    def test_split_polygon(self):

        poly = gpd.read_file(os.path.join(
            data_dir, 'test_polygon_split.geojson')).iloc[0]['geometry']
        reproj_poly = reproject_geometry(poly, input_crs=4326,
                                         target_crs=32611)
        split_geom_list = split_geom(reproj_poly, (1024,1024), resolution=30)
        assert len(split_geom_list) == 47

    def test_split_multigeom_gdf(self):

        multi_gdf = _check_gdf_load(
            os.path.join(data_dir, 'multigeom.geojson'))
        expected_result = _check_gdf_load(
            os.path.join(data_dir, 'multigeom_split_result.geojson'))
        single_gdf = split_multi_geometries(multi_gdf)

        assert single_gdf.equals(expected_result)
