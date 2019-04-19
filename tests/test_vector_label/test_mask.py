import os
import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from shapely.geometry import Polygon
import skimage
import rasterio
from cw_geodata.data import data_dir
from cw_geodata.vector_label.mask import footprint_mask, boundary_mask, \
    contact_mask, df_to_px_mask, mask_to_poly_geojson


class TestFootprintMask(object):
    """Tests for cw_geodata.vector_label.mask.footprint_mask."""

    def test_make_mask(self):
        """test creating a basic mask using a csv input."""
        output_mask = footprint_mask(os.path.join(data_dir, 'sample.csv'),
                                     geom_col="PolygonWKT_Pix")
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_fp_mask.tif'))

        assert np.array_equal(output_mask, truth_mask)

    def test_make_mask_w_output_file(self):
        """test creating a basic mask and saving the output to a file."""
        output_mask = footprint_mask(
            os.path.join(data_dir, 'sample.csv'),
            geom_col="PolygonWKT_Pix",
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif')
            )
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_fp_mask.tif'))
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask)
        # clean up
        os.remove(os.path.join(data_dir, 'test_out.tif'))

    def test_make_mask_w_file_and_transform(self):
        """Test creating a mask using a geojson and an affine xform."""
        output_mask = footprint_mask(
            os.path.join(data_dir, 'geotiff_labels.geojson'),
            reference_im=os.path.join(data_dir, 'sample_geotiff.tif'),
            do_transform=True,
            out_file=os.path.join(data_dir, 'test_out.tif')
            )
        truth_mask = skimage.io.imread(
            os.path.join(data_dir, 'sample_fp_mask_from_geojson.tif')
            )
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask)
        # clean up
        os.remove(os.path.join(data_dir, 'test_out.tif'))


class TestBoundaryMask(object):
    """Tests for cw_geodata.vector_label.mask.boundary_mask."""

    def test_make_inner_mask_from_fp(self):
        """test creating a boundary mask using an existing footprint mask."""
        fp_mask = skimage.io.imread(os.path.join(data_dir,
                                                 'sample_fp_mask.tif'))
        output_mask = boundary_mask(fp_mask)
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_b_mask_inner.tif'))

        assert np.array_equal(output_mask, truth_mask)

    def test_make_outer_mask_from_fp(self):
        """test creating a boundary mask using an existing footprint mask."""
        fp_mask = skimage.io.imread(os.path.join(data_dir,
                                                 'sample_fp_mask.tif'))
        output_mask = boundary_mask(fp_mask, boundary_type="outer")
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_b_mask_outer.tif'))

        assert np.array_equal(output_mask, truth_mask)

    def test_make_thick_outer_mask_from_fp(self):
        """test creating a 10-px thick boundary mask."""
        fp_mask = skimage.io.imread(os.path.join(data_dir,
                                                 'sample_fp_mask.tif'))
        output_mask = boundary_mask(fp_mask, boundary_type="outer",
                                    boundary_width=10)
        truth_mask = skimage.io.imread(
            os.path.join(data_dir, 'sample_b_mask_outer_10.tif')
            )

        assert np.array_equal(output_mask, truth_mask)

    def test_make_binary_and_fp(self):
        """test creating a boundary mask and a fp mask together."""
        output_mask = boundary_mask(df=os.path.join(data_dir, 'sample.csv'),
                                    geom_col="PolygonWKT_Pix")
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_b_mask_inner.tif'))

        assert np.array_equal(output_mask, truth_mask)


class TestContactMask(object):
    """Tests for cw_geodata.vector_label.mask.contact_mask."""

    def test_make_contact_mask_w_save(self):
        """test creating a contact point mask."""
        output_mask = contact_mask(
            os.path.join(data_dir, 'sample.csv'), geom_col="PolygonWKT_Pix",
            contact_spacing=10,
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif')
            )
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_c_mask.tif'))
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask)
        os.remove(os.path.join(data_dir, 'test_out.tif'))  # clean up after


class TestDFToPxMask(object):
    """Tests for cw_geodata.vector_label.mask.df_to_px_mask."""

    def test_basic_footprint_w_save(self):
        output_mask = df_to_px_mask(
            os.path.join(data_dir, 'sample.csv'),
            geom_col='PolygonWKT_Pix',
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif'))
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_fp_from_df2px.tif')
                                       )
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask[:, :, 0])
        os.remove(os.path.join(data_dir, 'test_out.tif'))  # clean up after

    def test_border_footprint_w_save(self):
        output_mask = df_to_px_mask(
            os.path.join(data_dir, 'sample.csv'), channels=['boundary'],
            geom_col='PolygonWKT_Pix',
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif'))
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_b_from_df2px.tif')
                                       )
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask[:, :, 0])
        os.remove(os.path.join(data_dir, 'test_out.tif'))  # clean up after

    def test_contact_footprint_w_save(self):
        output_mask = df_to_px_mask(
            os.path.join(data_dir, 'sample.csv'), channels=['contact'],
            geom_col='PolygonWKT_Pix',
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif'))
        truth_mask = skimage.io.imread(os.path.join(data_dir,
                                                    'sample_c_from_df2px.tif')
                                       )
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask[:, :, 0])
        os.remove(os.path.join(data_dir, 'test_out.tif'))  # clean up after

    def test_all_three_w_save(self):
        """Test creating a 3-channel mask."""
        output_mask = df_to_px_mask(
            os.path.join(data_dir, 'sample.csv'),
            channels=['footprint', 'boundary', 'contact'],
            boundary_type='outer', boundary_width=5, contact_spacing=15,
            geom_col='PolygonWKT_Pix',
            reference_im=os.path.join(data_dir, "sample_geotiff.tif"),
            out_file=os.path.join(data_dir, 'test_out.tif'))
        truth_mask = skimage.io.imread(
            os.path.join(data_dir, 'sample_fbc_from_df2px.tif')
            )
        saved_output_mask = skimage.io.imread(os.path.join(data_dir,
                                                           'test_out.tif'))

        assert np.array_equal(output_mask, truth_mask)
        assert np.array_equal(saved_output_mask, truth_mask)
        os.remove(os.path.join(data_dir, 'test_out.tif'))  # clean up after


class TestMaskToGDF(object):
    """Tests for converting pixel masks to geodataframes or geojsons."""

    def test_mask_to_gdf_basic(self):
        gdf = mask_to_poly_geojson(
            os.path.join(data_dir, 'sample_fp_mask_from_geojson.tif'))
        truth_gdf = gpd.read_file(os.path.join(data_dir,
                                               'gdf_from_mask_1.geojson'))
        assert truth_gdf[['geometry', 'value']].equals(gdf)

    def test_mask_to_gdf_geoxform_simplified(self):
        gdf = mask_to_poly_geojson(
            os.path.join(data_dir, 'sample_fp_mask_from_geojson.tif'),
            reference_im=os.path.join(data_dir, 'sample_geotiff.tif'),
            do_transform=True,
            min_area=100,
            simplify=True
            )
        truth_gdf = gpd.read_file(os.path.join(data_dir,
                                               'gdf_from_mask_2.geojson'))
        assert truth_gdf[['geometry', 'value']].equals(gdf)
