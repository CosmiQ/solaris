import os
import numpy as np
import pickle
import subprocess
import skimage
from cw_geodata.data import data_dir
import networkx as nx


class TestCLI(object):
    """CLI tests using subprocess."""
    def test_geotransform_footprints(self):
        """test the geotransform_footprints CLI command."""
        # make sure the result directory is empty
        dest_loc = os.path.join(data_dir, 'cli_test', 'result',
                                'to_px_test.geojson')
        if os.path.exists(dest_loc):
            os.remove(dest_loc)
        # run the CLI command
        subprocess.run('geotransform_footprints -s ' +
                       os.path.join(data_dir, 'geotiff_labels.geojson') +
                       ' -r ' +
                       os.path.join(data_dir, 'sample_geotiff.tif') +
                       ' -o ' +
                       dest_loc +
                       ' -p -d 0',
                       shell=True)
        # compare results
        subprocess.run('diff ' + os.path.join(data_dir, 'cli_test', 'expected',
                                              'gj_to_px_result.geojson') +
                       ' ' + data_dir,
                       shell=True)
        # clean up
        os.remove(dest_loc)

    def test_make_graphs(self):
        """test the make_graphs CLI command."""
        # prep paths and clear existing result
        dest_loc = os.path.join(data_dir, 'cli_test', 'result',
                                'sample_graph.pkl')
        src_loc = os.path.join(data_dir, 'sample_roads.geojson')
        truth_loc = os.path.join(data_dir, 'cli_test', 'expected',
                                 'sample_graph.pkl')
        if os.path.exists(dest_loc):
            os.remove(dest_loc)
        # run the CLI command
        subprocess.run('make_graphs -s ' + src_loc + ' -o ' + dest_loc,
                       shell=True)
        with open(truth_loc, 'rb') as f:
            truth_graph = pickle.load(f)
            f.close()
        with open(dest_loc, 'rb') as f:
            result_graph = pickle.load(f)
            f.close()

        assert nx.is_isomorphic(truth_graph, result_graph)
        # clean up
        os.remove(dest_loc)

    def test_make_masks(self):
        """Test the make_masks CLI command."""
        # set up variables
        args = (('sample_fp_mask.tif', ' -f '),
                ('sample_b_inner_mask.tif', ' -e '),
                ('sample_b_outer10_mask.tif', ' -e -et outer -ew 10 '),
                ('sample_c_mask.tif', ' -c -cs 10 '),
                ('sample_fbc_mask.tif', ' -f -e -c -et outer -ew 5 -cs 15 '))
        dest_dir = os.path.join(data_dir, 'cli_test/result')
        expected_dir = os.path.join(data_dir, 'cli_test/expected')
        src_vector_path = os.path.join(data_dir, 'sample.csv')
        src_geotiff_path = os.path.join(data_dir, 'sample_geotiff.tif')
        cmd_start = 'make_masks -s ' + src_vector_path + ' -r ' + src_geotiff_path + ' -g PolygonWKT_Pix -o '
        for im_fname, arg in args:
            # clean up destination
            if os.path.exists(os.path.join(dest_dir, im_fname)):
                os.remove(os.path.join(dest_dir, im_fname))
            # run the CLI command
            subprocess.run(cmd_start + os.path.join(dest_dir, im_fname) + arg,
                           shell=True)
            truth_im = skimage.io.imread(os.path.join(expected_dir, im_fname))
            result_im = skimage.io.imread(os.path.join(dest_dir, im_fname))
            # compare the expected to the result
            assert np.array_equal(truth_im, result_im)
            # clean up after
            os.remove(os.path.join(dest_dir, im_fname))
