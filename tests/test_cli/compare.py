import os
import numpy as np
import skimage
import pickle
import networkx as nx
import sys

im_fnames = ['sample_fp_mask.tif',
             'sample_fbc_mask.tif',
             'sample_c_mask.tif',
             'sample_b_inner_mask.tif',
             'sample_b_outer10_mask.tif']


def main(path):
    os.chdir(path)  # set to the directory containing this script

    # compare expected mask results and truth images
    for im_fname in im_fnames:
        truth_im = skimage.io.imread(os.path.join('expected', im_fname))
        result_im = skimage.io.imread(os.path.join('results', im_fname))
        assert np.array_equal(truth_im, result_im)

    # compare graphs
    with open(os.path.join('expected', 'sample_graph.pkl'), 'rb') as f:
        truth_graph = pickle.load(f)
        f.close()
    with open(os.path.join('results', 'sample_graph.pkl'), 'rb') as f:
        result_graph = pickle.load(f)
        f.close()

    assert nx.is_isomorphic(truth_graph, result_graph)


if __name__ == '__main__':
    main(sys.argv[1])
