import os
import pickle

import networkx as nx

from solaris.vector.graph import geojson_to_graph

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))


class TestGeojsonToGraph(object):
    """Tests for cw_geodata.vector_label.graph.geojson_to_graph."""

    def test_graph_creation(self):
        """Test if a newly created graph is identical to an existing one."""
        with open(os.path.join(data_dir, "sample_graph.pkl"), "rb") as f:
            truth_graph = pickle.load(f)
            f.close()
        output_graph = geojson_to_graph(os.path.join(data_dir, "sample_roads.geojson"))

        assert nx.is_isomorphic(truth_graph, output_graph)
