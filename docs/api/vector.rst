.. title:: solaris.vector API reference

``solaris.vector`` API reference
================================

.. contents::

``solaris.vector`` class and function list
------------------------------------------

.. autosummary::

   solaris.vector.polygon.convert_poly_coords
   solaris.vector.polygon.affine_transform_gdf
   solaris.vector.polygon.georegister_px_df
   solaris.vector.polygon.geojson_to_px_gdf
   solaris.vector.polygon.get_overlapping_subset
   solaris.vector.polygon.gdf_to_yolo
   solaris.vector.graph.Node
   solaris.vector.graph.Edge
   solaris.vector.graph.Path
   solaris.vector.graph.geojson_to_graph
   solaris.vector.graph.get_nodes_paths
   solaris.vector.graph.parallel_linestring_to_path
   solaris.vector.graph.linestring_to_edges
   solaris.vector.graph.graph_to_geojson
   solaris.vector.mask.df_to_px_mask
   solaris.vector.mask.footprint_mask
   solaris.vector.mask.boundary_mask
   solaris.vector.mask.contact_mask
   solaris.vector.mask.mask_to_poly_geojson
   solaris.vector.mask.road_mask

``solaris.vector.polygon`` vector polygon management
----------------------------------------------------

.. automodule:: solaris.vector.polygon
   :members:

``solaris.vector.graph`` graph and road network analysis
--------------------------------------------------------

.. automodule:: solaris.vector.graph
   :members:

``solaris.vector.mask`` vector <-> training mask interconversion
----------------------------------------------------------------

.. automodule:: solaris.vector.mask
  :members:
