.. title:: API reference

CosmiQ Works GeoData API reference
=====================================

.. contents::

cw-geodata class and function list
----------------------------------

.. autosummary::

  cw_geodata.raster_image.image.get_geo_transform
  cw_geodata.vector_label.polygon.affine_transform_gdf
  cw_geodata.vector_label.polygon.convert_poly_coords
  cw_geodata.vector_label.polygon.geojson_to_px_gdf
  cw_geodata.vector_label.polygon.georegister_px_df
  cw_geodata.vector_label.polygon.get_overlapping_subset
  cw_geodata.vector_label.graph.geojson_to_graph
  cw_geodata.vector_label.graph.get_nodes_paths
  cw_geodata.vector_label.graph.process_linestring
  cw_geodata.vector_label.mask.boundary_mask
  cw_geodata.vector_label.mask.contact_mask
  cw_geodata.vector_label.mask.df_to_px_mask
  cw_geodata.vector_label.mask.footprint_mask
  cw_geodata.utils.geo.geometries_internal_intersection
  cw_geodata.utils.geo.list_to_affine
  cw_geodata.utils.geo.split_multi_geometries



Raster/Image functionality
--------------------------

Image submodule
~~~~~~~~~~~~~~~

.. automodule:: cw_geodata.raster_image.image
   :members:

Vector/Label functionality
--------------------------

Polygon submodule
~~~~~~~~~~~~~~~~~

.. automodule:: cw_geodata.vector_label.polygon
   :members:

Graph submodule
~~~~~~~~~~~~~~~

.. automodule:: cw_geodata.vector_label.graph
   :members:

Mask submodule
~~~~~~~~~~~~~~

.. automodule:: cw_geodata.vector_label.mask
   :members:

Utility functions
-----------------

Core utility submodule
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cw_geodata.utils.core
   :members:

Geo utility submodule
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cw_geodata.utils.geo
   :members:
