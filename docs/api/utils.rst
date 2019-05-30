.. title:: solaris.utils API reference

``solaris.utils`` API reference
===============================

.. contents::

``solaris.utils`` class and function list
-----------------------------------------

.. autosummary::

   solaris.utils.core.get_data_paths
   solaris.utils.core.get_files_recursively
   solaris.utils.config.parse
   solaris.utils.io.imread
   solaris.utils.io.preprocess_im_arr
   solaris.utils.io.scale_for_model
   solaris.utils.io.rescale_arr
   solaris.utils.geo.list_to_affine
   solaris.utils.geo.geometries_internal_intersection
   solaris.utils.geo.split_multi_geometries
   solaris.utils.geo.get_subgraph
   solaris.utils.tile.utm_getZone
   solaris.utils.tile.utm_isNorthern
   solaris.utils.tile.calculate_UTM_crs
   solaris.utils.tile.get_utm_vrt
   solaris.utils.tile.get_utm_vrt_profile
   solaris.utils.tile.tile_read_utm
   solaris.utils.tile.tile_exists_utm
   solaris.utils.tile.get_wgs84_bounds
   solaris.utils.tile.get_utm_bounds
   solaris.utils.tile.read_vector_file
   solaris.utils.tile.transformToUTM
   solaris.utils.tile.search_gdf_bounds
   solaris.utils.tile.search_gdf_polygon
   solaris.utils.tile.vector_tile_utm
   solaris.utils.tile.getCenterOfGeoFile
   solaris.utils.tile.clip_gdf
   solaris.utils.tile.rasterize_gdf
   solaris.utils.tile.vector_gdf_get_projection_unit
   solaris.utils.tile.raster_get_projection_unit
   solaris.utils.raster.reorder_axes



``solaris.utils.core`` Core utilities
-------------------------------------

.. automodule:: solaris.utils.core
   :members:

``solaris.utils.config`` Configuration file utilities
-----------------------------------------------------

.. automodule:: solaris.utils.config
   :members:

``solaris.utils.io`` Imagery and vector I/O utilities
-----------------------------------------------------

.. automodule:: solaris.utils.io
   :members:

``solaris.utils.geo`` Geographic coordinate system management utilities
-----------------------------------------------------------------------

.. automodule:: solaris.utils.geo
   :members:

``solaris.utils.tile`` Tiling utilities
---------------------------------------

.. automodule:: solaris.utils.tile
   :members:

``solaris.utils.raster`` Raster image and array management utilities
--------------------------------------------------------------------

.. automodule:: solaris.utils.raster
   :members:
