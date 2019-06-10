.. title:: API reference contents

###################
Solaris API summary
###################

Complete submodule documentation
================================
* `solaris.tile <api/tile.html>`_: Tiling functionality for imagery and vector labels
* `solaris.raster <api/raster.html>`_: Raster (imagery) coordinate management and formatting
* `solaris.vector <api/vector.html>`_: Vector (label) management and format interconversion
* `solaris.nets <api/nets.html>`_: Deep learning model ingestion, creation, training, and inference
* `solaris.eval <api/eval.html>`_: Deep learning model performance evaluation
* `solaris.utils <api/utils.html>`_: Utility functions for the above toolsets


Submodule summaries
===================

`solaris.tile <api/tile.html>`_: Tiling functionality for imagery and vector labels
-------------------------------------------------------------------------------------------------

.. autosummary::

  solaris.tile.main.tile_utm_source
  solaris.tile.main.tile_utm
  solaris.tile.main.get_chip
  solaris.tile.main.calculate_anchor_points
  solaris.tile.main.calculate_cells
  solaris.tile.main.calculate_analysis_grid
  solaris.tile.vector_utils.read_vector_file
  solaris.tile.vector_utils.transformToUTM
  solaris.tile.vector_utils.search_gdf_bounds
  solaris.tile.vector_utils.search_gdf_polygon
  solaris.tile.vector_utils.vector_tile_utm
  solaris.tile.vector_utils.clip_gdf


`solaris.raster <api/raster.html>`_: Raster (imagery) coordinate management and formatting
--------------------------------------------------------------------------------------------------------

.. autosummary::

  solaris.raster.image.get_geo_transform
  solaris.raster.image.stitch_images

`solaris.vector <api/vector.html>`_: Vector (label) management
----------------------------------------------------------------------------

.. autosummary::

  solaris.vector.graph.geojson_to_graph
  solaris.vector.graph.get_nodes_paths
  solaris.vector.graph.parallel_linestring_to_path
  solaris.vector.graph.linestring_to_edges
  solaris.vector.graph.graph_to_geojson
  solaris.vector.graph.Node
  solaris.vector.graph.Edge
  solaris.vector.graph.Path
  solaris.vector.mask.df_to_px_mask
  solaris.vector.mask.footprint_mask
  solaris.vector.mask.boundary_mask
  solaris.vector.mask.contact_mask
  solaris.vector.mask.mask_to_poly_geojson
  solaris.vector.mask.road_mask
  solaris.vector.polygon.convert_poly_coords
  solaris.vector.polygon.affine_transform_gdf
  solaris.vector.polygon.georegister_px_df
  solaris.vector.polygon.geojson_to_px_gdf
  solaris.vector.polygon.get_overlapping_subset
  solaris.vector.polygon.gdf_to_yolo

`solaris.nets <api/nets.html>`_: Deep learning model creation, training, and inference
----------------------------------------------------------------------------------------------------

.. autosummary::

  solaris.nets.model_io.get_model
  solaris.nets.model_io.reset_weights
  solaris.nets.train.Trainer
  solaris.nets.train.get_train_val_dfs
  solaris.nets.infer.Inferer
  solaris.nets.infer.get_infer_df
  solaris.nets.datagen.make_data_generator
  solaris.nets.datagen.KerasSegmentationSequence
  solaris.nets.datagen.TorchDataset
  solaris.nets.datagen.InferenceTiler
  solaris.nets.callbacks.get_callbacks
  solaris.nets.callbacks.KerasTerminateOnMetricNaN
  solaris.nets.callbacks.get_lr_schedule
  solaris.nets.callbacks.keras_lr_schedule
  solaris.nets.torch_callbacks.TorchEarlyStopping
  solaris.nets.torch_callbacks.TorchTerminateOnNaN
  solaris.nets.torch_callbacks.TorchTerminateOnMetricNaN
  solaris.nets.torch_callbacks.TorchModelCheckpoint
  solaris.nets.losses.get_loss
  solaris.nets.losses.get_single_loss
  solaris.nets.losses.keras_composite_loss
  solaris.nets.losses.TorchCompositeLoss
  solaris.nets.metrics.get_metrics
  solaris.nets.metrics.dice_coef_binary
  solaris.nets.metrics.precision
  solaris.nets.metrics.recall
  solaris.nets.metrics.f1_score
  solaris.nets.optimizers.get_optimizer
  solaris.nets.transform.build_pipeline
  solaris.nets.transform.process_aug_dict
  solaris.nets.transform.get_augs
  solaris.nets.zoo.XDXD_SpaceNet4_UNetVGG16

`solaris.eval <api/eval.html>`_: Deep learning model performance evaluation
-----------------------------------------------------------------------------------------

.. autosummary::

  solaris.eval.base.Evaluator
  solaris.eval.iou.calculate_iou
  solaris.eval.iou.process_iou
  solaris.eval.challenges.off_nadir_dataset.get_collect_id
  solaris.eval.challenges.off_nadir_dataset.get_aoi
  solaris.eval.challenges.spacenet_buildings2_dataset.spacenet_buildings_2

`solaris.utils <api/utils.html>`_: Utility functions for the above toolsets
-----------------------------------------------------------------------------------------

.. autosummary::

  solaris.utils.config.parse
  solaris.utils.core.get_files_recursively
  solaris.utils.geo.list_to_affine
  solaris.utils.geo.geometries_internal_intersection
  solaris.utils.geo.split_multi_geometries
  solaris.utils.geo.get_subgraph
  solaris.utils.io.imread
  solaris.utils.io.preprocess_im_arr
  solaris.utils.io.scale_for_model
  solaris.utils.io.rescale_arr
  solaris.utils.raster.reorder_axes
  solaris.utils.tile.utm_getZone
  solaris.utils.tile.utm_isNorthern
  solaris.utils.tile.calculate_UTM_crs
  solaris.utils.tile.get_utm_vrt
  solaris.utils.tile.get_utm_vrt_profile
  solaris.utils.tile.tile_read_utm
  solaris.utils.tile.get_wgs84_bounds
  solaris.utils.tile.get_utm_bounds
  solaris.utils.tile.read_vector_file
  solaris.utils.tile.transformToUTM
  solaris.utils.tile.search_gdf_bounds
  solaris.utils.tile.search_gdf_polygon
  solaris.utils.tile.vector_tile_utm
  solaris.utils.tile.clip_gdf
  solaris.utils.tile.rasterize_gdf
  solaris.utils.tile.vector_gdf_get_projection_unit
  solaris.utils.tile.raster_get_projection_unit

CLI commands
============
Documentation coming soon!


Solaris API reference: Index
============================

Solaris submodules
------------------

.. toctree::
   :maxdepth: 2

   api/tile
   api/raster
   api/vector
   api/nets
   api/eval
   api/utils
