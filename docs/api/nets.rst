.. title:: solaris.nets API reference

``solaris.nets`` API reference
==============================

.. contents::

``solaris.nets`` class and function list
----------------------------------------

.. autosummary::

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
   solaris.nets._keras_losses.k_dice_loss
   solaris.nets._keras_losses.k_jaccard_loss
   solaris.nets._keras_losses.k_focal_loss
   solaris.nets._keras_losses.k_lovasz_hinge
   solaris.nets._keras_losses.tf_lovasz_grad
   solaris.nets._keras_losses.k_weighted_bce
   solaris.nets._keras_losses.k_layered_weighted_bce
   solaris.nets._torch_losses.TorchFocalLoss
   solaris.nets._torch_losses.torch_lovasz_hinge
   solaris.nets._torch_losses.lovasz_hinge_flat
   solaris.nets._torch_losses.flatten_binary_scores
   solaris.nets._torch_losses.TorchStableBCELoss
   solaris.nets._torch_losses.binary_xloss
   solaris.nets._torch_losses.lovasz_grad
   solaris.nets._torch_losses.iou_binary
   solaris.nets._torch_losses.iou
   solaris.nets._torch_losses.isnan
   solaris.nets._torch_losses.mean
   solaris.nets.transform.Rotate
   solaris.nets.transform.RandomScale
   solaris.nets.transform.scale
   solaris.nets.transform.build_pipeline
   solaris.nets.transform.process_aug_dict
   solaris.nets.transform.get_augs
   solaris.nets.optimizers.get_optimizer
   solaris.nets.model_io.get_model
   solaris.nets.model_io.reset_weights
   solaris.nets.datagen.make_data_generator
   solaris.nets.datagen.KerasSegmentationSequence
   solaris.nets.datagen.TorchDataset
   solaris.nets.datagen.InferenceTiler
   solaris.nets.metrics.get_metrics
   solaris.nets.metrics.dice_coef_binary
   solaris.nets.metrics.precision
   solaris.nets.metrics.recall
   solaris.nets.metrics.f1_score
   solaris.nets.zoo.XDXD_SpaceNet4_UNetVGG16
   solaris.nets.train.Trainer
   solaris.nets.train.get_train_val_dfs
   solaris.nets.infer.Inferer
   solaris.nets.infer.get_infer_df



``solaris.nets.callbacks`` Keras-like callbacks
-----------------------------------------------

.. automodule:: solaris.nets.callbacks
   :members:

.. automodule:: solaris.nets.torch_callbacks
   :members:

``solaris.nets.losses`` Loss functions for Geo CV model training
----------------------------------------------------------------

.. automodule:: solaris.nets.losses
   :members:

.. automodule:: solaris.nets._keras_losses
   :members:

.. automodule:: solaris.nets._torch_losses
   :members:

``solaris.nets.transform`` Augmentation pipeline prep for Geo imagery
---------------------------------------------------------------------

.. automodule:: solaris.nets.transform
   :members:

``solaris.nets.optimizers`` Model training optimizer management
---------------------------------------------------------------

.. automodule:: solaris.nets.optimizers
   :members:

``solaris.nets.model_io`` Model I/O and model weight management
---------------------------------------------------------------

.. automodule:: solaris.nets.model_io
   :members:

``solaris.nets.datagen`` Data generators for model training
-----------------------------------------------------------

.. automodule:: solaris.nets.datagen
   :members:

``solaris.nets.metrics`` Metrics for evaluating model performance
-----------------------------------------------------------------

.. automodule:: solaris.nets.metrics
   :members:

``solaris.nets.zoo`` Model definitions for geospatial image analysis
--------------------------------------------------------------------

.. automodule:: solaris.nets.zoo
   :members:

``solaris.nets.train`` Model training functionality
---------------------------------------------------

.. automodule:: solaris.nets.train
   :members:

``solaris.nets.infer`` Prediction with Geo CV models
----------------------------------------------------

.. automodule:: solaris.nets.infer
   :members:
