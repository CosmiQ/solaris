.. _pretrained_models:

##########################################
Pretrained models available in ``solaris``
##########################################

``solaris`` provides access to a number of pre-trained models from
`the SpaceNet challenges <https://spacenet.ai>`_.  See the table below for a
summary. Note that the model name in the first column should be used as the
``"model_name"`` argument in
`the config file <tutorials/notebooks/creating_the_yaml_config_file.html>`_ if you wish to use that model with ``solaris``. Note that we re-trained the
 competitors' models for compatibility with ``solaris`` and the training parameters,
 inputs, and performance may vary slightly from their original models.

Model details
=============

+------------------------------------+---------------------+-----------------------+----------------+-------------+-------------+---------------------------------+---------------------------------------+
| Model name                         | Model type          | Model details         | # Parameters   | Input shape |Output shape | Config file                     | Weights file                          |
+====================================+=====================+=======================+================+=============+=============+=================================+=======================================+
| xdxd_spacenet4                     | Segmentation UNet   | Encoder: VGG16        | 29.3M          | 3x512x512   | 1x512x512   | `link <XDXDconfig_>`_           | `link <XDXDweights_>`_  (117 MB)      |
+------------------------------------+---------------------+-----------------------+----------------+-------------+-------------+---------------------------------+---------------------------------------+
| selimsef_spacenet4_resnet34unet    | Segmentation UNet   | Encoder: ResNet-34    | 30.0M          | 4x416x416   | 3x416x416   | `link <ssresnet34config_>`_     | `link <ssresnet34weights_>`_ (120 MB) |
+------------------------------------+---------------------+-----------------------+----------------+-------------+-------------+---------------------------------+---------------------------------------+
| selimsef_spacenet4_densenet121unet | Segmentation UNet   | Encoder: DenseNet-121 | 15.6M          | 3x384x384   | 3x384x384   | `link <ssdense121config_>`_     | `link <ssdense121weights_>`_ (63 MB)  |
+------------------------------------+---------------------+-----------------------+----------------+-------------+-------------+---------------------------------+---------------------------------------+
| selimsef_spacenet4_densenet161unet | Segmentation UNet   | Encoder: DenseNet-161 | 41.1M          | 3x384x384   | 3x384x384   | `link <ssdense161config_>`_     | `link <ssdense161weights_>`_ (158 MB) |
+------------------------------------+---------------------+-----------------------+----------------+-------------+-------------+---------------------------------+---------------------------------------+

Training details
================

Below is a summary of the training hyperparameters for each model. For image
pre-processing and augmentation pipelines see the config files linked above.
*Note that our hyperparameters may differ from the competitors' original values.*
See `their solution descriptions <https://github.com/spacenetchallenge>`_ for
more on their implementations.

+------------------------------------+-------------------------+-------------------+---------------+------------------------+-----------------+------------+-----------------+---------------------+
| Model name                         | Loss function           | Optimizer         | Learning Rate | Training input         | Training mask   | Batch size | Training Epochs | Pre-trained weights |
+====================================+=========================+===================+===============+========================+=================+============+=================+=====================+
| xdxd_spacenet4                     | BCE +                   | Adam              | 1e-4          | SpaceNet 4             | Footprints only | 12         | 60              | None                |
|                                    | Jaccard (4:1)           | default params    | with decay    | Pan-sharpened RGB      |                 |            |                 |                     |
+------------------------------------+-------------------------+-------------------+---------------+------------------------+-----------------+------------+-----------------+---------------------+
| selimsef_spacenet4_resnet34unet    | Focal + Dice            | AdamW             | 2e-4          | SpaceNet 4             | 3-channel (FP,  | 42         | 70              | ImageNet (encoder   |
|                                    | (1:1)                   | 1e-3 weight decay | with decay    | Pan-sharpened RGB+NIR  | (edge, contact) |            |                 | only)               |
+------------------------------------+-------------------------+-------------------+---------------+------------------------+-----------------+------------+-----------------+---------------------+
| selimsef_spacenet4_densenet121unet | Focal + Dice            | AdamW             | 2e-4          | SpaceNet 4             | 3-channel (FP,  | 32         | 70              | ImageNet (encoder   |
|                                    | (1:1)                   | 1e-3 weight decay | with decay    | Pan-sharpened RGB      | (edge, contact) |            |                 | only)               |
+------------------------------------+-------------------------+-------------------+---------------+------------------------+-----------------+------------+-----------------+---------------------+
| selimsef_spacenet4_densenet161unet | Focal + Dice            | AdamW             | 2e-4          | SpaceNet 4             | 3-channel (FP,  | 20         | 60              | ImageNet (encoder   |
|                                    | (1:1)                   | 1e-3 weight decay | with decay    | Pan-sharpened RGB      | (edge, contact) |            |                 | only)               |
+------------------------------------+-------------------------+-------------------+---------------+------------------------+-----------------+------------+-----------------+---------------------+

.. _XDXDconfig: https://github.com/CosmiQ/solaris/blob/master/solaris/nets/configs/xdxd_spacenet4.yml
.. _ssresnet34config: https://github.com/CosmiQ/solaris/blob/master/solaris/nets/configs/selimsef_resnet34unet_spacenet4.yml
.. _ssdense121config: https://github.com/CosmiQ/solaris/blob/master/solaris/nets/configs/selimsef_densenet121unet_spacenet4.yml
.. _ssdense161config: https://github.com/CosmiQ/solaris/blob/master/solaris/nets/configs/selimsef_densenet161unet_spacenet4.yml
.. _XDXDweights: https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/xdxd_spacenet4_solaris_weights.pth
.. _ssresnet34weights: https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_resnet34unet_solaris_weights.pth
.. _ssdense121weights: https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_densenet121unet_solaris_weights.pth
.. _ssdense161weights: https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/selimsef_spacenet4_densenet161unet_solaris_weights.pth
