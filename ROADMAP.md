## Solaris in 2022
Deep learning approaches for classification, object detection, semantic segmentation, and instance segmentation are now commonplace in the geospatial world. However, tooling and standards to work with geospatial data and deep learning methods are still limited and under developed. We hope that solaris can help advance tooling and standards for doing GeoAI by making it easy to set up model inputs for experiments and easy to combine, clean, and evaluate detection results that are sourced from large satellite images.

## Statement of Purpose
Version 0.5 of solaris is a new chapter in the library's history that has a more narrow focus on setting up model inputs before an experiment, and combining, cleaning, and evaluating detection results after an experiment. This new iteration of the library does not provide pretrained models, training logic, data loaders, or any functionality with a dependency on Pytorch or Tensorflow 2. Instead, we leave this to the wider ecosystem of other ML frameworks, and expect that solaris will be a component of one's toolkit that makes using these ML frameworks easier.

Since solaris was started by Cosmiq Works in 2019, other frameworks such as detectron2, mmdetection, the [Tensorflow 2 Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), [torchvision](https://pytorch.org/vision/stable/index.html), and [fastai](https://github.com/fastai/fastai) have made it accessible to train object detection and segmentation models. Of particular note is the [torchgeo](https://github.com/microsoft/torchgeo) library, which contains models geared toward the satellite and aerial imagery domains. These frameworks handle the management of model zoos, training logic, model configuration, and data loading better than solaris can, and so we intend to focus on pieces of the ML pipeline that are not addressed by these frameworks.

## Roadmap
This document lists general directions that core contributors are interested to see developed in solaris. The fact that an item is listed here is in no way a promise that it will happen, as resources are limited. Rather, it is an indication that help is welcomed on this topic. See the github issues and active PRs to see if a topic is being actively worked on.

- refactor solaris into installable sub-modules for lighterweight dependencies
- remove gdal dependency and rely on rasterio+rasterio’s gdal binaries to simplify the code base and make installation easier
- move ci/cd to github actions
- performant conversion between geospatial and ml formats [(including geojson to coco)](https://solaris.readthedocs.io/en/latest/api/data.html#module-solaris.data.coco)
- support for converting vector labels to segmentation formats (coco) that have multi polygons (holes, gaps in a single instance)
- functionality for reassigning geospatial metadata to ML predictions
- functionality for performantly merging overlapping vector predictions (that were made on overlapping tiles)
- metrics for classification, object detection, and segmentation
- visualizing confusion matrices and metrics starting from different detection outputs. For example, comparing class-wise results for semantic segmentation and instance segmentation outputs
- tiling Cloud Optimized Geotiffs (COGs) and vector labels in parellel
- examples integrating with ML frameworks that handle model architecture and model training logic

## What solaris will not cover (because other frameworks/libraries handle this scope)

- [novel model implementations for GeoAI](https://github.com/microsoft/torchgeo#documentation)
- data loaders for geospatial ML inputs: many apis have their own way of performantly handling common and custom formats, [like fastai for example](https://docs.fast.ai/data.block.html#DataBlock.dataloaders)
- any model training logic (we don't want to deal with TF or Pytorch dependencies that are always changing and cumbersome to install. this is handled by other ML frameworks)
- anything related to serving predictions
- anything related to specific challenges or published datasets. use solaris version 0.4.0 for functionality related to the Spacenet challenges.
