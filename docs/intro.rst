.. _intro:


##############################
An introduction to ``solaris``
##############################

--------------

What is ``solaris``?
====================

``solaris`` is a Python library with two main purposes:

#. Run existing geospatial computer vision models on any overhead imagery with
   a single line of code

#. Accelerate research in the geospatial computer vision domain by providing
   efficient implementations of common utility functions:

   * Imagery and vector-formatted label tiling
   * Interconversion between geospatial and machine learning data formats
   * Loss functions common in geospatial computer vision applications
   * Standardized evaluation of model performance on geospatial analysis tasks
   * *And more!*

--------------

Why should I use ``solaris``?
=============================
Most geospatial machine learning researchers discover early that they need to
write custom code to massage their data into a machine learning-compatible
format. This poses three major problems:

#. It is very challenging to evaluate models developed elsewhere or using different
   data, precluding deployment of geospatial ML solutions.

#. Researchers must have deep expertise in both GIS concepts and computer vision
   to advance the field, meaning less research gets done, slowing progress.

#. Every geospatial ML practitioner uses different data formats,
   imagery normalization methods, and machine learning frameworks during algorithm
   development. This makes comparison between models and application to new data
   time-consuming, if not impossible.

``solaris`` aims to overcome these obstacles by providing a single, centralized,
open source tool suite that can:

#. Accommodate any geospatial imagery and label formats,

#. prepare data for use in machine learning in a standardized fashion,

#. train computer vision models and generate predictions on geospatial imagery
   data using common deep learning frameworks, and

#. score model performance using domain-relevant metrics in a reproducible
   manner.

--------------

How do I use ``solaris``?
=========================
After `installing solaris <installation.html>`_, there are two usage
modes:

Command line: train or test models performance with a single command
--------------------------------------------------------------------
``solaris`` will provide a command line interface (CLI) tool to run an entire
geospatial imagery analysis pipeline from raw, un-chipped imagery, through model
training (if applicable) and prediction, to vector-formatted outputs. If you
provide ground truth labels over your prediction area, ``solaris`` can generate
quality metrics for the predictions. See
`an introduction to the solaris CLI <tutorials/cli.html>`_ for more.


Python API: Use ``solaris`` to accelerate model development
-----------------------------------------------------------
Alongside the simple CLI, all of ``solaris``'s functionality is accessible via
the Python API. The entirely open source codebase provides classes and functions
to:

* Tile imagery and labels
* Convert geospatial raster and vector data to formats compatible with machine
  learning frameworks
* Train deep learning models using PyTorch and Tensorflow (Keras) - more
  frameworks coming soon!
* Generate predictions on any geospatial imagery using your own models or
  existing pre-trained models from past `SpaceNet <https://www.spacenet.ai>`_
  challenges
* Convert model outputs to geospatial raster or vector formats
* Score model performance using standardized, geospatial-specific metrics

The ``solaris`` Python API documentation can be found `here <api/index>`_, and
`we have provided tutorials for common use cases <tutorials/index.html>`_.
The open source codebase is available `on GitHub <https://github.com/cosmiq/solaris>`_.

Follow us at our blog `The DownlinQ <https://medium.com/the-downlinq>`_ or
`on Twitter <https://twitter.com/cosmiqworks>`_ for updates!
