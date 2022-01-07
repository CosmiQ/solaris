.. _tutorials_index:

##############################
solaris Tutorials and Cookbook
##############################

.. toctree::
  :maxdepth: 3
  :glob:
  :hidden:

  cli*
  notebooks/*


There are two different ways to use ``solaris``:

* :ref:`The Command Line Interface` (Simple use with existing models)
* :ref:`The Python API` (Python users who wish to develop their own models)

Here we provide a brief introduction to these two approaches, links to tutorials,
and usage recipes to complete common tasks. If there's a common use case not
covered here, `submit an issue on the GitHub repo to request its inclusion. <https://github.com/CosmiQ/solaris/issues>`_

---------------

The command line interface
==========================
The command line interface (CLI) is the simplest way to use solaris. Using the CLI,
you can run format conversions or metrics without writing a single line of python code.

After :doc:`installing solaris <../installation>`, you can run simple commands from a
terminal or command prompt for standard operations to create training masks
using vector labels and run evaluation metrics.

If these relatively narrow use cases don't cover your needs, the ``solaris`` python
API can help!

The Python API
==============
The ``solaris`` Python API provides functionality needed to perform common tasks related to deep learning
analysis of overhead imagery data:

* Customizable imagery and vector label tiling, with different size and coordinate system options.
* Training mask creation functions, with the option to create custom width edge masks, building footprint masks, road network masks, multi-class masks, and even masks which label narrow spaces between objects.
* TODO

The :doc:`Python API Reference <../api/index>` provides full documentation of
everything described above and more. For usage examples to get you started, see
the tutorials below.

* `Tiling imagery <notebooks/api_tiling_tutorial.ipynb>`_
* `Creating training masks <notebooks/api_masks_tutorial.ipynb>`_
* `Converting pixel masks to vector labels <notebooks/api_mask_to_vector.ipynb>`_
* `Scoring your model's performance with the solaris Python API <notebooks/api_evaluation_tutorial.ipynb>`_
* `Creating COCO-formatted datasets <notebooks/api_coco_tutorial.ipynb>`_

Reference
=========
* :doc:`API reference <../api/index>`

Index
=====
* :ref:`genindex`
* :ref:`modindex`
