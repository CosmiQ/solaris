.. _tutorials_index:

##############################
Solaris Tutorials and Cookbook
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
covered here, `submit an issue on the GitHub repo to request its inclusion. <https://github.com/cosmiq/solaris/issues>`_

---------------

The command line interface
==========================
The command line interface (CLI) is the simplest way to use Solaris. Using the CLI,
you can run training and/or prediction on overhead imagery using `SpaceNet <https://www.spacenet.ai>`_ models
without writing a single line of python code.

After :doc:`installing Solaris <../installation>`, you can run simple commands from a
terminal or command prompt for standard operations, from creating training masks
using vector labels to running an entire deep learning pipeline through
evaluating model performance. Instead of having to write code to help ``solaris``
find your data, you just make basic edits to a configuration file template, then
``solaris`` does all the work to make your analysis pipeline fit together. Tutorials
on creating configuration files and running the CLI can be found below.

* `Creating the .yml config file <notebooks/creating_the_yaml_config_file.ipynb>`_
* `Creating reference files to help solaris find your imagery <notebooks/creating_im_reference_csvs.ipynb>`_
* `Creating training masks with the solaris CLI <notebooks/cli_mask_creation.ipynb>`_
* `Running a full deep learning pipeline using the solaris CLI <notebooks/cli_ml_pipeline.ipynb>`_
* `Evaluating prediction quality on SpaceNet data with the solaris CLI <notebooks/cli_spacenet_evaluation.ipynb>`_
* `Mapping vehicles with the cowc dataset <notebooks/map_vehicles_cowc.ipynb>`_

If these relatively narrow use cases don't cover your needs, the ``solaris`` python
API can help!

The Python API
==============
The ``solaris`` Python API provides every functionality needed to perform deep learning
analysis of overhead imagery data:

* Customizable imagery and vector label tiling, with different size and coordinate system options.
* Training mask creation functions, with the option to create custom width edge masks, building footprint masks, road network masks, multi-class masks, and even masks which label narrow spaces between objects.
* All required deep learning functionality, from augmentation (including >3 channel imagery tools!) to data ingestion to model training and inference to evaluation during training. These functions are currently implemented with both PyTorch and TensorFlow backends.
* The ability to use pre-trained or freshly initialized `SpaceNet <https://www.spacenet.ai>`_ models, as well as your own custom models
* Model performance evaluation tools for the SpaceNet IoU metric (APLS coming soon!)

The :doc:`Python API Reference <../api/index>` provides full documentation of
everything described above and more. For usage examples to get you started, see
the tutorials below.

* `Tiling imagery <notebooks/api_tiling_tutorial.ipynb>`_
* `Creating training masks <notebooks/api_masks_tutorial.ipynb>`_
* `Training a SpaceNet model <notebooks/api_training_spacenet.ipynb>`_
* `Inference with a pre-trained SpaceNet model <notebooks/api_inference_spacenet.ipynb>`_
* `Training a custom model <notebooks/api_training_custom.ipynb>`_
* `Converting pixel masks to vector labels <notebooks/api_mask_to_vector.ipynb>`_
* `Scoring your model's performance with the solaris Python API <notebooks/api_evaluation_tutorial.ipynb>`_
* `Creating COCO-formatted datasets <notebooks/api_coco_tutorial.ipynb>`_
* `Preprocessing Part 1: Pipelines <notebooks/preprocessing_pipelines.ipynb>`_
* `Preprocessing Part 2: Branching <notebooks/preprocessing_branching.ipynb>`_
* `Preprocessing Part 3: SAR <notebooks/preprocessing_sar.ipynb>`_

Reference
=========
* :doc:`API reference <../api/index>`

Index
=====
* :ref:`genindex`
* :ref:`modindex`


Check back here and `follow us on Twitter <https://twitter.com/CosmiqWorks>`_ or
on our blog, `The DownlinQ <https://medium.com/the-downlinq>`_ for updates!
