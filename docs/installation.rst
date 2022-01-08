.. _installation:

######################
Installing ``solaris``
######################

There are several methods available for installing `solaris <https://github.com/CosmiQ/solaris>`_:

* :ref:`github-install`
* :ref:`pip-only`

----------

Prerequisites
=============

Regardless of installation method, you'll need Python version 3.7 or greater.
More details on installing Python can be found
`here <https://www.python.org/about/gettingstarted/>`_.
--------------

.. _github-install:

Installing from GitHub using a ``conda`` environment and ``pip``
================================================================
If you wish to install a bleeding-edge version of ``solaris`` that isn't available
on conda-forge yet, you can install from GitHub. You'll need
`anaconda`_ for this installation as well.

From a terminal, run::

  git clone https://github.com/CosmiQ/solaris.git
  cd solaris
  git checkout [branch_name]  # for example, git checkout dev for bleeding-edge

Then:

  conda env create -f environment.yml


Finally, run the last two lines (for installs both with or without GPU)::

  conda activate solaris
  pip install .

The above installation will create a new conda environment called ``solaris``
containing your desired version of solaris and all of its dependencies.

----------

.. _pip-only:

Installing with only ``pip``
============================
*Use this method at your own risk!*

If you have already installed the dependencies with underlying binaries that
don't install well with ``pip`` (i.e. ``GDAL`` and ``rtree``), you can easily
``pip install`` the rest::

  pip install solaris

Note that this will raise an error if you don't already have ``GDAL`` installed.


.. _anaconda: https://docs.anaconda.com/anaconda/install/
