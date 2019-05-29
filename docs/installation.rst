######################
Installing ``solaris``
######################

There are several methods available for installing `solaris <https://github.com/cosmiq/solaris>`_:

* :ref:`conda-forge` **(recommended)**
* :ref:`github-install`
* :ref:`pip-only` *use at your own risk!*

----------

.. _conda-forge:

Installing from ``conda-forge``
===============================
**This is the recommended installation method.**

If you have `anaconda`_ installed,
you can create a new ``conda`` environment and install ``solaris`` there with ease::

  conda create solaris python=3.6 solaris -c conda-forge

We recommend installing ``solaris`` in a new environment to avoid conflicts with
existing installations of packages (``GDAL`` incompatibility being the usual problem),
but installing ``solaris`` in an existing environment can work in some cases.

----------

.. _github-install:

Installing from GitHub using a ``conda`` environment and ``pip``
================================================================
If you wish to install a bleeding-edge version of ``solaris`` that isn't available
on conda-forge yet, you can install from GitHub. You'll need
`anaconda`_ for this installation as well.

From a terminal, run::

  git clone https://github.com/cosmiq/solaris.git
  cd solaris
  git checkout [branch_name]  # for example, git checkout dev
  conda env create -f environment.yml
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
