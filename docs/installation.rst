.. _installation:

######################
Installing ``solaris``
######################

There are several methods available for installing `solaris <https://github.com/cosmiq/solaris>`_:

* :ref:`github-install`
* :ref:`pip-only` *use at your own risk!*

----------

Prerequisites
=============

Regardless of installation method, you'll need Python version 3.6 or greater.
More details on installing Python can be found
`here <https://www.python.org/about/gettingstarted/>`_. Additionally, if you
plan to use the SpaceNet dataset with ``solaris`` (it features prominently in
many of the tutorials), you'll need `a free Amazon Web Services account <https://aws.amazon.com/>`_
and the AWS CLI `installed <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html>`_
and `configured <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html>`_.
If you're just going to work with your own data, you can skip these steps.

--------------

.. _github-install:

Installing from GitHub using a ``conda`` environment and ``pip``
================================================================
If you wish to install a bleeding-edge version of ``solaris`` that isn't available
on conda-forge yet, you can install from GitHub. You'll need
`anaconda`_ for this installation as well.

From a terminal, run::

  git clone https://github.com/cosmiq/solaris.git
  cd solaris
  git checkout [branch_name]  # for example, git checkout dev for bleeding-edge

If you have access to a GPU where you're installing ``solaris``, use the following::

  conda env create -f environment-gpu.yml

If you don't have access to a GPU::

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
