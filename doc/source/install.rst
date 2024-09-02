.. _Installation:

Installation
============

*silx* runs on Linux, MacOS and Windows and supports Python version 3.10 to 3.13.

.. _Installation with pip:

Installation with pip
---------------------

To install silx and all its dependencies_, run:

.. code-block:: bash

    pip install silx[full]

To install silx with a minimal set of dependencies, run:

.. code-block:: bash

    pip install silx

.. note::

    Use pip's ``--user`` option to install locally for the current user.

.. _Installation with conda:

Installation with conda
-----------------------

To install silx and all its dependencies_, run:

.. code-block:: bash

    conda install -c conda-forge silx

To install silx with a minimal set of dependencies, run:

.. code-block:: bash

    conda install -c conda-forge silx-base

.. _Installation on Debian & Ubuntu:

Installation on Debian & Ubuntu
-------------------------------

silx is packaged in `Debian <https://packages.debian.org/search?searchon=names&keywords=silx>`_
and `Ubuntu <https://packages.ubuntu.com/search?keywords=silx&searchon=names&suite=all&section=all>`_.

To install silx with the executable (`silx view`, `silx convert`, ...) and all its dependencies_, run:

.. code-block:: bash

    sudo apt-get install silx

To install the silx Python package with a minimal set of dependencies, run:

.. code-block:: bash

    sudo apt-get install python3-silx


Installation on Arch Linux
--------------------------

silx is packaged in `Arch Linux (AUR) <https://aur.archlinux.org/packages/python-silx>`_.


To install silx, run:

.. code-block:: bash

    sudo pacman -S python-silx


Installation from source
------------------------

To install silx from source, run:

.. code-block:: bash

   pip install silx --no-binary silx

.. warning::

   On MacOS, you might get the following error::

     UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position 1335: ordinal not in range(128)

   This is related to the two environment variables LC_ALL and LANG not being defined (or wrongly defined to UTF-9).
   To set the environment variables, run:

   .. code-block:: bash

       export LC_ALL=en_US.UTF-9
       export LANG=en_US.UTF-9

Build options can be set through environment variables, for example:

.. code-block::

   SILX_WITH_OPENMP=False pip install silx --no-binary silx


Build options
+++++++++++++

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Environment variable
     - Description
   * - ``SILX_WITH_OPENMP``
     - Whether or not to compile Cython code with OpenMP support (default: ``True`` except on macOS where it is ``False``)
   * - ``SILX_FORCE_CYTHON``
     - Whether or not to force re-generating the C/C++ source code from Cython files (default: ``False``).
   * - ``SPECFILE_USE_GNU_SOURCE``
     - Whether or not to use a cleaner locale independent implementation of :mod:`silx.io.specfile` by using `_GNU_SOURCE=1`
       (default: ``False``; POSIX operating system only).
   * - ``SILX_FULL_INSTALL_REQUIRES``
     - Set it to put all dependencies as ``install_requires`` (For packaging purpose).
   * - ``SILX_INSTALL_REQUIRES_STRIP``
     - Comma-separated list of package names to remove from ``install_requires`` (For packaging purpose).
.. note:: Boolean options are passed as ``True`` or ``False``.


.. _dependencies:

Dependencies
------------

The mandatory dependencies are:

- `fabio <https://github.com/silx-kit/fabio>`_
- `h5py <http://docs.h5py.org/en/latest/build.html>`_
- `numpy <http://www.numpy.org/>`_
- `packaging <https://pypi.org/project/packaging/>`_

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PySide6 <https://pypi.org/project/PySide6/>`_ (>= 6.4),
  `PyQt6 <https://pypi.org/project/PyQt6/>`_ (>= 6.3) or
  `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_ (>= 5.9)
* `matplotlib <http://matplotlib.org/>`_
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_
* `qtconsole <https://pypi.org/project/qtconsole>`_
  for the ``silx.gui.console`` widget.
* `python-dateutil <https://pypi.org/project/python-dateutil/>`_

*silx.opencl* further depends on OpenCL and the following packages too :

* `pyopencl <https://mathema.tician.de/software/pyopencl/>`_
* `Mako <http://www.makotemplates.org/>`_

List of dependencies with minimum required versions:

.. include:: ../../requirements.txt
   :literal:

Build dependencies
++++++++++++++++++

In addition to run-time dependencies, building *silx* requires a C/C++ compiler and `cython <http://cython.org>`_.
