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


.. hint:: To install in `editable mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_, see :ref:`install-silx-editable`.


Build options
+++++++++++++

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Build option
     - Description
   * - ``use_openmp``
     - Whether or not to compile Cython code with OpenMP support.
       Accepted values: ``auto`` (default), ``enabled``, ``disabled``.
   * - ``specfile_use_gnu_source``
     - Whether or not to use a cleaner locale independent implementation of :mod:`silx.io.specfile` by using `_GNU_SOURCE=1`.
       Only used on POSIX operating systems.
       Accepted values: ``false`` (default), ``true``.


Build options can be passed to
`meson's setup-args <https://mesonbuild.com/meson-python/reference/config-settings.html#cmdoption-arg-setup-args>`_
through `pip install -C <https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-C>`_,
for example:

.. code-block:: bash

   pip install silx --no-binary silx -Csetup-args="-Duse_openmp=disabled"


.. _dependencies:

Dependencies
------------

*silx* provides a minimal installation and a number of optional dependency groups enabling additional features.

Base installation
+++++++++++++++++

The default installation:

.. code-block:: bash

    pip install silx

provides:

* Data I/O support, including HDF5 and many scientific image formats
* Numerical algorithms and data processing
* Utility modules and command-line tools
* Non-GUI applications and scripts

It installs:

* `numpy <http://www.numpy.org/>`_ for numerical computing
* `h5py <http://docs.h5py.org/en/latest/build.html>`_ for HDF5 file access
* `fabio <https://github.com/silx-kit/fabio>`_ for reading scientific image formats
* `packaging <https://pypi.org/project/packaging/>`_ for package and version handling

OpenCL acceleration
+++++++++++++++++++

The `opencl` extra:

.. code-block:: bash

    pip install silx[opencl]

adds support for GPU/OpenCL accelerated algorithms provided by :mod:`silx.opencl`.

It installs:

* `pyopencl <https://mathema.tician.de/software/pyopencl/>`_
  for interfacing with OpenCL devices
* `Mako <http://www.makotemplates.org/>`_
  for generating OpenCL kernels

Required system dependencies:

* An OpenCL runtime (CPU or GPU implementation) provided by the system
  or hardware vendor drivers (e.g. Intel, NVIDIA, AMD)

Scientific and visualization features
+++++++++++++++++++++++++++++++++++++

The `full_no_qt` extra:

.. code-block:: bash

    pip install silx[full_no_qt]

adds support for most optional *silx* features except the Qt binding itself.

It installs:

* `silx[opencl]` for OpenCL-accelerated algorithms
* `matplotlib <http://matplotlib.org/>`_
  for Matplotlib Qt backend support
* `python-dateutil <https://pypi.org/project/python-dateutil/>`_
  for date and time handling in plots
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_
  for 3D visualization widgets
* `qtconsole <https://pypi.org/project/qtconsole>`_
  for embedding an interactive IPython console in Qt applications
* `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_
  for reading and writing HDF5 datasets using additional compression filters
* `pint <https://pint.readthedocs.io/>`_
  for storing and restoring physical quantities with units
* `scipy <https://scipy.org/>`_
  for advanced numerical algorithms used by selected modules
* `pooch <https://www.fatiando.org/pooch/>`_
  for downloading example datasets used by some demonstrations
* `Pillow <https://python-pillow.org/>`_
  for image handling used by selected image-processing features
* `qtawesome <https://github.com/spyder-ide/qtawesome>`_
  for providing icon sets used by graphical applications

Required system dependencies:

* A working OpenGL implementation provided by the system graphics drivers
  (NVIDIA, AMD, Intel) or by Mesa on Linux (hardware or software rendering)

GUI support
+++++++++++

The `full` extra:

.. code-block:: bash

    pip install silx[full]

adds a Qt binding on top of `full_no_qt` and enables all graphical user interface components.

It installs:

* `PySide6 <https://pypi.org/project/PySide6/>`_,
  the default Qt binding used by *silx*

Alternative supported Qt bindings are:

* `PyQt6 <https://pypi.org/project/PyQt6/>`_
* `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_

HSDS support
++++++++++++

The `h5pyd` extra:

.. code-block:: bash

    pip install silx[h5pyd]

adds support for accessing HDF5 datasets hosted through HSDS (Highly Scalable Data Service).

It installs:

* `h5pyd <https://github.com/HDFGroup/h5pyd>`_, the Python client for HSDS


Build dependencies
++++++++++++++++++

*silx* uses `meson-python <https://mesonbuild.com/meson-python/>`_ build backend and
requires `cython <http://cython.org>`_ and a C/C++ compiler.
