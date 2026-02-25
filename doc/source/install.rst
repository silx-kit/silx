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

The mandatory dependencies are:

- `fabio <https://github.com/silx-kit/fabio>`_
- `h5py <http://docs.h5py.org/en/latest/build.html>`_
- `numpy <http://www.numpy.org/>`_
- `packaging <https://pypi.org/project/packaging/>`_

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PySide6 <https://pypi.org/project/PySide6/>`_ (>= 6.4),
  `PyQt6 <https://pypi.org/project/PyQt6/>`_ (>= 6.3) or
  `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_ (>= 5.14)
* `matplotlib <http://matplotlib.org/>`_ (>= 3.6)
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_
* `qtconsole <https://pypi.org/project/qtconsole>`_
  for the ``silx.gui.console`` widget.
* `python-dateutil <https://pypi.org/project/python-dateutil/>`_

*silx.opencl* further depends on OpenCL and the following packages too :

* `pyopencl <https://mathema.tician.de/software/pyopencl/>`_
* `Mako <http://www.makotemplates.org/>`_

*h5pyd* support to access HSDS urls depends on:

* `h5pyd <https://github.com/HDFGroup/h5pyd>`_ (>= 0.20.0)


Build dependencies
++++++++++++++++++

*silx* uses `meson-python <https://mesonbuild.com/meson-python/>`_ build backend and
requires `cython <http://cython.org>`_ and a C/C++ compiler.
