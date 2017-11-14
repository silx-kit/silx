
silx toolkit
============

The silx project aims at providing a collection of Python packages to support the
development of data assessment, reduction and analysis applications at synchrotron
radiation facilities.
It aims at providing reading/writing different file formats, data reduction routines
and a set of Qt widgets to browse and visualize data.

The current version provides:

* reading `HDF5 <https://www.hdfgroup.org/HDF5/>`_  file format (with support of
  `SPEC <https://certif.com/spec.html>`_ file format and
  `FabIO <http://www.silx.org/doc/fabio/dev/getting_started.html#list-of-file-formats-that-fabio-can-read-and-write>`_
  images)
* histogramming
* fitting
* 1D and 2D visualization widgets using multiple backends (matplotlib or OpenGL)
* an OpenGL-based widget to display 3D scalar field with isosurface and cutting plane
* an image plot widget with a set of associated tools
* a unified browser for HDF5, SPEC and image file formats supporting inspection and
  visualization of n-dimensional datasets.
* a unified viewer (*silx view filename*) for HDF5, SPEC and image file formats
* a unified converter to HDF5 format (*silx convert filename*)
* median filters on images (C and OpenCL implementations)
* image alignement (sift - OpenCL implementation)
* filtered backprojection and forward projection for tomography

Installation
------------

To install silx, run::
 
    pip install silx
    
Or with Anaconda on Linux and MacOS::
    
    conda install silx -c conda-forge

To install silx locally, run::
 
    pip install silx --user

Unofficial packages for different distributions are available :

- Unofficial Debian8 packages are available at http://www.silx.org/pub/debian/
- CentOS 7 rpm packages are provided by Max IV at the following url: http://pubrepo.maxiv.lu.se/rpm/el7/x86_64/
- Fedora 23 rpm packages are provided by Max IV at http://pubrepo.maxiv.lu.se/rpm/fc23/x86_64/
- Arch Linux (AUR) packages are also available: https://aur.archlinux.org/packages/python-silx

Beside this, we provide a certain number of wheels (pre-compiled binary packages) to be installed
onto a pre-existing Python installation:

- On Windows, binary wheels are available for Python 2.7, 3.5 and 3.6.
- On MacOS, binary wheels are available for Python 2.7, 3.5 and 3.6.
- On Linux, manylinux1 binary wheels are available for Python 2.7, 3.4, 3.5 and 3.6.

Those builds are made from "up-date" systems at the time of the release, i.e. they use
the latest stable version of numpy (and cython). 
Hence your system should use a fairly recent version of numpy to be compatible with silx.
This can be achieved simply by::

    pip install numpy --upgrade


The latest development version can be obtained from the git repository::

    git clone https://github.com/silx-kit/silx.git
    cd silx
    pip install . [--user]

Dependencies
------------

* `Python <https://www.python.org/>`_ 2.7, 3.4 or above.
* `numpy <http://www.numpy.org>`_

The GUI widgets of the silx package depend on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ (using API version 2) or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_ for the silx.gui.plot package
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_ for the silx.gui.plot3d package

Most modules and functions dealing with `HDF5 <https://www.hdfgroup.org/HDF5/>`_ input/output depend on:

* `h5py <http://www.h5py.org/>`_

Parallel algorithms depend on:

* `PyOpenCL <https://documen.tician.de/pyopencl/>`_

The console widgets depend on:

* `ipython <https://ipython.org/>`_
* `qtconsole <https://pypi.python.org/pypi/qtconsole>`_


Supported platforms: Linux, Windows, Mac OS X.

Documentation
-------------

Documentation of latest release is available at http://www.silx.org/doc/silx/latest/

Documentation of previous releases and nightly build is available at http://www.silx.org/doc/silx/

To build the documentation from the source (requires `Sphinx <http://www.sphinx-doc.org>`_), run::

    python setup.py build build_doc

Testing
-------

- Travis CI status: |Travis Status|
- Appveyor CI status: |Appveyor Status|

To run the tests from the python interpreter, run:

>>> import silx.test
>>> silx.test.run_tests()

To run the tests, from the source directory, run::

    python run_tests.py

Examples
--------

Some examples of sample code using silx are provided with the
`silx documentation <http://www.silx.org/doc/silx/dev/sample_code/index.html>`_.


License
-------

The source code of silx is licensed under the MIT license.
See the `LICENSE <https://github.com/silx-kit/silx/blob/master/LICENSE>`_ and `copyright <https://github.com/silx-kit/silx/blob/master/copyright>`_ files for details.

Citation
--------

silx releases can be cited by their DOI on Zenodo: |zenodo DOI|

.. |Travis Status| image:: https://travis-ci.org/silx-kit/silx.svg?branch=master
   :target: https://travis-ci.org/silx-kit/silx
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/qgox9ei0wxwfagrb/branch/master?svg=true
   :target: https://ci.appveyor.com/project/ESRF/silx
.. |zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.591709.svg
   :target: https://doi.org/10.5281/zenodo.591709
