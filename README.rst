
silx toolkit
============

The silx project aims at providing a collection of Python packages to support the development of data assessment, reduction and analysis applications at synchrotron radiation facilities.
It aims at providing reading/writing different file formats, data reduction routines and a set of Qt widgets to browse and visualize data.

The current version provides reading `SPEC <https://certif.com/spec.html>`_ file format, histogramming, curves and image plot widget with a set of associated tools
(See `changelog file <https://github.com/silx-kit/silx/blob/master/CHANGELOG.rst>`_).

Installation
------------

To install silx, run::
 
    pip install silx

To install silx locally, run::
 
    pip install silx --user

On Linux, to install silx with pip, you must install numpy first.
Unofficial Debian8 packages are available at http://www.silx.org/pub/debian/

On Windows, pre-compiled binaries (aka Python wheels) are available for Python 2.7 and 3.5.

On Mac OS X, pre-compiled binaries (aka Python wheels) are available for Python 2.7.


The latest development version can be obtained from the git repository::

    git clone https://github.com/silx-kit/silx.git
    cd silx
    pip install . [--user]

Dependencies
------------

* `Python <https://www.python.org/>`_ 2.7, 3.4 and 3.5.
* `numpy <http://www.numpy.org>`_

The GUI widgets of the silx package depend on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ (using API version 2) or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_

Most modules and functions dealing with `HDF5 <https://www.hdfgroup.org/HDF5/>`_ input/output depend on the following extra package:
* `h5py <http://www.h5py.org/>`_

Supported platforms: Linux, Windows, Mac OS X.

Documentation
-------------

Documentation of releases is available at https://pythonhosted.org/silx/

Latest documentation (nightly build) is available at http://www.silx.org/doc/silx/

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

License
-------

The source code of silx is licensed under the MIT and LGPL licenses.
See the `copyright file <https://github.com/silx-kit/silx/blob/master/copyright>`_ for details.

.. |Travis Status| image:: https://travis-ci.org/silx-kit/silx.svg?branch=master
   :target: https://travis-ci.org/silx-kit/silx
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/qgox9ei0wxwfagrb/branch/master?svg=true
   :target: https://ci.appveyor.com/project/ESRF/silx
