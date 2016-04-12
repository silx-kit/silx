
silx toolkit
============

The silx project aims at providing a collection of Python packages to support the development of data assessment, reduction and analysis applications at synchrotron radiation facilities.
It aims at providing reading/writing different file formats, data reduction routines and a set of Qt widgets to browse and visualize data.

The current version provides reading `SPEC <https://certif.com/spec.html>_` file format, histogramming, curves and image plot widget with a set of associated tools
(See `changelog file <https://github.com/silx-kit/silx/blob/master/CHANGELOG.rst>`_).

Installation
------------

.. After release
 To install silx, run::
 
     pip install silx
 
 To install silx locally, run::
 
     pip install silx --user

The latest development version can be obtained from the git repository::

    git clone https://github.com/silx-kit/silx.git
    cd silx
    pip install . [--user]

Dependencies
------------

* `Python <https://www.python.org/>`_ 2.7, 3.4 and 3.5.
* `numpy <http://www.numpy.org>`_
* `h5py <http://www.h5py.org/>`_

The GUI widgets of the silx package depends on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_

Documentation
-------------

.. After release
  Documentation of releases is available at http://www.pythonhosted.org/silx

To build the documentation from the source (requires `Sphinx <http://www.sphinx-doc.org>`_), run::

    python setup.py build build_doc

License
-------

The source code of silx is licensed under the MIT and LGPL licenses.
See the `copyright file <https://github.com/silx-kit/silx/blob/master/copyright>`_ for details.
