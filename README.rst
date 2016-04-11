
silx toolkit
============

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

* `PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_
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
