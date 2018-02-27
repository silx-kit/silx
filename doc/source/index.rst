silx |version|
==============

The silx project aims at providing a collection of Python packages to support the
development of data assessment, reduction and analysis at synchrotron radiation
facilities.
It intends to provide reading/writing tools for different file formats, data
reduction routines and a set of Qt widgets to browse and visualise data.
Silx can be cited by its DOIs referenced on
`Zenodo <https://doi.org/10.5281/zenodo.591709>`_.

The current version (v0.7) caters for:

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
* image alignment (sift - OpenCL implementation)
* filtered backprojection for tomography

.. toctree::
   :hidden:

   overview.rst
   install.rst
   description/index.rst
   tutorials.rst
   modules/index.rst
   applications/index.rst
   changelog.rst
   license.rst
   virtualenv.rst

:doc:`overview`
    Releases, repository, issue tracker, mailing list, ...

:doc:`install`
    How to install *silx* on Linux, Windows and MacOS X

:doc:`description/index`
    Description of the different algorithms and their implementation

:doc:`tutorials`
    Tutorials and sample code

:doc:`modules/index`
    Documentation of the packages included in *silx*

:doc:`applications/index`
    Documentation of the applications provided by *silx*

:doc:`modules/gui/gallery`
    Widgets gallery and screenshots

:doc:`changelog`
    List of changes between releases

:doc:`license`
    License and copyright information

Indices
=======

* :ref:`modindex`
* :ref:`search`
* :ref:`genindex`
