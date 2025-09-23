User Guide
==========


.. |silxView| image:: http://www.silx.org/doc/silx/img/silx-view-v1-0.gif
   :height: 80px

The silx project aims to provide a collection of Python packages to support the
development of data assessment, reduction and analysis at synchrotron radiation
facilities.
It intends to provide reading/writing tools for different file formats, data
reduction routines and a set of Qt widgets to browse and visualise data.
Silx can be cited by its DOIs referenced on
`Zenodo <https://doi.org/10.5281/zenodo.591709>`_.

The current version (v\ |version|) caters for:

* Supporting `HDF5 <https://www.hdfgroup.org/HDF5/>`_,
  `SPEC <https://certif.com/spec.html>`_ and
  `FabIO <http://www.silx.org/doc/fabio/dev/getting_started.html#list-of-file-formats-that-fabio-can-read-and-write>`_
  images file formats.
* OpenCL-based data processing: image alignment (SIFT),
  image processing (median filter, histogram),
  filtered backprojection for tomography
* Data reduction: histogramming, fitting, median filter
* A set of Qt widgets, including:

  * 1D and 2D visualization widgets with a set of associated tools using multiple backends (matplotlib or OpenGL)
  * OpenGL-based widgets to visualize data in 3D (scalar field with isosurface and cut plane, scatter plot)
  * a unified browser for HDF5, SPEC and image file formats supporting inspection and
    visualization of n-dimensional datasets.

* a set of applications:

    * a unified viewer (:ref:`silx-view` *filename*) for HDF5, SPEC and image file formats

      |silxView|

    * a unified converter to HDF5 format (*silx convert filename*)


.. toctree::
   :hidden:

   overview.rst
   install.rst
   description/index.rst
   license.rst
   troubleshooting.rst
   contribute/index.rst

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

:doc:`troubleshooting`
    When things do not work as expected

:doc:`contribute/index`
    How to contribute to the *silx* project


Indices
=======

* :ref:`modindex`
* :ref:`genindex`
