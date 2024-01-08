silx |version|
==============

.. toctree::
   :hidden:

   user_guide.rst
   applications/index.rst
   tutorials.rst
   modules/index.rst
   changelog.rst

.. |silxView| image:: http://www.silx.org/doc/silx/img/silx-view-v1-0.gif
   :width: 480px

silx provides applications and Python modules to support the
development of data assessment, reduction and analysis at synchrotron radiation
facilities.
It provides reading/writing tools for different file formats, data
reduction routines and a set of Qt widgets to browse and visualise data.

:doc:`install`
--------------

You can install ``silx`` via `pip <https://pypi.org/project/pip>`_, `conda <https://docs.conda.io>`_ or on Linux with the following commands:

.. tabs::

   .. tab:: pip

      .. code-block:: bash

         pip install silx[full]

   .. tab:: conda

      .. code-block:: bash

         conda install -c conda-forge silx

   .. tab:: Debian & Ubuntu

      .. code-block:: bash

         sudo apt-get install silx

:doc:`applications/index`
-------------------------

The :ref:`silx view` unified viewer supports HDF5, SPEC and image file formats:

|silxView|


Python package
--------------

Features:

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

Resources:

- :doc:`tutorials`
- :doc:`modules/gui/gallery`
- :doc:`modules/index`
