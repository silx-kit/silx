silx |version|
==============

.. toctree::
   :hidden:

   user_guide.rst
   applications/index.rst
   tutorials.rst
   modules/index.rst
   changelog.rst

silx provides applications and Python modules to support the
development of data assessment, reduction and analysis at synchrotron radiation
facilities.
It provides reading/writing tools for different file formats, data
reduction routines and a set of Qt widgets to browse and visualise data.

:ref:`Installation`
-------------------

You can install **silx** via `pip <https://pypi.org/project/pip>`_, `conda <https://docs.conda.io>`_ or on Linux with the following commands:

.. tab-set::

   .. tab-item:: pip

      .. code-block:: bash

         pip install silx[full]

   .. tab-item:: conda

      .. code-block:: bash

         conda install -c conda-forge silx

   .. tab-item:: Debian & Ubuntu

      .. code-block:: bash

         sudo apt-get install silx

|silx_installer_btn| or decompress the |silx_archive|.

:ref:`Applications`
-------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::
      :columns: 12
      :text-align: center
      :link: silx-view
      :link-type: ref

      silx view
      ^^^

      .. image:: img/silx-view.gif

      Unified viewer supporting HDF5, SPEC and image file formats

   .. grid-item-card::
      :text-align: center
      :link: silx-compare
      :link-type: ref

      silx compare
      ^^^

      .. image:: applications/img/silx-compare.png

      User interface to compare 2D data from files

   .. grid-item-card::
      :text-align: center
      :link: silx-convert
      :link-type: ref

      silx convert
      ^^^

      Converter of legacy file formats into HDF5 file

:ref:`Python modules<API Reference>`
------------------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Qt widgets
      :link: module-silx-gui
      :link-type: ref

      silx.gui
      ^^^
      * 1D and 2D visualization widgets and associated tools
      * OpenGL-based 3D visualization widgets
      * A unified HDF5, SPEC and image data file browser and n-dimensional dataset viewer

   .. grid-item-card:: OpenCL-based data processing
      :link: module-silx-opencl
      :link-type: ref

      silx.opencl
      ^^^

      * Image alignment (SIFT)
      * Image processing (median filter, histogram)
      * Filtered backprojection for tomography

   .. grid-item-card:: Supporting HDF5, SPEC and FabIO images file formats
      :link: module-silx-io
      :link-type: ref

      silx.io
      ^^^

   .. grid-item-card:: Data reduction: histogramming, fitting, median filter
      :link: module-silx-math
      :link-type: ref

      silx.math
      ^^^
