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

.. tabbed:: pip

   .. code-block:: bash

      pip install silx[full]

.. tabbed:: conda

   .. code-block:: bash

      conda install -c conda-forge silx

.. tabbed:: Debian & Ubuntu

   .. code-block:: bash

      sudo apt-get install silx

|silx_installer_btn| or decompress the |silx_archive|.

:ref:`Applications`
-------------------

.. panels::

   :column: col-lg-12
   :body: text-center

   **silx view**
   ^^^^^^^^^^^^^

   .. image:: img/silx-view.gif

   .. link-button:: applications/view
      :type: ref
      :text: Unified viewer supporting HDF5, SPEC and image file formats
      :classes: stretched-link

   ---

   **silx compare**
   ^^^^^^^^^^^^^^^^

   .. image:: applications/img/silx-compare.png

   .. link-button:: applications/compare
      :type: ref
      :text: User interface to compare 2D data from files
      :classes: stretched-link

   ---

   **silx convert**
   ^^^^^^^^^^^^^^^^

   .. link-button:: applications/convert
      :type: ref
      :text: Converter of legacy file formats into HDF5 file
      :classes: stretched-link

:ref:`Python modules<API Reference>`
------------------------------------

.. panels::

   **silx.gui**
   ^^^^^^^^^^^^

   .. link-button:: modules/gui/index
      :type: ref
      :text: Qt widgets:
      :classes: stretched-link

   * 1D and 2D visualization widgets and associated tools
   * OpenGL-based 3D visualization widgets
   * a unified HDF5, SPEC and image data file browser and n-dimensional dataset viewer

   ---

   **silx.opencl**
   ^^^^^^^^^^^^^^^

   .. link-button:: modules/opencl/index
      :type: ref
      :text: OpenCL-based data processing:
      :classes: stretched-link

   * Image alignment (SIFT)
   * Image processing (median filter, histogram)
   * Filtered backprojection for tomography

   ---

   **silx.io**
   ^^^^^^^^^^^

   .. link-button:: modules/io/index
      :type: ref
      :text: Supporting HDF5, SPEC and FabIO images file formats
      :classes: stretched-link

   ---

   **silx.math**
   ^^^^^^^^^^^^^

   .. link-button:: modules/math/index
      :type: ref
      :text: Data reduction: histogramming, fitting, median filter
      :classes: stretched-link
