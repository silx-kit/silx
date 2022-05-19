
silx toolkit
============

.. |silxView| image:: http://www.silx.org/doc/silx/img/silx-view-v1-0.gif
   :height: 480px

The purpose of the *silx* project is to provide a collection of Python packages to support the
development of data assessment, reduction and analysis applications at synchrotron
radiation facilities.
*silx* aims to provide reading/writing tools for different file formats, data reduction routines
and a set of Qt widgets to browse and visualise data.

The current version features:

* Support of `HDF5 <https://www.hdfgroup.org/HDF5/>`_,
  `SPEC <https://certif.com/spec.html>`_ and
  `FabIO <http://www.silx.org/doc/fabio/dev/getting_started.html#list-of-file-formats-that-fabio-can-read-and-write>`_
  images file formats.
* OpenCL-based data processing: image alignment (SIFT),
  image processing (median filter, histogram),
  filtered backprojection for tomography,
  convolution
* Data reduction: histogramming, fitting, median filter
* A set of Qt widgets, including:

  * 1D and 2D visualization widgets with a set of associated tools using multiple backends (matplotlib or OpenGL)
  * OpenGL-based widgets to visualize data in 3D (scalar field with isosurface and cut plane, scatter plot)
  * a unified browser for HDF5, SPEC and image file formats supporting inspection and
    visualization of n-dimensional datasets.

* a set of applications:

  * a unified viewer (*silx view filename*) for HDF5, SPEC and image file formats
  
    |silxView|

  * a unified converter to HDF5 format (*silx convert filename*)


Installation
------------

To install silx (and all its dependencies), run:

.. code-block:: bash

    pip install silx[full]

To install silx with a minimal set of dependencies, run:

.. code-block:: bash

    pip install silx

Or using Anaconda on Linux and MacOS:

.. code-block:: bash

    conda install silx -c conda-forge

Unofficial packages for different distributions are available:

- Unofficial Debian10 and Ubuntu20.04 packages are available at http://www.silx.org/pub/linux-repo/
- CentOS 7 rpm packages are provided by Max IV at: http://pubrepo.maxiv.lu.se/rpm/el7/x86_64/
- Fedora 23 rpm packages are provided by Max IV at http://pubrepo.maxiv.lu.se/rpm/fc23/x86_64/
- Arch Linux (AUR) packages are also available: https://aur.archlinux.org/packages/python-silx

`Detailed installation instructions <http://www.silx.org/doc/silx/dev/install.html>`_
are available in the documentation.

Documentation
-------------

The documentation of `latest release <http://www.silx.org/doc/silx/latest/>`_ and
the documentation of `the nightly build <http://www.silx.org/doc/silx/dev>`_ are
available at http://www.silx.org/doc/silx/

Testing
-------

*silx* features a comprehensive test-suite used in continuous integration for
all major operating systems:

- Github Actions CI status: |Github Actions Status|
- Appveyor CI status: |Appveyor Status|

Please refer to the `documentation on testing <http://www.silx.org/doc/silx/dev/install.html#testing>`_
for details.

Examples
--------

Some examples of sample code using silx are provided with the
`silx documentation <http://www.silx.org/doc/silx/dev/sample_code/index.html>`_.


License
-------

The source code of *silx* is licensed under the MIT license.
See the `LICENSE <https://github.com/silx-kit/silx/blob/master/LICENSE>`_ and
`copyright <https://github.com/silx-kit/silx/blob/master/copyright>`_ files for details.

Citation
--------

*silx* releases can be cited via their DOI on Zenodo: |zenodo DOI|

.. |Github Actions Status| image:: https://github.com/silx-kit/silx/workflows/CI/badge.svg
   :target: https://github.com/silx-kit/silx/actions
.. |Appveyor Status| image:: https://ci.appveyor.com/api/projects/status/qgox9ei0wxwfagrb/branch/master?svg=true
   :target: https://ci.appveyor.com/project/ESRF/silx?branch=master
.. |zenodo DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.591709.svg
   :target: https://doi.org/10.5281/zenodo.591709
