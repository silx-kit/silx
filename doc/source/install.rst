
Simple instructions
===================

The simple way of installing the silx library is::

    pip install silx
    
On Linux, run this command as sudo::

    sudo pip install silx

If you don't have sudo permissions, you can install silx in your home 
directory::

    pip install silx --user
    
.. note::
    
    Replace the ``pip`` command with ``pip3`` to install silx or any
    other library for Python 3.
    
The rest of this document deals with specific technical details such as 
dependencies, building from sources, and alternative installation methods.
If you managed to install silx with one of the 3 previous commands and 
everything is working, you can stop reading.

Dependencies
============

silx is a Python library whose installation relies on Numpy.

The supported Python versions are: 2.7, 3.5

The GUI widgets depend on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_

Tools for reading and writing HDF5 files depend on the following package:

* h5py

  .. note::

      The instructions for installing h5py can be found here: http://docs.h5py.org/en/latest/build.html
      
Build dependencies
------------------

In addition to the run-time dependencies, building silx requires a C compiler.

C files are generated from `cython <http://cython.org>`_. Cython is only
needed for developing new binary modules. If you want to generate your own C
files, make sure your local Cython version supports memory-views (available
from Cython v0.17 and newer).

Installing silx
===============

Installing from Debian package
------------------------------

Debian 8 packages are available at http://www.edna-site.org/pub/debian/. 
This installation method has the advantage of taking care of the optional 
dependencies for you.

Download the following ``.deb`` files:

- ``python-silx_x.y.z-1~bpo8+1_amd64.deb`` (or ``python3-silx_x.y.z-1~bpo8+1_amd64.deb``)
- ``python-silx-doc_x.y.z-1~bpo8+1_all.deb``

.. note::
    
    Replace ``x.y.z`` with the actual version number.

Install them with the ``dpkg`` command::

    sudo dpkg -i python-silx_x.y.z-1~bpo8+1_amd64.deb python-silx-doc_x.y.z-1~bpo8+1_all.deb
    
Or for Python3::

    sudo dpkg -i python3-silx_x.y.z-1~bpo8+1_amd64.deb python-silx-doc_x.y.z-1~bpo8+1_all.deb

Source package
--------------

A source package can be downloaded from `the pypi project page <https://pypi.python.org/pypi/silx>`_.

After downloading the `silx-x.y.z.tar.gz` archive, extract its content::

    tar xzvf silx-x.y.z.tar.gz
    
You can now build and install silx from its sources::

    cd silx-x.y.z
    sudo pip install . --upgrade
    
Or::

    cd silx-x.y.z
    pip install . --user --upgrade
    
The ``--upgrade`` option is not mandatory, but it ensures that you install the
downloaded version even if a previous version silx was already installed.

Advanced building options
-------------------------

In case you want more control over the build procedure, the build command is::

    python setup.py build
    
After this build, you will still need to install silx to be able to use it::

    pip install . --upgrade

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if present) to re-generate the C source code. 
  Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default under MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (Default under Windows and Linux).

Windows specific instructions
=============================

Installing pip
--------------

Recent version of Python (> 2.7.9 or > 3.4) provide pip by default.

If you have an older version of Python and you do not wish to upgrade it, 
you can install pip yourself.

Download the script https://bootstrap.pypa.io/get-pip.py and execute it::

    python get-pip.py
   

Installing dependencies
-----------------------

Some of the dependencies can be simply installed with pip::

    pip install numpy
    pip install matplotlib
    pip install PyQt4
    pip install PySide

Dependencies that are not available as a wheel may require the
very specific compiler used to compile your version of Python.
But in most cases you can find an unofficial source for the
wheel.

Regarding `h5py` module, you can find a wheel at Christoph Gohlke's repository:

http://www.lfd.uci.edu/~gohlke/pythonlibs/

Download the appropriate `.whl` file for your system and install it with pip::

    pip install h5py*.whl
    
Mac OS X specific instructions
==============================

Until recently, the `h5py` developers provided Mac OS X wheels. Therefore,
the easiest way to install `h5py` on this system is to get an older version
using pip::

    pip install h5py==2.5.0
    
Starting from version `2.6.0`, you will need to compile `h5py` and it's
dependencies (mainly HDF5) yourself.