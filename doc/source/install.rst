
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
    
    Maybe you'll need to replace the ``pip`` command with ``pip3`` to install
    silx or any other library for Python 3.
        
.. note::
    
    This installs silx without the optional dependencies. 
    
The rest of this document deals with specific technical details such as 
dependencies, building from sources, and alternative installation methods.
If you managed to install silx with one of the 3 previous commands and 
everything is working, you can stop reading here.

Dependencies
============

silx is a Python library whose installation relies on Numpy.

The supported Python versions are: 2.7, 3.4, 3.5

On Windows it is recommended to use Python 3.5, because with previous Python
versions it might be difficult to compile the extensions.

The GUI widgets depend on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_

Tools for reading and writing HDF5 files depend on the following package:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_
      
Build dependencies
------------------

In addition to the run-time dependencies, building silx requires a C compiler.

C files are generated from `cython <http://cython.org>`_. Cython is only
needed for developing new binary modules. If you want to generate your own C
files, make sure your local Cython version supports memory-views (available
from Cython v0.17 and newer).

Installing silx
===============

Installing a Debian package
---------------------------

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

A source package can be downloaded from `the pypi project page <https://pypi.python.org/pypi/silx>`_.

After downloading the `silx-x.y.z.tar.gz` archive, extract its content::

    tar xzvf silx-x.y.z.tar.gz
    
You can now build and install silx from its sources::

    cd silx-x.y.z
    sudo pip install . --upgrade
    
Or::

    cd silx-x.y.z
    pip install . --user --upgrade
    
The ``--upgrade`` option is not mandatory, but it ensures that you install the
downloaded version even if a previous version of `silx` was already installed.

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

.. note::

    This assumes that the python interpreter is on your path. Otherwise,
    you need to specify the full python path (something like 
    ``c:\python35\python get-pip.py``)

Installing dependencies
-----------------------

Some of the dependencies can be simply installed with pip::

    pip install numpy
    pip install matplotlib
    pip install PyQt5
    pip install PySide

Dependencies that are not available as a wheel may require the
very specific compiler used to compile your version of Python.
But in most cases you can find an unofficial source for the
wheel.

Regarding the `h5py` and `PyQt4` modules, you can find the wheels at 
Christoph Gohlke's repository:

http://www.lfd.uci.edu/~gohlke/pythonlibs/

Download the appropriate `.whl` file for your system and install them with pip::

    pip install h5py*.whl
    pip install PyQt4*.whl
    
`PyQt5` can be downloaded as a binary package for `Python 3.5` on the 
`Riverbank Computing website <https://www.riverbankcomputing.com/software/pyqt/download5>`_.
This package contains everything needed for `PyQt5`, including `Qt`.

Mac OS X specific instructions
==============================

Until recently, the `h5py` developers provided Mac OS X wheels. Therefore,
the easiest way to install `h5py` on this system is to get an older version
using pip::

    pip install h5py==2.5.0
    
If you require `h5py` version `2.6.0`, you will need to compile it as well as
it's dependencies (mainly HDF5) yourself.

A PyQt5 wheel is now available for Python 3.5 on Mac OS X: 
https://pypi.python.org/simple/pyqt5/.
Download it and install it with::

    pip install PyQt5-5.6-cp35-cp35m-macosx_10_6_intel.whl

This should work for all versions of Mac OS X from 10.6. 