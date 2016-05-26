
Simple instructions
===================

To install *silx* on Windows, read the `Windows instructions`_.

To install *silx* on Linux, read the `Linux instructions`_.

To install *silx* on Mac Os X, read the `Mac OS X instructions`_.

You will find the simple instructions for each platform at the beginning
of each section, followed by more detailed instructions concerning
dependencies and alternative install methods.

Dependencies
============

Build dependencies
------------------

In addition to the run-time dependencies, building *silx* requires a C compiler
and Numpy.

The supported Python versions are: 2.7, 3.4, 3.5

On Windows it is recommended to use Python 3.5, because with previous Python
versions it might be difficult to compile the extensions.

C files are generated from `cython <http://cython.org>`_. Cython is only
needed for developing new binary modules. If you want to generate your own C
files, make sure your local Cython version supports memory-views (available
from Cython v0.17 and newer).

Optional dependencies
---------------------

The GUI widgets depend on the following extra packages:

* A Qt binding: `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_

Tools for reading and writing HDF5 files depend on the following package:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_


Linux instructions
==================

If NumPy is not installed on your system, you need to install it first::

    pip install numpy --user

On Linux, we recommend you install *silx* in your home
directory::

    pip install silx --user
    
.. note::
    
    Maybe you'll need to replace the ``pip`` command with ``pip3`` to install
    *silx* or any other library for Python 3.
        
.. note::
    
    This installs *silx* without the optional dependencies. 
    
To install *silx* on Debian, you should probably use the solution presented
in `Installing a Debian package`_. This method requires **sudo** privileges, but
has the benefit of installing dependencies in a simple way.

You can also choose to compile and install *silx* from it's sources: 
see `Source package`_.

Installing a Debian package
---------------------------

Debian 8 (Jessie) packages are available on http://www.silx.org/pub for amd64
computers.
To install it, you need to download this file::

    http://www.silx.org/pub/debian/silx.list

and copy it into the /etc/apt/source.list.d folder.
Then run ``apt-get update`` and ``apt-get install python-silx``

:: 

   wget http://www.silx.org/pub/debian/silx.list
   sudo cp silx.list /etc/apt/sources.list.d
   sudo apt-get update
   sudo apt-get install python-silx python3-silx

.. note::
    
    The packages are built automatically, hence not signed. 
    You have to accept the installation of non-signed packages.  

If the packages are not installed, it might be due to the priority list.
You can display the priority list using `apt-cache policy python-silx`.
If the Pin-number of silx.org is too low compared to other sources:
download http://www.silx.org/pub/debian/silx.pref into /etc/apt/preferences.d
and start the update/install procedure again.

Source package
--------------

A source package can be downloaded from `the pypi project page <https://pypi.python.org/pypi/silx>`_.

After downloading the `silx-x.y.z.tar.gz` archive, extract its content::

    tar xzvf silx-x.y.z.tar.gz
    
You can now build and install *silx* from its sources::

    cd silx-x.y.z
    sudo pip install . --upgrade
    
Or::

    cd silx-x.y.z
    pip install . --user --upgrade
    
The ``--upgrade`` option is not mandatory, but it ensures that you install the
downloaded version even if a previous version of *silx* was already installed.

Advanced building options
-------------------------

In case you want more control over the build procedure, the build command is::

    python setup.py build
    
After this build, you will still need to install *silx* to be able to use it::

    pip install . --upgrade

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if present) to re-generate the C source code.
  Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default under MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (Default under Windows and Linux).


Windows instructions
====================

The simple way of installing the *silx* library on Windows is to type following
commands in a command prompt::

    pip install silx
  
.. note::
    
    This installs *silx* without the optional dependencies.
    Instructions on how to install dependencies are given in the
    `Installing dependencies`_ section.
    
This assumes you have Python and pip installed and configured. If you don't,
read the following sections.


Installing Python
-----------------

Download and install Python from `python.org <https://www.python.org/downloads/>`_.
Python 3.5 or newer is recommended.

Configure Python as explained on `docs.python.org
<https://docs.python.org/3/using/windows.html#configuring-python>`_ to add
the python installation directory to your PATH environment variable.


Installing pip
--------------

Recent version of Python (`> 2.7.9` or `> 3.4`) provide pip by default.

If you have an older version of Python and you do not wish to upgrade it,
you can install pip yourself.

Download the script https://bootstrap.pypa.io/get-pip.py and execute it in a
command prompt::

    python get-pip.py  


Using pip
---------

Configure your PATH environment variable to include the pip installation
directory, the same way as described for Python.

The pip installation directory will likely be ``C:\Python35\Scripts\``.

Then you will be able to use all pip commands listed in following in a command
prompt.


Installing dependencies
-----------------------

Some of the dependencies may be simply installed with pip::

    pip install numpy
    pip install matplotlib
    pip install PyQt5
    pip install PySide

Regarding the `h5py` and `PyQt4` modules, you can find the wheels at
Christoph Gohlke's repository:

http://www.lfd.uci.edu/~gohlke/pythonlibs/

Download the appropriate `.whl` file for your system and install them with pip::

    pip install h5py*.whl
    pip install PyQt4*.whl
    
`PyQt5` can be downloaded as a binary package for `Python 3.5` on the
`Riverbank Computing website <https://www.riverbankcomputing.com/software/pyqt/download5>`_.
This package contains everything needed for `PyQt5`, including `Qt`.


Installing *silx*
-----------------

After numpy is installed, you can install *silx* with::

    pip install silx


Mac OS X instructions
=====================

The easy way to install *silx* on Mac OS X, is the same as on other platforms::

    pip install silx

This should work without issues, as binary wheels of *silx* are provided on
PyPi. The tricky part is to install the optional dependencies.

Until recently, the `h5py` developers provided Mac OS X wheels. Therefore,
the easiest way to install `h5py` on this system is to get an older version
using pip::

    pip install h5py==2.5.0
    
If you require `h5py` version `2.6.0` or newer, you will need to compile it as well as
it's dependencies (mainly HDF5) yourself.

A PyQt5 wheel is now available for Python 3.5 on Mac OS X:
https://pypi.python.org/simple/pyqt5/.
Download it and install it with::

    pip install PyQt5-5.6-cp35-cp35m-macosx_10_6_intel.whl

This should work for all versions of Mac OS X from 10.6.