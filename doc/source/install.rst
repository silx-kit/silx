
Installation steps
==================

*silx* supports `Python <https://www.python.org/>`_ versions 2.7, 3.4 and 3.5.

To install *silx* on Windows, read the `Windows instructions`_.

To install *silx* on Linux, read the `Linux instructions`_.

To install *silx* on Mac Os X, read the `Mac OS X instructions`_.

You will find the simple instructions for each platform at the beginning of each section, followed by more detailed instructions concerning dependencies and alternative installation methods.

For all platform, to install *silx* from the source, see `Installing from source`_.


Dependencies
------------

The only mandatory dependency of *silx* is `numpy <http://www.numpy.org/>`_.

Yet, a set of `Optional dependencies`_ is necessary to enable all *silx* features.

Optional dependencies
+++++++++++++++++++++

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_ or `PySide <https://pypi.python.org/pypi/PySide/>`_
* `matplotlib <http://matplotlib.org/>`_
* `IPython <https://ipython.org/>`_ and `qt_console <https://pypi.python.org/pypi/qtconsole>`_ for the ``silx.gui.console`` widget.

Tools for reading and writing HDF5 files depend on the following package:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_

Build dependencies
++++++++++++++++++

In addition to run-time dependencies, building *silx* requires a C/C++ compiler, `numpy <http://www.numpy.org/>`_ and `cython <http://cython.org>`_ (optional).

On Windows it is recommended to use Python 3.5, because with previous versions of Python, it might be difficult to compile the extensions.

This project uses cython to generate C files.
Cython is not mandatory to build *silx* and is only needed when developing binary modules.
If using cython, *silx* requires at least version 0.18 (with memory-view support).


Linux instructions
------------------

If NumPy is not installed on your system, you need to install it first
either with the package manager of your system (recommended way) or with pip::

    pip install numpy --user

On Linux, you can install *silx* in your home directory::

    pip install silx --user

.. note::
    
    Replace the ``pip`` command with ``pip3`` to install *silx* or any other library for Python 3.

.. note::
    
    This installs *silx* without the optional dependencies. 
    
To install *silx* on Debian 8, see `Installing a Debian package`_.
This method requires **sudo** privileges, but has the benefit of installing dependencies in a simple way.

You can also choose to compile and install *silx* from it's sources: 
see `Installing from source`_.


Installing a Debian package
+++++++++++++++++++++++++++

Debian 8 (Jessie) packages are available on http://www.silx.org/pub/debian/ for amd64 computers.
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


Windows instructions
--------------------

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
+++++++++++++++++

Download and install Python from `python.org <https://www.python.org/downloads/>`_. 

We recommend that you install the 64bits version of Python, which is not the default version suggested on the Python website. The 32bits version is limited to 2 GB of memory, and also we don't provide a silx wheel for it. This means that you would have to install silx from its sources, which requires you to install a C compiler first.

We also encourage you to use Python 3.5 or newer.

Configure Python as explained on `docs.python.org
<https://docs.python.org/3/using/windows.html#configuring-python>`_ to add
the python installation directory to your PATH environment variable.

Alternative Scientific Python stacks exists, such as `WinPython <http://winpython.github.io/>`_.
They all offer most of the scientific packages already installed which makes the installation of dependencies much easier.

Installing pip
++++++++++++++

Recent version of Python (`> 2.7.9` or `> 3.4`) provide pip by default.

If you have an older version of Python and you do not wish to upgrade it,
you can install pip yourself.

Download the script https://bootstrap.pypa.io/get-pip.py and execute it in a
command prompt::

    python get-pip.py  


Using pip
+++++++++

Configure your PATH environment variable to include the pip installation
directory, the same way as described for Python.

The pip installation directory will likely be ``C:\Python35\Scripts\``.

Then you will be able to use all pip commands listed in following in a command
prompt.


Installing dependencies
+++++++++++++++++++++++

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
+++++++++++++++++

Provided numpy is installed, you can install *silx* with::

    pip install silx


Mac OS X instructions
---------------------

The easy way to install *silx* on Mac OS X, is::

    pip install silx

This should work without issues, as binary wheels of *silx* are provided on
PyPi.
The tricky part is to install the optional dependencies.

Until recently, the `h5py` developers provided Mac OS X wheels.
Therefore, the easiest way to install `h5py` on this system is to get an older version using pip::

    pip install h5py==2.5.0
    
If you require `h5py` version `2.6.0` or newer, you will need to compile it as well as it's dependencies (mainly HDF5) yourself.

A PyQt5 wheel is now available for Python 3.5 on Mac OS X: https://pypi.python.org/simple/pyqt5/.
Download it and install it with::

    pip install PyQt5-5.6-cp35-cp35m-macosx_10_6_intel.whl

This should work for all versions of Mac OS X from 10.6.


Installing from source
----------------------

Building *silx* from the source requires some `Build dependencies`_.

Building from source
++++++++++++++++++++

Source package of *silx* releases can be downloaded from `the pypi project page <https://pypi.python.org/pypi/silx>`_.

After downloading the `silx-x.y.z.tar.gz` archive, extract its content::

    tar xzvf silx-x.y.z.tar.gz
    
Alternatively, you can get the latest source code from the master branch of the `git repository <https://github.com/silx-kit/silx>`_:  https://github.com/silx-kit/silx/archive/master.zip

You can now build and install *silx* from its sources::

    cd silx-x.y.z
    pip uninstall -y silx
    pip install . [--user]

Known issues
............

There are specific issues related to MacOSX. If thou get this error::

  UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 1335: ordinal not in range(128)

This is related to the two environment variable LC_ALL and LANG not defined (or wrongly defined to UTF-8).
To set the environment variable, type on the command line::

  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8

Advanced build options
++++++++++++++++++++++

In case you want more control over the build procedure, the build command is::

    python setup.py build

After this build, you will still need to install *silx* to be able to use it::

    python setup.py install [--user]

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if installed) to re-generate the C source code.
  Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default for MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (default for Windows and Linux).

To build the documentation (this requires `Sphinx <http://www.sphinx-doc.org/>`_), run::

    python setup.py build build_doc


Testing
+++++++

To run the tests of an installed version of *silx*, from the python interpreter, run:

>>> import silx.test
>>> silx.test.run_tests()
