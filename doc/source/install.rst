
Installation steps
==================

*silx* supports most operating systems, different version of the Python
programming language.
While `numpy <http://www.numpy.org/>`_ is the only mandatory dependency,
graphical widgets require Qt, management of data files requires
`h5py <http://docs.h5py.org/en/latest/build.html>`_ and
`fabio <https://github.com/silx-kit/fabio>`_, and high performance data-analysis
code on GPU requires `pyopencl <https://mathema.tician.de/software/pyopencl/>`_.

This table summarized the the support matrix of silx v0.7:

+------------+--------------+---------------------+
| System     | Python vers. | Qt and its bindings |
+------------+--------------+---------------------+
| `Windows`_ | 3.5, 3.6     | PyQt5.6+            |
+------------+--------------+---------------------+
| `MacOS`_   | 2.7, 3.5-3.6 | PyQt5.6+            |
+------------+--------------+---------------------+
| `Linux`_   | 2.7, 3.4-3.6 | PyQt4.8+, PyQt5.3+  |
+------------+--------------+---------------------+

For all platform, you can install *silx* from the source, see `Installing from source`_.

To install *silx* in a `Virtual Environment`_, there is short version here-after
and  a `longer description :ref:`silx-venv`.

Dependencies
------------

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PyQt5, PyQt4 <https://riverbankcomputing.com/software/pyqt/intro>`_,
  `PySide <https://pypi.python.org/pypi/PySide/>`_, or `PySide2 <https://wiki.qt.io/PySide2>`_
* `matplotlib <http://matplotlib.org/>`_
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_
* `qt_console <https://pypi.python.org/pypi/qtconsole>`_
  for the ``silx.gui.console`` widget.
* `dateutil <https://pypi.org/project/python-dateutil/>`_

Tools for reading and writing files depend on the following packages:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_ for HDF5 files
* `fabio <https://github.com/silx-kit/fabio>`_ for multiple image formats

*silx.opencl* further depends on OpenCL and the following packages to :

* `pyopencl <https://mathema.tician.de/software/pyopencl/>`_
* `Mako <http://www.makotemplates.org/>`_

The complete list of dependencies with the minimal version is described in the
`requirement.txt <https://github.com/silx-kit/silx/blob/0.7/requirements.txt>`_
at the top level of the source package.

Build dependencies
++++++++++++++++++

In addition to run-time dependencies, building *silx* requires a C/C++ compiler,
`numpy <http://www.numpy.org/>`_ and `cython <http://cython.org>`_ (optional).

On Windows it is recommended to use Python 3.5, because with previous versions
of Python, it might be difficult to compile extensions (i.e. binary modules).

This project uses Cython (version > 0.21) to generate C files.
Cython is now mandatory to build *silx* from the development branch and is only
needed when compiling binary modules.

The complete list of dependencies for building the package, including its
documentation, is described in the
`requirement-dev.txt <https://github.com/silx-kit/silx/blob/0.7/requirements-dev.txt>`_
at the top level of the source package.



Linux
-----

If NumPy is not installed on your system, you need to install it first,
preferably with the package manager of your system.
If you cannot use the package manager of your system (which requires the root
access), please refer to the `Virtual Environment`_ procedure.

On Linux, you can install *silx* in your home directory

.. code-block:: bash 

    pip install silx --user

.. note::
    
    Replace the ``pip`` command with ``pip3`` to install *silx* or any other library for Python 3.

.. note::
    
    This installs *silx* without the optional dependencies. 
    
To install *silx* on Debian or Ubuntu systems, see `Installing a Debian package`_.
This method requires **sudo** privileges, but has the benefit of installing
dependencies in a simple way.

`CentOS 7 RPM packages <http://pubrepo.maxiv.lu.se/rpm/el7/x86_64/>`_ and
`Fedora 23 rpm packages <http://pubrepo.maxiv.lu.se/rpm/fc23/x86_64/>`_
are provided by the Max IV institute at Lund, Sweden.

An `Arch Linux (AUR) package <https://aur.archlinux.org/packages/python-silx>`_
is provided by Leonid Bloch.

You can also choose to compile and install *silx* from it's sources:
see `Installing from source`_.

.. note::

    The Debian packages `python-silx` and `python3-silx` will not install executables 
    (`silx view`, `silx convert` ...). Please install the silx package.  


Installing a Debian package
+++++++++++++++++++++++++++

Debian 8 (Jessie) packages are available on http://www.silx.org/pub/debian/ for amd64 computers.
To install it, you need to download this file

.. code-block:: bash 

    http://www.silx.org/pub/debian/silx.list

and copy it into the /etc/apt/source.list.d folder.
Then run ``apt-get update`` and ``apt-get install python-silx``

.. code-block:: bash 

   wget http://www.silx.org/pub/debian/silx.list
   sudo cp silx.list /etc/apt/sources.list.d
   sudo apt-get update
   sudo apt-get install python-silx python3-silx silx

.. note::
    
    The packages are built automatically, hence not signed. 
    You have to accept the installation of non-signed packages.  

If the packages are not installed, it might be due to the priority list.
You can display the priority list using `apt-cache policy python-silx`.
If the Pin-number of silx.org is too low compared to other sources:
download http://www.silx.org/pub/debian/silx.pref into /etc/apt/preferences.d
and start the update/install procedure again.

Virtual Environment
-------------------

Virtual environments are self-contained directory tree that contains a Python
installation for a particular version of Python, plus a number of additional
packages.
They do require administrator privileges, nor *root* access.

To create a virtual environment, decide upon a directory where you want to place
it (for example *myenv*), and run the *venv* module as a script with the directory path:

.. code-block:: bash 

    python3 -m venv  myenv

This will create the *myenv* directory if it doesn’t exist, and also create
directories inside it containing a copy of the Python interpreter, the standard
library, and various supporting files.

Once you’ve created a virtual environment, you may activate it.

On Windows, run:

.. code-block:: bash 

  myenv\\Scripts\\activate.bat

On Unix or MacOS, run:

.. code-block:: bash 

   source myenv/bin/activate

You can install, upgrade, and remove packages using a program called *pip* within
your virtual environment.

.. code-block:: bash 

    pip install numpy
    pip install -r https://github.com/silx-kit/silx/raw/0.7/requirements.txt
    pip install silx
    
Windows
-------

The simple way of installing the *silx* library on Windows is to type the following
commands in a command prompt:

.. code-block:: bash

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

We recommend that you install the 64bits version of Python, which is not the
default version suggested on the Python website.
The 32bits version is limited to 2 GB of memory, and also we don't provide a
binary wheel for it.
This means that you would have to install *silx* from its sources, which requires
you to install a C compiler first.

We also encourage you to use Python 3.5 or newer, former versions are no more
officially supported.

Configure Python as explained on
`docs.python.org <https://docs.python.org/3/using/windows.html#configuring-python>`_
to add the python installation directory to your PATH environment variable.

Alternative Scientific Python stacks exists, such as
`WinPython <http://winpython.github.io/>`_ or `Anaconda <https://www.anaconda.com/download/#windows>`_.
They all offer most of the scientific packages already installed which makes the
installation of dependencies much easier.

Using pip
+++++++++

Configure your PATH environment variable to include the pip installation
directory, the same way as described for Python.

The pip installation directory will likely be ``C:\Python35\Scripts\``.

Then you will be able to use all pip commands listed in following in a command
prompt.


Installing dependencies
+++++++++++++++++++++++

All dependencies may be simply installed with pip::

.. code-block:: bash 

    pip install -r https://github.com/silx-kit/silx/raw/0.7/requirements.txt


Installing *silx*
+++++++++++++++++

Provided numpy is installed, you can install *silx* with::

.. code-block:: bash 

    pip install silx


MacOS
-----

While Apple ships Python 2.7 by default on their operating systems, we recommand
using Python 3.5 or newer to ease the installation of the Qt library.
This can simply be performed by:

.. code-block:: bash 

    pip install -r https://github.com/silx-kit/silx/raw/0.7/requirements.txt

Then install *silx* with:

.. code-block:: bash 

    pip install silx

This should work without issues, as binary wheels of *silx* are provided on
PyPi.


Installing from source
----------------------

Building *silx* from the source requires some `Build dependencies`_ which may be
installed using:

.. code-block:: bash 

    pip install -r https://github.com/silx-kit/silx/raw/0.7/requirements-dev.txt


Building from source
++++++++++++++++++++

Source package of *silx* releases can be downloaded from
`the pypi project page <https://pypi.python.org/pypi/silx>`_.

After downloading the `silx-x.y.z.tar.gz` archive, extract its content::

    tar xzvf silx-x.y.z.tar.gz
    
Alternatively, you can get the latest source code from the master branch of the
`git repository <https://github.com/silx-kit/silx/archive/master.zip>`_: https://github.com/silx-kit/silx

You can now build and install *silx* from its sources:


.. code-block:: bash 

    cd silx-x.y.z
    pip uninstall -y silx
    pip install . [--user]

Known issues
............

There are specific issues related to MacOSX. If you get this error::

  UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 1335: ordinal not in range(128)

This is related to the two environment variable LC_ALL and LANG not defined (or wrongly defined to UTF-8).
To set the environment variable, type on the command line:

.. code-block:: bash 

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8

Advanced build options
++++++++++++++++++++++

In case you want more control over the build procedure, the build command is:

.. code-block:: bash 

    python setup.py build

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if installed) to re-generate the C source code.
  Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default for MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (default for Windows and Linux).

Run the test suite of silx (may take a couple of minutes):

.. code-block:: bash 

    python run_tests.py

Package the built into a wheel and install it:

.. code-block:: bash 

    python setup.py bdist_wheel
    pip install dist/silx*.whl 

To build the documentation, using  `Sphinx <http://www.sphinx-doc.org/>`_:

.. code-block:: bash 

    python setup.py build build_doc


Testing
+++++++

To run the tests of an installed version of *silx*, from the python interpreter, run:

.. code-block:: python
    
     import silx.test
     silx.test.run_tests()

To run the test suite of a development version, use the *run_tests.py* script at
the root of the source project.

.. code-block:: bash
    
     python ./run_tests.py
