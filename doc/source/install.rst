
Installation steps
==================

*silx* supports most operating systems and different versions of the Python
programming language.

This table summarizes the support matrix of silx:

+------------+--------------+---------------------+
| System     | Python vers. | Qt and its bindings |
+------------+--------------+---------------------+
| `Windows`_ | 3.5, 3.6-3.7 | PyQt5.6+, PySide2   |
+------------+--------------+---------------------+
| `MacOS`_   | 2.7, 3.5-3.7 | PyQt5.6+, PySide2   |
+------------+--------------+---------------------+
| `Linux`_   | 2.7, 3.4-3.7 | PyQt5.3+, PySide2   |
+------------+--------------+---------------------+

For the description of *silx* dependencies, see the Dependencies_ section.

For all platforms, you can install *silx* with pip, see `Installing with pip`_.

To install *silx* in a `Virtual Environment`_, there is short version here-after
and  a longer description: :ref:`silx-venv`.

You can also install *silx* from the source, see `Installing from source`_.


Installing with pip
-------------------

To install silx (and all its dependencies_), run:

.. code-block:: bash

    pip install silx[full]

To install silx with a minimal set of dependencies, run:

.. code-block:: bash

    pip install silx

.. note::

    Use pip's ``--user`` option to install locally for the current user.

.. note::

    - If numpy is not yet installed, you might need to install it first.
    - Replace the ``pip`` command with ``pip3`` to install *silx* or any other library for Python 3.


Dependencies
------------

.. _dependencies:

The mandatory dependencies are:

- `numpy <http://www.numpy.org/>`_
- `h5py <http://docs.h5py.org/en/latest/build.html>`_
- `fabio <https://github.com/silx-kit/fabio>`_

The GUI widgets depend on the following extra packages:

* A Qt binding: either `PyQt5 <https://riverbankcomputing.com/software/pyqt/intro>`_,
  or `PySide2 <https://wiki.qt.io/Qt_for_Python>`_
* `matplotlib <http://matplotlib.org/>`_
* `PyOpenGL <http://pyopengl.sourceforge.net/>`_
* `qt_console <https://pypi.python.org/pypi/qtconsole>`_
  for the ``silx.gui.console`` widget.
* `dateutil <https://pypi.org/project/python-dateutil/>`_

Tools for reading and writing files depend on the following packages:

* `h5py <http://docs.h5py.org/en/latest/build.html>`_ for HDF5 files
* `fabio <https://github.com/silx-kit/fabio>`_ for multiple image formats

*silx.opencl* further depends on OpenCL and the following packages too :

* `pyopencl <https://mathema.tician.de/software/pyopencl/>`_
* `Mako <http://www.makotemplates.org/>`_

The complete list of dependencies with the minimal version is described in the
`requirement.txt <https://github.com/silx-kit/silx/blob/0.8/requirements.txt>`_
at the top level of the source package.

Build dependencies
++++++++++++++++++

In addition to run-time dependencies, building *silx* requires a C/C++ compiler,
`numpy <http://www.numpy.org/>`_ and `cython <http://cython.org>`_ (optional).

On Windows it is recommended to use Python 3.5, because with previous versions
of Python, it might be difficult to compile extensions (i.e. binary modules).

This project uses Cython (version > 0.21) to generate C files.
Cython is mandatory to build *silx* from the development branch.

The complete list of dependencies for building the package, including its
documentation, is described in the
`requirement-dev.txt <https://github.com/silx-kit/silx/blob/0.8/requirements-dev.txt>`_
at the top level of the source package.


Linux
-----

Packages are available for a few distributions:

- Debian 8: see `Installing a Debian package`_.
- `CentOS 7 RPM packages <http://pubrepo.maxiv.lu.se/rpm/el7/x86_64/>`_ provided by the Max IV institute at Lund, Sweden.
- `Fedora 23 rpm packages <http://pubrepo.maxiv.lu.se/rpm/fc23/x86_64/>`_ provided by the Max IV institute at Lund, Sweden.
- `Arch Linux (AUR) package <https://aur.archlinux.org/packages/python-silx>`_ provided by Leonid Bloch.

You can also follow one of those installation procedures:

- `Installing with pip`_
- Installing in a `Virtual Environment`_
- `Installing from source`_


Installing a Debian package
+++++++++++++++++++++++++++

Debian 8 (Jessie) packages are available on http://www.silx.org/pub/debian/ for amd64 computers.
To install it, you need to download this file :

.. code-block:: bash 

    http://www.silx.org/pub/debian/silx.list

and copy it into the /etc/apt/source.list.d folder.
Then run ``apt-get update`` and ``apt-get install python-silx``

.. code-block:: bash 

   wget http://www.silx.org/pub/debian/silx.list
   sudo cp silx.list /etc/apt/sources.list.d
   sudo apt-get update
   sudo apt-get install python-silx python3-silx silx

The `python-silx` and `python3-silx` packages provide the library,
while the `silx` package provides the executable (`silx view`, `silx convert` ...).

.. note::
    
    The packages are built automatically, hence not signed. 
    You have to accept the installation of non-signed packages.  

If the packages are not installed, it might be due to the priority list.
You can display the priority list using `apt-cache policy python-silx`.
If the Pin-number of silx.org is too low compared to other sources:
download http://www.silx.org/pub/debian/silx.pref into /etc/apt/preferences.d
and start the update/install procedure again.

    
Windows
-------

The simplest way of installing *silx* on Windows is to install it with ``pip``, see `Installing with pip`_::

    pip install silx[full]

This assumes you have Python and pip installed and configured.
If you don't, read the following sections.

Alternatively, you can check:

- Installing in a `Virtual Environment`_
- `Installing from source`_

Installing Python
+++++++++++++++++

Download and install Python from `python.org <https://www.python.org/downloads/>`_.

We recommend that you install the 64bit version of Python, which is not the
default version suggested on the Python website.
The 32bit version has limited memory, and also we don't provide a
binary wheel for it.
This means that you would have to install *silx* from its sources, which requires
you to install a C compiler first.

We also encourage you to use Python 3.5 or newer as former versions are no longer
officially supported.

Configure Python as explained on
`docs.python.org <https://docs.python.org/3/using/windows.html#configuring-python>`_
to add the python installation directory to your PATH environment variable.

Alternative Scientific Python stacks exists such as
`WinPython <http://winpython.github.io/>`_ or `Anaconda <https://www.anaconda.com/download/#windows>`_.
They all offer most of the scientific packages already installed which makes the
installation of dependencies much easier.

Using pip
+++++++++

Configure your PATH environment variable to include the pip installation
directory, the same way as described for Python.

The pip installation directory will likely be ``C:\Python35\Scripts\``.

Then you will be able to use all the pip commands listed below in a command
prompt.


MacOS
-----

While Apple ships Python 2.7 by default on their operating systems, we recommend
using Python 3.5 or newer to ease the installation of the Qt library.

Then, install *silx* with ``pip``, see `Installing with pip`_::

    pip install silx[full]

This should work without issues, as binary wheels of *silx* are provided on
PyPi.


Virtual Environment
-------------------

Virtual environments are self-contained directory trees that contain a Python
installation for a particular version of Python, plus a number of additional
packages.
They do not require administrator privileges, nor *root* access.

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
your virtual environment (see `Installing with pip`_).

.. code-block:: bash

    pip install silx[full]


Installing from source
----------------------

Building *silx* from the source requires some `Build dependencies`_ which may be
installed using:

.. code-block:: bash 

    pip install -r https://github.com/silx-kit/silx/raw/0.8/requirements-dev.txt


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

This is related to the two environment variables LC_ALL and LANG not being defined (or wrongly defined to UTF-8).
To set the environment variables, type on the command line:

.. code-block:: bash 

    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8

Advanced build options
++++++++++++++++++++++

In case you want more control over the build procedure, the build command is:

.. code-block:: bash 

    python setup.py build

There are few advanced options to ``setup.py build``:

* ``--no-cython``: Prevent Cython (even if installed) from re-generating the C source code.
  Use the one provided by the development team.
* ``--no-openmp``: Recompiles the Cython code without OpenMP support (default for MacOSX).
* ``--openmp``: Recompiles the Cython code with OpenMP support (default for Windows and Linux).

Package the build into a wheel and install it:

.. code-block:: bash 

    python setup.py bdist_wheel
    pip install dist/silx*.whl 

To build the documentation, using  `Sphinx <http://www.sphinx-doc.org/>`_:

.. code-block:: bash 

    python setup.py build build_doc


Testing
+++++++

To run the tests of an installed version of *silx*, run the following on the python interpreter:

.. code-block:: python
    
     import silx.test
     silx.test.run_tests()

To run the test suite of a development version, use the *run_tests.py* script at
the root of the source project.

.. code-block:: bash
    
     python ./run_tests.py
