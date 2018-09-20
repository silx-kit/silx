
.. _silx-venv:

Installing silx in a virtualenv
===============================

This step-by-step guide explains how to install *silx* in a virtualenv.


Prerequisites
-------------

This guide assumes that your system meets the following requirements:

   - a version of python compatible with *silx* is installed (python 2.7 or python >= 3.5)
   - the *pip* installer for python packages is installed
   - the Qt and PyQt libraries are installed (optional, required for using ``silx.gui``)

Installation procedure
----------------------


Install vitrualenv
******************

.. code-block:: bash

    pip install virtualenv --user

.. note::

    This step is not required for recent version of Python 3.
    Virtual environments are created using a builtin standard library,
    ``venv``.
    On Debian platforms, you might need to install the ``python3-venv``
    package.


Create a virtualenv
*******************

The files required by a virtual environment are created in a new folder
with the same name as the virtualenv. So make sure you are in a directory
in which you have write permissions.

In this tutorial we use a folder ``venvs`` in our home directory, and we create
a virtual environment named ``silx_venv``

.. code-block:: bash

    cd
    mkdir -p venvs
    cd venvs
    virtualenv silx_venv


A virtualenv contains a copy of your default python interpreter with a few tools
to install packages (pip, setuptools).

To use a different python interpreter, you can specify it on the command line.
For example, to use python 3.4:

.. code-block:: bash

    virtualenv -p /usr/bin/python3.4 silx_venv

But for python 3  you should use the builtin ``venv`` module:

.. code-block:: bash

    python3 -m venv /path/to/new/virtual/environment

.. note::

    If you don't need to start with a clean environment and you don't want
    to install each required library one by one, you can use a command line
    option to create a virtualenv with access to all system packages:
    ``--system-site-packages``


Activate a virtualenv
*********************

A script is provided in your virtualenv to activate it.

.. code-block:: bash

    source silx_venv/bin/activate

After activating your new virtualenv, this python interpreter and its
package tools are used, instead of the ones from the system.

Any libraries you will install or upgrade will be inside the virtual
environment, and will not affect the rest of system.

To deactivate the virtual environment, just type ``deactivate``.

Upgrade pip
***********

After activating *silx_venv*, you should upgrade *pip*:

.. code-block:: bash

    python -m pip install --upgrade pip


Upgrade setuptools and wheel
****************************

Upgrading the python packaging related libraries can make installing the
rest of the libraries much easier.

.. code-block:: bash

    pip install setuptools --upgrade
    pip install wheel --upgrade

Install build dependencies
**************************

The following command installs libraries that are required to build and
install *silx*:

.. code-block:: bash

    pip install numpy cython

.. since 0.5, numpy is now automatically installed when doing `pip install silx`

Install optional dependencies
*****************************

The following command installs libraries that are needed by various modules
of *silx*:

.. code-block:: bash

    pip install matplotlib fabio h5py

The next command installs libraries that are used by the python modules
handling parallel computing:

.. code-block:: bash

    pip install pyopencl mako


Install pyqt
************

If your python version is 3.5 or newer, installing PyQt5 and all required packages
is as simple as typing:

.. code-block:: bash

    pip install PyQt5

For previous versions of python, there are no PyQt wheels available, so the installation
is not as simple.

The simplest way, assuming that PyQt is installed on your system, is to use that
system package directly. For this, you need to add a symbolic link to your virtualenv.

If you want to use PyQt5 installed in ``/usr/lib/python2.7/dist-packages/``, type:

.. code-block:: bash

    ln -s /usr/lib/python2.7/dist-packages/PyQt5 silx_venv/lib/python2.7/site-packages/
    ln -s /usr/lib/python2.7/dist-packages/sip.so silx_venv/lib/python2.7/site-packages/


Install silx
************

.. code-block:: bash

    pip install silx


To test *silx*, open an interactive python console. If you managed to install PyQt5 or PySide2
in your virtualenv, type:

.. code-block:: bash

    python

If you don't have PyQt, use:

.. code-block:: bash

    WITH_QT_TEST=False python

Run the test suite using:

    >>> import silx.test
    >>> silx.test.run_tests()






