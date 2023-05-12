
.. _silx-venv:

Installing silx in a virtualenv
===============================

This step-by-step guide explains how to install *silx* in a virtualenv.


Prerequisites
-------------

This guide assumes that your system meets the following requirements:

   - a version of python compatible with *silx* is installed
   - the *pip* installer for python packages is installed

Installation procedure
----------------------

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
    python -m venv silx_venv

A virtualenv contains a copy of your default python interpreter with a few tools
to install packages (pip, setuptools).

Virtual environments are created using a builtin standard library, ``venv`` (Python3 only):

.. code-block:: bash

    python3 -m venv /path/to/new/virtual/environment

.. note::

    On Debian platforms, you might need to install the ``python3-venv`` package.

    If you don't need to start with a clean environment and you don't want
    to install each required library one by one, you can use a command line
    option to create a virtualenv with access to all system packages:
    ``--system-site-packages``

To use a different python interpreter, use it to create the virtual environment.
For example, to use python 3.10:

.. code-block:: bash

    /usr/bin/python3.10 -m venv silx_venv


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

Install silx
************

To install silx with minimal dependencies, run:

.. code-block:: bash

    pip install silx

To install silx with all dependencies, run:

.. code-block:: bash

    pip install silx[full]

To test *silx*, open an interactive python console:

.. code-block:: bash

    python

If you don't have PyQt5, PySide6 or PyQt6, run:

.. code-block:: bash

    WITH_QT_TEST=False python

Run the test suite using:

    >>> import silx.test
    >>> silx.test.run_tests()
