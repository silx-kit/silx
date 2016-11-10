
.. _silx-venv:

Installing silx in a virtualenv
===============================

This step-by-step guide explains how to install *silx* in a virtualenv.


Prerequisites
-------------

This guide assumes that your system meets the following requirements:

   - a version of python compatible with *silx* is installed (python 2.7 or python >= 3.4)
   - the *pip* installer for python packages is installed
   - the Qt libraries is installed (optional, required for using ``silx.gui``)

Installation procedure
----------------------


Install vitrualenv
******************

.. code-block:: bash

    pip install virtualenv --user

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


Activate a virtualenv
*********************

A script is provided in your virtualenv to activate it.

.. code-block:: bash

    source silx_venv/bin/activate

After activating your new virtualenv, this python interpreter and the
package tools are used, instead of the ones from the system.

Any libraries you will install or upgrade will be inside the virtual
environment, and will not affect the rest of system.

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

.. TODO: Qt

If your python version is 3.5, installing PyQt5 and all required packages
is as simple as typing:

.. code-block:: bash

    pip install PyQt5

For previous versions of python, there are no wheels available, so the installation
is much more complicated.

If the Qt libraries are installed, you can install *pyqt* in your virtualenv.
This is optional, but none of the silx widgets will work if you don't have a python
binding for Qt.

You must start by installing SIP:

.. code-block:: bash

    hg clone http://www.riverbankcomputing.com/hg/sip
    cd sip

    python build.py prepare  # FIXME:  sh: 1: flex: not found
    python configure.py -d ~/venvs/silx_venv/lib/python2.7/site-packages
    make
    make install
    make clean



Download PyQt5 or PyQt4, depending on your Qt version.

For Qt 4, download the latest `PyQt4 <https://www.riverbankcomputing.com/software/pyqt/download>`_
tarball.

.. code-block:: bash

    wget http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.11.4/PyQt-x11-gpl-4.11.4.tar.gz

Alternatively, download the file using your web browser and save it in
your virtualenv folder.

.. code-block:: bash

    tar -xvf PyQt-x11-gpl-4.11.4.tar.gz
    cd PyQt-x11-gpl-4.11.4/

Now, configure the PyQt4 installer to install the library inside your virtualenv:

.. code-block:: bash

    python configure.py --destdir ~/venvs/silx_venv/lib/python2.7/site-packages
    make
    make install
    make clean

Install silx
************

.. code-block:: bash

    pip install silx


To test *silx*, open an interactive python console. If you managed to install PyQt or PySide
in your virtualenv, type:

.. code-block:: bash

    python

If you don't have Qt, use:

.. code-block:: bash

    WITH_QT_TEST=False python

.. FIXME: if pyqt works, remove WITH_QT_TEST=False

Run the test suite using:

    >>> import silx.test
    >>> silx.test.run_tests()






